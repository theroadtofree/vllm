# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""AF_SMC 双机 demo 公共层:地址族、帧协议、精确收发、链路状态检测。

对应边云数据面迁移方案(HCCL -> AF_SMC)见同目录
HCCL_TO_AF_SMC_MIGRATION.md,本模块的帧格式与正式方案保持一致:

    +---------+---------+--------------+---------+---------------+
    | magic   | seqno   | payload_len  | aux_len | payload + aux |
    | 4 bytes | uint64  | uint64       | uint64  | (字节连续)     |
    +---------+---------+--------------+---------+---------------+

头部定长 28 字节,网络字节序。payload 模拟 UP 通道主帧
(hidden states, bf16 [num_tokens, hidden_size]),aux 模拟
UP 通道附加帧(mrope positions, int64 [num_tokens, 3]),与
vllm_ascend lwd_comm 的 LwdCommRequest(tensor + aux_tensor)对应。

SMC/TCP 都是字节流:顺序天然保证,帧头 seqno 只做防御性校验
(错位立刻报配对错误,等价于 HCCL 方案里 send/recv 提交顺序契约)。

仅依赖标准库;只能在 Linux(>= 4.11,建议 6.6)双机上运行。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import socket
import struct
import time

# linux/uapi linux/socket.h: AF_SMC = 43(PF_SMC 同值);
# SMCPROTO_SMC = 0(Linux 内核 SMC 协议默认值;IPPROTO_SMC=265 是
# 另一个协议号,某些内核未注册时会 Errno 93,统一用 0 兜底)。
AF_SMC = getattr(socket, "AF_SMC", 43)
IPPROTO_SMC = getattr(socket, "IPPROTO_SMC", 265)  # 保留供参考,open_socket 不用

# Python socket 模块的 bind/connect 只会打包 AF_INET/AF_INET6/AF_UNIX
# 三种地址族,对 AF_SMC=43 走 default 分支抛 "bad family"(根本到不了
# 内核)。但内核 net/smc/af_smc.c 的 smc_bind/smc_connect 实际接受
# sockaddr_in(sa_family=AF_INET)与 sockaddr_in6。所以 AF_SMC socket
# 的 bind/connect 必须绕过 Python,直接调 libc,把 sockaddr_in 喂给内核。
# 用 try/except 包加载:非 Linux 上 import 本模块不会炸,实际调用时再报错。
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    _libc.bind.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _libc.bind.restype = ctypes.c_int
    _libc.connect.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _libc.connect.restype = ctypes.c_int
    _libc.accept.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    _libc.accept.restype = ctypes.c_int
except OSError:
    _libc = None  # 非 Linux 平台;AF_SMC 本就只支持 Linux,这里只是不阻断 import


def _pack_sockaddr_in(host: str, port: int) -> bytes:
    """struct sockaddr_in(16 字节):
        sa_family_t sin_family;   // uint16,native 字节序
        in_port_t   sin_port;     // uint16,网络序
        struct in_addr sin_addr; // uint32,网络序
        char        sin_zero[8];  // padding
    仅 IPv4;IPv6 待后续按需扩展(sockaddr_in6,28 字节)。

    字节序:sa_family 是 native(宿主)序,用 '='(native,无 padding);
    sin_port/sin_addr 是网络序,用 htons/inet_aton 转好后再按 native
    写出(x86 上 htons(port) 的字节落地就是网络序,等价 '<H',但
    big-endian 架构上 '<H' 会错,'=H' 跨架构都对)。"""
    return struct.pack(
        "=H H 4s 8s",
        socket.AF_INET,
        socket.htons(port),
        socket.inet_aton(host),
        b"\x00" * 8,
    )


def _raise_errno(err: int, call: str, addr: tuple = ()) -> None:
    # EINPROGRESS 在非阻塞 connect 是正常返回;本 demo 用阻塞 socket,
    # 不期望出现,但保险起见透传。
    raise OSError(err, f"{call}: {os.strerror(err)}", addr) from None

HEADER = struct.Struct("!4sQQQ")  # magic, seqno, payload_len, aux_len
MAGIC = b"LWD1"
HEADER_SIZE = HEADER.size  # 28


# SMC-R 的 RMBE 尺寸按 SO_SNDBUF/SO_RCVBUF 协商:默认 socket buffer
# (~208KB)会让 RMBE 取最小档,4MB 大帧被切成十几个小块、逐块等对端
# CDC 确认才能复用空间,ping-pong 拖垮吞吐(实测 SMC-R 反而比 TCP 慢)。
# 两端宿主需同步调大内核上限,否则 setsockopt 会被静默截断:
#   sysctl -w net.core.rmem_max=16777216 net.core.wmem_max=16777216
# TCP 对照组同样设置,保证对比公平(默认小 buffer 下 TCP 也吃亏)。
BUF_MB = 8


def _set_big_buffers(sock: socket.socket, mb: int = BUF_MB) -> None:
    nbytes = mb * 1024 * 1024
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, nbytes)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, nbytes)


def open_socket(family: str) -> socket.socket:
    """按 family 建流式 socket:smc = AF_SMC(RDMA 数据面),tcp = 对照组。"""
    if family == "smc":
        sock = socket.socket(AF_SMC, socket.SOCK_STREAM, 0)
        _set_big_buffers(sock)
    elif family == "tcp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        # TCP 对照组关 Nagle 以公平对比帧延迟;SMC 数据面走 RDMA WRITE,
        # 无 Nagle 概念,保持默认。
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _set_big_buffers(sock)
    else:
        raise ValueError(f"unknown family: {family!r} (expect 'smc' or 'tcp')")
    return sock


def bind_socket(sock: socket.socket, family: str, host: str, port: int) -> None:
    """统一 bind 入口:AF_SMC 走 libc 直传 sockaddr_in(Python 不认 AF_SMC
    地址);TCP 走原生 socket.bind。host 必须是 IPv4 字符串。"""
    if family == "smc":
        if _libc is None:
            raise RuntimeError("AF_SMC 需要 libc;非 Linux 平台无法 bind")
        addr = _pack_sockaddr_in(host, port)
        fd = sock.fileno()
        if _libc.bind(fd, addr, len(addr)) < 0:
            _raise_errno(ctypes.get_errno(), "bind", (host, port))
    else:
        sock.bind((host, port))


def connect_socket(sock: socket.socket, family: str, host: str, port: int) -> None:
    """统一 connect 入口:AF_SMC 走 libc 直传 sockaddr_in;TCP 走原生。
    阻塞 socket,connect 返回即建连完成(SMC 含 CLC 协商 + RDMA 链路建立)。"""
    if family == "smc":
        if _libc is None:
            raise RuntimeError("AF_SMC 需要 libc;非 Linux 平台无法 connect")
        addr = _pack_sockaddr_in(host, port)
        fd = sock.fileno()
        if _libc.connect(fd, addr, len(addr)) < 0:
            _raise_errno(ctypes.get_errno(), "connect", (host, port))
    else:
        sock.connect((host, port))


def _parse_sockaddr_in(buf: bytes) -> tuple:
    """从 sockaddr_in 字节解 (host, port);长度不足或非 AF_INET 返回占位。
    内核 SMC 的 getname 对外暴露 sockaddr_in(sa_family=AF_INET),
    port/addr 字段都是网络序。"""
    if len(buf) < 8:
        return ("?", 0)
    sa_family = int.from_bytes(buf[0:2], "little")  # native LE(x86) 足够
    if sa_family != socket.AF_INET:
        return (f"<family {sa_family}>", 0)
    port = struct.unpack("!H", buf[2:4])[0]
    host = socket.inet_ntoa(buf[4:8])
    return (host, port)


def accept_socket(srv: socket.socket, family: str) -> tuple[socket.socket, tuple]:
    """统一 accept 入口:AF_SMC 走 libc accept(Python socket.accept 内部
    先 getsockaddrlen 决定 peer 缓冲区长度,对 AF_SMC 抛 "bad family"),
    再用 socket.socket(fileno=) 把 fd 包回 Python socket 对象。TCP 走原生。

    返回 (conn, peer) 与原生 accept 一致;peer 为 (host, port)。
    AF_SMC socket 的 recv/send 不感知地址族,后续读写不受影响。"""
    if family == "smc":
        if _libc is None:
            raise RuntimeError("AF_SMC 需要 libc;非 Linux 平台无法 accept")
        # 128 字节足以容纳 sockaddr_in(16)/sockaddr_in6(28)/sockaddr_un(110)
        addrbuf = ctypes.create_string_buffer(128)
        addrlen = ctypes.c_uint(128)
        new_fd = _libc.accept(srv.fileno(), addrbuf, ctypes.byref(addrlen))
        if new_fd < 0:
            _raise_errno(ctypes.get_errno(), "accept")
        # socket.socket(fileno=...) 不调 socket() syscall,只是收养 fd;
        # 仍按 AF_SMC 标注 family,后续 getpeername 会再炸,但 demo 用不到。
        conn = socket.socket(AF_SMC, socket.SOCK_STREAM, 0, fileno=new_fd)
        peer = _parse_sockaddr_in(addrbuf.raw[: addrlen.value])
        return conn, peer
    else:
        return srv.accept()


def send_frame(sock: socket.socket, seqno: int, payload: bytes, aux: bytes = b"") -> None:
    """发送一帧:定长头 + payload + aux(单次 sendall,大帧由内核分片)。"""
    sock.sendall(HEADER.pack(MAGIC, seqno, len(payload), len(aux)) + payload + aux)


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """精确收 n 字节;对端关闭/中断立即报错(半帧错位等价于连接损坏)。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"peer closed with {n - len(buf)}/{n} bytes still expected"
            )
        buf += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[int, bytes, bytes]:
    """接收一帧,返回 (seqno, payload, aux)。"""
    magic, seqno, plen, alen = HEADER.unpack(recv_exactly(sock, HEADER_SIZE))
    if magic != MAGIC:
        raise ConnectionError(f"bad frame magic {magic!r}, stream misaligned")
    body = recv_exactly(sock, plen + alen)
    return seqno, body[:plen], body[plen:]


class FrameStats:
    """逐帧耗时统计(接收侧口径:一帧完整落地的墙钟)。"""

    def __init__(self) -> None:
        self.frame_ms: list[float] = []
        self.bytes_total = 0

    def add(self, nbytes: int, elapsed_s: float) -> None:
        self.frame_ms.append(elapsed_s * 1000.0)
        self.bytes_total += nbytes

    def summary(self, wall_s: float) -> str:
        ms = sorted(self.frame_ms)
        gbps = self.bytes_total * 8 / wall_s / 1e9 if wall_s > 0 else 0.0

        def pct(p: float) -> float:
            return ms[min(len(ms) - 1, int(len(ms) * p))]

        return (
            f"frames={len(ms)} bytes={self.bytes_total / 1e6:.1f}MB "
            f"wall={wall_s * 1000:.1f}ms "
            f"goodput={gbps:.2f}Gbps "
            f"frame_ms[p50={pct(0.50):.2f} p95={pct(0.95):.2f} "
            f"max={ms[-1]:.2f}]"
        )


def print_smc_state(tag: str) -> None:
    """Best-effort 确认 SMC 真的走上 RDMA(而不是静默降级回 TCP)。

    /proc/net/smc(内核 >= 4.19)列出 SMC-R 链路与连接;连接表非空
    且无 fallback 记录,说明本次传输走的是 RDMA 数据面。读不到时
    用 smc-tools 复查(见 README"确认没有降级"一节)。
    """
    try:
        with open("/proc/net/smc", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
    except OSError:
        print(f"[{tag}] /proc/net/smc 不可读,请用 `smc dbg` 或 "
              f"`dmesg | grep -i smc` 人工确认链路状态")
        return
    head = lines[0] if lines else "(empty)"
    conns = len(lines) - 1
    print(f"[{tag}] /proc/net/smc: header={head!r} entries={conns}")
    for ln in lines[1:4]:
        print(f"[{tag}]   {ln}")
    if conns == 0:
        print(f"[{tag}] 注意:没有活跃 SMC 连接,本次传输大概率降级为 TCP "
              f"(原因排查见 README)")


def now() -> float:
    return time.perf_counter()
