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

import socket
import struct
import time

# linux/uapi linux/socket.h: AF_SMC = 43(PF_SMC 同值);
# linux/uapi linux/in.h: IPPROTO_SMC = 265。
# 较新的 Python 3 自带这两个常量,老版本用数值兜底。
AF_SMC = getattr(socket, "AF_SMC", 43)
IPPROTO_SMC = getattr(socket, "IPPROTO_SMC", 265)

HEADER = struct.Struct("!4sQQQ")  # magic, seqno, payload_len, aux_len
MAGIC = b"LWD1"
HEADER_SIZE = HEADER.size  # 28


def open_socket(family: str) -> socket.socket:
    """按 family 建流式 socket:smc = AF_SMC(RDMA 数据面),tcp = 对照组。"""
    if family == "smc":
        sock = socket.socket(AF_SMC, socket.SOCK_STREAM, 0)
    elif family == "tcp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        # TCP 对照组关 Nagle 以公平对比帧延迟;SMC 数据面走 RDMA WRITE,
        # 无 Nagle 概念,保持默认。
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    else:
        raise ValueError(f"unknown family: {family!r} (expect 'smc' or 'tcp')")
    return sock


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
