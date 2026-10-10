# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""verbs 传输 demo:libibverbs(ctypes) RC SEND,与正式实现 verbs_wire.py
同款模型,双机验证 verbs 数据面的连通性与吞吐(无 NPU 依赖)。

与 smc/tcp 模式的关系(--family verbs 时启用):
  * 帧协议同款(28B magic/seqno/payload_len/aux_len,见 smc_common);
  * 传输层换成 RC QP:一帧一次 post_send,接收一帧一个 CQE,无逐块
    确认模型——拉远高 RTT 场景不受 SMC-R 的 RMBE/RTT 窗口上限
    (实测 500km/5ms 下 ~0.8Gbps)约束,同链路 ib_send_bw ~9.4Gbps;
  * 握手复用一条小 TCP 连接,交换 lid/qpn/psn/gid/gid_index/max_frame
    (消息格式与 verbs_wire.py 完全一致);
  * 块池:N_BLOCKS × BLOCK_MB 注册内存(bytearray + ibv_reg_mr,
    注册即 pin,无需 torch),发送直写注册块单 sge 单次 SEND,
    接收直落预挂块——每方向比 SMC 少一次内核拷贝。

用法(与 smc/tcp 模式同参数):
  云侧: python3 smc_server.py --family verbs --host <云IP> --port 19901
  边侧: python3 smc_client.py --family verbs --host <云IP> --port 19901
--port 即握手 TCP 端口(正式实现的握手用 base+2/+3,demo 单通道
直接占用该端口)。

env 可调:VERBS_BLOCK_MB(默认 64)、VERBS_N_BLOCKS(默认 4)、
VERBS_GID_INDEX(自动探测不准时手动指定,RoCE v2)、VERBS_DEV
(多网卡指定设备名)。

依赖:libibverbs.so.1(能跑 ib_write_bw 的机器即具备)。结构体/
枚举对应内核 uapi infiniband/verbs.h;若 modify_qp 报 EINVAL(22),
对照 /usr/include/infiniband/verbs.h 的 ibv_qp_attr_mask 核对常量。
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import struct
import time
import zlib

from smc_common import (
    FrameStats,
    HEADER,
    MAGIC,
    accept_socket,
    bind_socket,
    connect_socket,
    now,
    open_socket,
)

# --------------------------------------------------------------------- #
# 配置                                                                  #
# --------------------------------------------------------------------- #
_BLOCK_BYTES = int(os.getenv("VERBS_BLOCK_MB", "64")) << 20
_N_BLOCKS = max(2, int(os.getenv("VERBS_N_BLOCKS", "4")))
_GID_INDEX_ENV = int(os.getenv("VERBS_GID_INDEX", "-1"))
_DEVICE_ENV = os.getenv("VERBS_DEV", "")

# --------------------------------------------------------------------- #
# libibverbs ctypes 绑定(与 verbs_wire.py 保持一致)                     #
# --------------------------------------------------------------------- #
try:
    _ibv = ctypes.CDLL("libibverbs.so.1", use_errno=False)
except OSError:
    _ibv = None

_IBV_QPT_RC = 2
_IBV_QPS_INIT = 1
_IBV_QPS_RTR = 2
_IBV_QPS_RTS = 3
_IBV_MTU_4096 = 4
_IBV_ACCESS_LOCAL_WRITE = 1
_IBV_ACCESS_REMOTE_WRITE = 2
_IBV_WR_SEND = 0
_IBV_WC_SUCCESS = 0
_IBV_WC_RECV = 0x80
_IBV_SEND_SIGNALED = 1
_IBV_LINK_LAYER_ETHERNET = 2

# ibv_qp_attr_mask 位序(linux uapi verbs.h)
_QP_STATE = 1 << 0
_QP_PKEY_INDEX = 1 << 3
_QP_PORT = 1 << 4
_QP_RQ_PSN = 1 << 6
_QP_SQ_PSN = 1 << 7
_QP_DEST_QP_NUM = 1 << 8
_QP_ACCESS_FLAGS = 1 << 9
_QP_PATH_MTU = 1 << 10
_QP_TIMEOUT = 1 << 11
_QP_RETRY_CNT = 1 << 12
_QP_RNR_RETRY = 1 << 13
_QP_MAX_QP_RD_ATOMIC = 1 << 14
_QP_MAX_DEST_RD_ATOMIC = 1 << 15
_QP_MIN_RNR_TIMER = 1 << 16
_QP_AV = 1 << 17

_QP_TIMEOUT_VAL = 16
_QP_RETRY_CNT_VAL = 7
_QP_RNR_RETRY_VAL = 7
_QP_MIN_RNR_TIMER_VAL = 16
_QP_MAX_RD_ATOMIC_VAL = 1


class _Sge(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("length", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
    ]


class _SendWr(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("next", ctypes.c_void_p),
        ("sg_list", ctypes.POINTER(_Sge)),
        ("num_sge", ctypes.c_int),
        ("opcode", ctypes.c_int),
        ("send_flags", ctypes.c_int),
        ("imm_data", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class _RecvWr(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("next", ctypes.c_void_p),
        ("sg_list", ctypes.POINTER(_Sge)),
        ("num_sge", ctypes.c_int),
    ]


class _Wc(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("status", ctypes.c_int),
        ("opcode", ctypes.c_int),
        ("vendor_err", ctypes.c_uint32),
        ("byte_len", ctypes.c_uint32),
        ("imm_data", ctypes.c_uint32),
        ("qp_num", ctypes.c_uint32),
        ("src_qp", ctypes.c_uint32),
        ("wc_flags", ctypes.c_int),
        ("pkey_index", ctypes.c_uint16),
        ("slid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("dlid_path_bits", ctypes.c_uint8),
    ]


class _QpCap(ctypes.Structure):
    _fields_ = [
        ("max_send_wr", ctypes.c_uint32),
        ("max_recv_wr", ctypes.c_uint32),
        ("max_send_sge", ctypes.c_uint32),
        ("max_recv_sge", ctypes.c_uint32),
        ("max_inline_data", ctypes.c_uint32),
    ]


class _QpInitAttr(ctypes.Structure):
    _fields_ = [
        ("qp_context", ctypes.c_void_p),
        ("send_cq", ctypes.c_void_p),
        ("recv_cq", ctypes.c_void_p),
        ("srq", ctypes.c_void_p),
        ("cap", _QpCap),
        ("qp_type", ctypes.c_int),
        ("sq_sig_all", ctypes.c_int),
    ]


class _GlobalRoute(ctypes.Structure):
    _fields_ = [
        ("dgid", ctypes.c_uint8 * 16),
        ("flow_label", ctypes.c_uint32),
        ("sgid_index", ctypes.c_uint8),
        ("hop_limit", ctypes.c_uint8),
        ("traffic_class", ctypes.c_uint8),
    ]


class _AhAttr(ctypes.Structure):
    _fields_ = [
        ("dlid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("src_path_bits", ctypes.c_uint8),
        ("static_rate", ctypes.c_uint8),
        ("is_global", ctypes.c_uint8),
        ("port_num", ctypes.c_uint8),
        ("grh", _GlobalRoute),
    ]


class _QpAttr(ctypes.Structure):
    _fields_ = [
        ("qp_state", ctypes.c_int),
        ("cur_qp_state", ctypes.c_int),
        ("path_mtu", ctypes.c_uint32),
        ("path_mig_state", ctypes.c_int),
        ("qkey", ctypes.c_uint32),
        ("rq_psn", ctypes.c_uint32),
        ("sq_psn", ctypes.c_uint32),
        ("dest_qp_num", ctypes.c_uint32),
        ("qp_access_flags", ctypes.c_uint16),
        ("pkey_index", ctypes.c_uint16),
        ("alt_pkey_index", ctypes.c_uint16),
        ("en_sqd_async_notify", ctypes.c_uint8),
        ("sq_draining", ctypes.c_uint8),
        ("max_rd_atomic", ctypes.c_uint8),
        ("max_dest_rd_atomic", ctypes.c_uint8),
        ("min_rnr_timer", ctypes.c_uint32),
        ("port_num", ctypes.c_uint32),
        ("timeout", ctypes.c_uint16),
        ("retry_cnt", ctypes.c_uint16),
        ("rnr_retry", ctypes.c_uint16),
        ("alt_port_num", ctypes.c_uint16),
        ("alt_timeout", ctypes.c_uint16),
        ("ah_attr", _AhAttr),
        ("alt_ah_attr", _AhAttr),
    ]


class _PortAttr(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int),
        ("max_mtu", ctypes.c_uint32),
        ("active_mtu", ctypes.c_uint32),
        ("gid_tbl_len", ctypes.c_int),
        ("port_cap_flags", ctypes.c_uint32),
        ("max_msg_sz", ctypes.c_uint32),
        ("bad_pkey_cntr", ctypes.c_uint32),
        ("qkey_viol_cntr", ctypes.c_uint32),
        ("pkey_tbl_len", ctypes.c_uint16),
        ("lid", ctypes.c_uint16),
        ("sm_lid", ctypes.c_uint16),
        ("lmc", ctypes.c_uint8),
        ("max_vl_num", ctypes.c_uint8),
        ("sm_sl", ctypes.c_uint8),
        ("subnet_timeout", ctypes.c_uint8),
        ("init_type_replay", ctypes.c_uint8),
        ("active_width", ctypes.c_uint8),
        ("active_speed", ctypes.c_uint8),
        ("phys_state", ctypes.c_uint8),
        ("link_layer", ctypes.c_uint8),
    ]


class _Mr(ctypes.Structure):
    _fields_ = [
        ("context", ctypes.c_void_p),
        ("pd", ctypes.c_void_p),
        ("addr", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("handle", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
        ("rkey", ctypes.c_uint32),
    ]


def _bind_ibv() -> None:
    if _ibv is None:
        return
    _ibv.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    _ibv.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    _ibv.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    _ibv.ibv_get_device_name.restype = ctypes.c_char_p
    _ibv.ibv_open_device.argtypes = [ctypes.c_void_p]
    _ibv.ibv_open_device.restype = ctypes.c_void_p
    _ibv.ibv_query_port.argtypes = [
        ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(_PortAttr),
    ]
    _ibv.ibv_query_port.restype = ctypes.c_int
    _ibv.ibv_query_gid.argtypes = [
        ctypes.c_void_p, ctypes.c_uint8, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8 * 16),
    ]
    _ibv.ibv_query_gid.restype = ctypes.c_int
    _ibv.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
    _ibv.ibv_alloc_pd.restype = ctypes.c_void_p
    _ibv.ibv_create_cq.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
    ]
    _ibv.ibv_create_cq.restype = ctypes.c_void_p
    _ibv.ibv_create_qp.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_QpInitAttr),
    ]
    _ibv.ibv_create_qp.restype = ctypes.c_void_p
    _ibv.ibv_modify_qp.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_QpAttr), ctypes.c_int,
    ]
    _ibv.ibv_modify_qp.restype = ctypes.c_int
    _ibv.ibv_reg_mr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ]
    _ibv.ibv_reg_mr.restype = ctypes.POINTER(_Mr)
    _ibv.ibv_post_send.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_SendWr),
        ctypes.POINTER(ctypes.POINTER(_SendWr)),
    ]
    _ibv.ibv_post_send.restype = ctypes.c_int
    _ibv.ibv_post_recv.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_RecvWr),
        ctypes.POINTER(ctypes.POINTER(_RecvWr)),
    ]
    _ibv.ibv_post_recv.restype = ctypes.c_int
    _ibv.ibv_poll_cq.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_Wc),
    ]
    _ibv.ibv_poll_cq.restype = ctypes.c_int
    _ibv.ibv_destroy_qp.argtypes = [ctypes.c_void_p]
    _ibv.ibv_destroy_cq.argtypes = [ctypes.c_void_p]
    _ibv.ibv_dereg_mr.argtypes = [ctypes.POINTER(_Mr)]
    _ibv.ibv_dealloc_pd.argtypes = [ctypes.c_void_p]
    _ibv.ibv_close_device.argtypes = [ctypes.c_void_p]
    _ibv.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]


_bind_ibv()

_HS_MAGIC = b"LWDV1\0\0\0"
_HS_FMT = "!8sIIII16sI"  # magic, lid, qpn, psn, mtu, gid, max_frame
_HS_SIZE = struct.calcsize(_HS_FMT)


class _Rc:
    """一条 RC QP 及其本地资源(demo 版,单线程使用,无 IO 线程)。"""

    def __init__(self) -> None:
        if _ibv is None:
            raise RuntimeError("libibverbs.so.1 not found")
        self._ctx = None
        self._pd = None
        self._cq = None
        self.qp = None
        self.qpn = 0
        self.psn = 0
        self.lid = 0
        self.port = 1
        self.gid = bytes(16)
        self.gid_index = 0
        self.mtu = _IBV_MTU_4096
        self._open_and_init_qp()

    def _open_and_init_qp(self) -> None:
        n = ctypes.c_int(0)
        dev_list = _ibv.ibv_get_device_list(ctypes.byref(n))
        if not dev_list or n.value <= 0:
            raise RuntimeError("no RDMA devices (ibv_get_device_list empty)")
        chosen = None
        try:
            for i in range(n.value):
                dev = dev_list[i]
                if not dev:
                    continue
                name = _ibv.ibv_get_device_name(dev) or b""
                if _DEVICE_ENV and _DEVICE_ENV.encode() not in name:
                    continue
                ctx = _ibv.ibv_open_device(dev)
                if not ctx:
                    continue
                pa = _PortAttr()
                if _ibv.ibv_query_port(ctx, 1, ctypes.byref(pa)) == 0 \
                        and pa.link_layer == _IBV_LINK_LAYER_ETHERNET:
                    chosen = (ctx, name.decode(errors="replace"))
                    break
                _ibv.ibv_close_device(ctx)
            if chosen is None:
                raise RuntimeError(
                    f"no active Ethernet(RoCE) RDMA port found (dev filter "
                    f"{_DEVICE_ENV!r})" if _DEVICE_ENV else
                    "no active Ethernet(RoCE) RDMA port found"
                )
        finally:
            _ibv.ibv_free_device_list(dev_list)
        self._ctx, self.dev_name = chosen
        self._pd = _ibv.ibv_alloc_pd(self._ctx)
        if not self._pd:
            raise RuntimeError("ibv_alloc_pd failed")
        self._cq = _ibv.ibv_create_cq(self._ctx, 64, None, None, 0)
        if not self._cq:
            raise RuntimeError("ibv_create_cq failed")
        pa = _PortAttr()
        if _ibv.ibv_query_port(self._ctx, self.port, ctypes.byref(pa)):
            raise RuntimeError("ibv_query_port failed")
        self.lid = pa.lid
        self._resolve_gid()
        attr = _QpInitAttr()
        attr.send_cq = ctypes.c_void_p(self._cq)
        attr.recv_cq = ctypes.c_void_p(self._cq)
        attr.cap.max_send_wr = _N_BLOCKS * 2
        attr.cap.max_recv_wr = _N_BLOCKS * 2
        attr.cap.max_send_sge = 1
        attr.cap.max_recv_sge = 1
        attr.qp_type = _IBV_QPT_RC
        attr.sq_sig_all = 1
        self.qp = _ibv.ibv_create_qp(ctypes.c_void_p(self._pd),
                                     ctypes.byref(attr))
        if not self.qp:
            raise RuntimeError("ibv_create_qp failed")
        # struct ibv_qp 头部 {ctx, pd, handle(u32), qp_num(u32)}:
        # qp_num 位于 offset 20
        qp_words = ctypes.cast(ctypes.c_void_p(self.qp),
                               ctypes.POINTER(ctypes.c_uint32))
        self.qpn = int(qp_words[5])
        self.psn = 0
        self._to_init()

    def _resolve_gid(self) -> None:
        pa = _PortAttr()
        _ibv.ibv_query_port(self._ctx, self.port, ctypes.byref(pa))
        n_gids = max(pa.gid_tbl_len, 1)
        if _GID_INDEX_ENV >= 0:
            self.gid_index = _GID_INDEX_ENV
        else:
            self.gid_index = -1
            for idx in range(min(n_gids, 32)):
                gid = (ctypes.c_uint8 * 16)()
                if _ibv.ibv_query_gid(self._ctx, self.port, idx,
                                      ctypes.byref(gid)) != 0:
                    continue
                raw = bytes(gid)
                # IPv4-mapped(::ffff:a.b.c.d)= RoCE v2 gid 判据
                if raw[:10] == b"\x00" * 10 and raw[10] == 0xFF \
                        and raw[11] == 0xFF:
                    self.gid_index = idx
                    break
            if self.gid_index < 0:
                self.gid_index = 0
                print(f"[verbs] warning: no IPv4-mapped (RoCE v2) gid on "
                      f"{self.dev_name}; using gid index 0 — set "
                      f"VERBS_GID_INDEX if wrong")
        gid = (ctypes.c_uint8 * 16)()
        if _ibv.ibv_query_gid(self._ctx, self.port, self.gid_index,
                              ctypes.byref(gid)) != 0:
            raise RuntimeError("ibv_query_gid failed")
        self.gid = bytes(gid)

    def _modify(self, attr: _QpAttr, mask: int, what: str) -> None:
        rc = _ibv.ibv_modify_qp(ctypes.c_void_p(self.qp),
                                ctypes.byref(attr), mask)
        if rc != 0:
            raise RuntimeError(
                f"ibv_modify_qp({what}) failed errno={rc};若 rc==22 请对照"
                f"/usr/include/infiniband/verbs.h 核对 QP attr 常量"
            )

    def _to_init(self) -> None:
        attr = _QpAttr()
        attr.qp_state = _IBV_QPS_INIT
        attr.pkey_index = 0
        attr.port_num = self.port
        attr.qp_access_flags = (_IBV_ACCESS_LOCAL_WRITE
                                | _IBV_ACCESS_REMOTE_WRITE)
        self._modify(attr,
                     _QP_STATE | _QP_PKEY_INDEX | _QP_PORT | _QP_ACCESS_FLAGS,
                     "RESET->INIT")

    def transition_rtr(self, qpn, psn, mtu, gid: bytes) -> None:
        attr = _QpAttr()
        attr.qp_state = _IBV_QPS_RTR
        attr.path_mtu = min(self.mtu, mtu)
        attr.dest_qp_num = qpn
        attr.rq_psn = psn
        attr.max_dest_rd_atomic = _QP_MAX_RD_ATOMIC_VAL
        attr.min_rnr_timer = _QP_MIN_RNR_TIMER_VAL
        attr.ah_attr.dlid = 0xFFFF
        attr.ah_attr.is_global = 1
        attr.ah_attr.port_num = self.port
        for i, b in enumerate(gid):
            attr.ah_attr.grh.dgid[i] = b
        attr.ah_attr.grh.sgid_index = self.gid_index
        attr.ah_attr.grh.hop_limit = 1
        self._modify(
            attr,
            _QP_STATE | _QP_PATH_MTU | _QP_DEST_QP_NUM | _QP_RQ_PSN
            | _QP_MAX_DEST_RD_ATOMIC | _QP_MIN_RNR_TIMER | _QP_AV,
            "INIT->RTR",
        )

    def transition_rts(self) -> None:
        attr = _QpAttr()
        attr.qp_state = _IBV_QPS_RTS
        attr.sq_psn = self.psn
        attr.timeout = _QP_TIMEOUT_VAL
        attr.retry_cnt = _QP_RETRY_CNT_VAL
        attr.rnr_retry = _QP_RNR_RETRY_VAL
        attr.max_rd_atomic = _QP_MAX_RD_ATOMIC_VAL
        self._modify(
            attr,
            _QP_STATE | _QP_SQ_PSN | _QP_TIMEOUT | _QP_RETRY_CNT
            | _QP_RNR_RETRY | _QP_MAX_QP_RD_ATOMIC,
            "RTR->RTS",
        )

    def reg_mr(self, buf: bytearray) -> int:
        addr = (ctypes.c_char * len(buf)).from_buffer(buf)
        mr = _ibv.ibv_reg_mr(
            ctypes.c_void_p(self._pd),
            ctypes.cast(addr, ctypes.c_void_p), len(buf),
            _IBV_ACCESS_LOCAL_WRITE | _IBV_ACCESS_REMOTE_WRITE,
        )
        if not mr:
            raise RuntimeError(
                f"ibv_reg_mr({len(buf)} bytes) failed — 块池过大?调小 "
                f"VERBS_BLOCK_MB/VERBS_N_BLOCKS"
            )
        return mr.contents.lkey

    def post_send(self, wr: _SendWr) -> None:
        bad = ctypes.POINTER(_SendWr)()
        if _ibv.ibv_post_send(ctypes.c_void_p(self.qp), ctypes.byref(wr),
                              ctypes.byref(bad)) != 0:
            raise RuntimeError("ibv_post_send failed")

    def post_recv(self, wr: _RecvWr) -> None:
        bad = ctypes.POINTER(_RecvWr)()
        if _ibv.ibv_post_recv(ctypes.c_void_p(self.qp), ctypes.byref(wr),
                              ctypes.byref(bad)) != 0:
            raise RuntimeError("ibv_post_recv failed")

    def poll(self, n: int) -> list[_Wc]:
        wcs = (_Wc * n)()
        got = _ibv.ibv_poll_cq(ctypes.c_void_p(self._cq), n, wcs)
        return [wcs[i] for i in range(max(0, got))]

    def destroy(self) -> None:
        if self.qp:
            _ibv.ibv_destroy_qp(ctypes.c_void_p(self.qp))
            self.qp = None
        if self._cq:
            _ibv.ibv_destroy_cq(ctypes.c_void_p(self._cq))
            self._cq = None
        if self._pd:
            _ibv.ibv_dealloc_pd(ctypes.c_void_p(self._pd))
            self._pd = None
        if self._ctx:
            _ibv.ibv_close_device(ctypes.c_void_p(self._ctx))
            self._ctx = None


def _hs_pack(rc: _Rc) -> bytes:
    return struct.pack(_HS_FMT, _HS_MAGIC, rc.lid, rc.qpn, rc.psn, rc.mtu,
                       rc.gid, _BLOCK_BYTES)


def _hs_unpack(data: bytes) -> tuple:
    magic, _lid, qpn, psn, mtu, gid, max_frame = struct.unpack(_HS_FMT, data)
    if magic != _HS_MAGIC:
        raise ConnectionError(f"bad handshake magic {magic!r}")
    if max_frame < _BLOCK_BYTES:
        raise ConnectionError(
            f"peer block {max_frame} < local {_BLOCK_BYTES};对齐两端 "
            f"VERBS_BLOCK_MB"
        )
    return qpn, psn, mtu, bytes(gid)


def _hs_xchg(sock, rc: _Rc) -> tuple:
    """TCP 握手交换 QP 标识(双方同时发、再收,定长小消息无死锁),
    完成后关闭 TCP。"""
    sock.sendall(_hs_pack(rc))
    peer = _hs_unpack(_recv_exact(sock, _HS_SIZE))
    sock.close()
    rc.transition_rtr(*peer)
    rc.transition_rts()
    return peer


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed during handshake")
        buf += chunk
    return bytes(buf)


class _Block:
    __slots__ = ("buf", "addr", "lkey", "index", "sge", "wr", "recv_sge",
                 "recv_wr")

    def __init__(self, rc: _Rc, index: int) -> None:
        self.buf = bytearray(_BLOCK_BYTES)
        self.addr = ctypes.addressof(
            (ctypes.c_char * _BLOCK_BYTES).from_buffer(self.buf))
        self.lkey = rc.reg_mr(self.buf)
        self.index = index
        self.sge = None
        self.wr = None
        self.recv_sge = None
        self.recv_wr = None


class VerbsEndpoint:
    """单线程 demo endpoint:握手建 QP + 块池收发。

    send_frame 流水线:post 后即返回,块由 poll 回收(块池耗尽时阻塞
    回收);recv_frame 阻塞至 RECV CQE。RC 全双工,ACK 用同一条 QP
    反向 SEND。

    块生命周期约束:recv_frame 返回的零拷贝 memoryview 指向接收块,
    该块在*下一次* recv_frame 调用时才重挂 post_recv——调用方必须在
    两次 recv 之间消费完 view(demo 的 crc/json 都是即取即用)。"""

    def __init__(self, rc: _Rc) -> None:
        self.rc = rc
        # send/recv 分池:发送块只在 CQE 后回池,接收块只挂 recv WQE
        # ——共池时全部预挂 recv 后发送 ACK 会因无空闲块死锁
        self._send_blocks = [_Block(rc, i) for i in range(_N_BLOCKS)]
        self._recv_blocks = [
            _Block(rc, _N_BLOCKS + i) for i in range(_N_BLOCKS)]
        self._blocks = self._send_blocks + self._recv_blocks  # wr_id 索引
        self._free: list[_Block] = list(self._send_blocks)
        self._posted: dict[int, _Block] = {}    # 发送在途
        self._recv_pending: dict[int, _Block] = {}  # 已挂 recv WQE
        self._consumed: list[_Block] = []       # 已收待重挂(等下次 recv)
        self._recv_done: list[tuple[_Block, int]] = []  # (块, 帧长)完成
        self._inflight = 0

    # ---------------- 握手入口 ---------------- #

    @classmethod
    def connect(cls, host: str, port: int) -> "VerbsEndpoint":
        rc = _Rc()
        try:
            sock = open_socket("tcp")
            connect_socket(sock, "tcp", host, port)
            _hs_xchg(sock, rc)
        except Exception:
            rc.destroy()
            raise
        ep = cls(rc)
        ep._post_all_free_as_recv()
        return ep

    @classmethod
    def serve(cls, host: str, port: int) -> tuple["VerbsEndpoint", str]:
        srv = open_socket("tcp")
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bind_socket(srv, "tcp", host, port)
        srv.listen(1)
        print(f"[cloud] verbs handshake listening on {host}:{port}")
        conn, peer = accept_socket(srv, "tcp")
        srv.close()
        rc = _Rc()
        try:
            _hs_xchg(conn, rc)
        except Exception:
            rc.destroy()
            raise
        ep = cls(rc)
        ep._post_all_free_as_recv()
        return ep, f"{peer[0]}:{peer[1]}"

    # ---------------- 块管理 ---------------- #

    def _post_all_free_as_recv(self) -> None:
        """接收池全部预挂 recv(池大小=接收在途上限,对端发送侧受其
        发送池限制,两侧数量一致则不会 RNR)。"""
        for blk in self._recv_blocks:
            self._post_recv(blk)

    def _post_recv(self, blk: _Block) -> None:
        sge = _Sge(blk.addr, _BLOCK_BYTES, blk.lkey)
        wr = _RecvWr()
        wr.wr_id = blk.index
        wr.sg_list = ctypes.pointer(sge)
        wr.num_sge = 1
        blk.recv_sge = sge
        blk.recv_wr = wr
        self.rc.post_recv(wr)
        self._recv_pending[blk.index] = blk

    def _repost_consumed(self) -> None:
        while self._consumed:
            self._post_recv(self._consumed.pop())

    # ---------------- 统一 CQE 处理 ---------------- #

    def _poll_once(self) -> None:
        for wc in self.rc.poll(_N_BLOCKS * 2):
            if wc.status != _IBV_WC_SUCCESS:
                raise ConnectionError(
                    f"verbs CQE status={wc.status} "
                    f"vendor={wc.vendor_err}")
            blk = self._blocks[wc.wr_id]
            if wc.opcode == _IBV_WC_RECV:
                if blk.index not in self._recv_pending:
                    raise ConnectionError("unexpected recv CQE")
                del self._recv_pending[blk.index]
                self._recv_done.append((blk, wc.byte_len))
            else:  # SEND 完成(RC:对端已 ACK)
                done = self._posted.pop(blk.index, None)
                if done is not None:
                    done.sge = None
                    done.wr = None
                    self._free.append(done)
                    self._inflight -= 1

    # ---------------- 收发 ---------------- #

    def send_frame(self, seqno: int, payload: bytes, aux: bytes = b"") -> None:
        total = HEADER.size + len(payload) + len(aux)
        if total > _BLOCK_BYTES:
            raise RuntimeError(
                f"frame {total} > block {_BLOCK_BYTES};调大 VERBS_BLOCK_MB")
        blk = self._acquire_block()
        blk.buf[: HEADER.size] = HEADER.pack(
            MAGIC, seqno, len(payload), len(aux))
        blk.buf[HEADER.size: HEADER.size + len(payload)] = payload
        if aux:
            blk.buf[HEADER.size + len(payload): total] = aux
        sge = _Sge(blk.addr, total, blk.lkey)
        wr = _SendWr()
        wr.wr_id = blk.index
        wr.sg_list = ctypes.pointer(sge)
        wr.num_sge = 1
        wr.opcode = _IBV_WR_SEND
        wr.send_flags = _IBV_SEND_SIGNALED
        blk.sge = sge
        blk.wr = wr
        self._posted[blk.index] = blk
        self._inflight += 1
        self.rc.post_send(wr)

    def _acquire_block(self) -> _Block:
        deadline = time.monotonic() + 10.0
        while not self._free:
            if time.monotonic() > deadline:
                raise TimeoutError("verbs send completion timeout")
            self._poll_once()
        return self._free.pop()

    def recv_frame(
        self, timeout_s: float = 60.0,
    ) -> tuple[int, memoryview, memoryview]:
        """阻塞收一帧,返回 (seqno, payload, aux) 的零拷贝视图。"""
        # 上一次返回的 view 已被调用方消费,此时才重挂其块
        self._repost_consumed()
        deadline = time.monotonic() + timeout_s
        while not self._recv_done:
            if time.monotonic() > deadline:
                raise TimeoutError("verbs recv timeout")
            self._poll_once()
            if not self._recv_done:
                time.sleep(0.0002)
        blk, byte_len = self._recv_done.pop(0)
        frame = memoryview(blk.buf)[:byte_len]
        magic, seqno, plen, alen = HEADER.unpack_from(frame)
        if magic != MAGIC or HEADER.size + plen + alen != byte_len:
            raise ConnectionError(
                f"bad frame magic={magic!r} len={byte_len}")
        result = (seqno, frame[HEADER.size: HEADER.size + plen],
                  frame[HEADER.size + plen:])
        self._consumed.append(blk)  # 下次 recv_frame 时重挂
        return result

    def drain(self) -> None:
        """等全部发送 CQE(收 ACK 前确保数据完整到达对端)。"""
        deadline = time.monotonic() + 30.0
        while self._inflight > 0:
            if time.monotonic() > deadline:
                raise TimeoutError("verbs drain timeout")
            self._poll_once()

    def close(self) -> None:
        self.rc.destroy()


# --------------------------------------------------------------------- #
# client / server runner(与 smc_client/smc_server 的参数与输出对齐)    #
# --------------------------------------------------------------------- #

def run_client(args) -> int:
    payload = os.urandom(args.rows * args.hidden * 2)
    aux = os.urandom(args.rows * 3 * 8)
    frame_bytes = len(payload) + len(aux)
    expected_crc = zlib.crc32(payload)

    t0 = now()
    ep = VerbsEndpoint.connect(args.host, args.port)
    print(f"[edge] family=verbs connected to {args.host}:{args.port} "
          f"(dev={ep.rc.dev_name} qpn={ep.rc.qpn} "
          f"gid_index={ep.rc.gid_index}) in {(now() - t0) * 1000:.1f}ms; "
          f"frame={frame_bytes / 1e6:.2f}MB frames={args.frames} "
          f"blocks={_N_BLOCKS}x{_BLOCK_BYTES >> 20}MiB")

    send_stats = FrameStats()
    t_start = now()
    for seqno in range(args.frames):
        t0 = now()
        ep.send_frame(seqno, payload, aux)
        send_stats.add(frame_bytes, now() - t0)
        if (seqno + 1) % 20 == 0:
            print(f"[edge] sent {seqno + 1}/{args.frames}")
        if args.interval_ms > 0:
            time.sleep(args.interval_ms / 1000.0)

    ep.drain()
    ack_seqno, ack_body, _ = ep.recv_frame()
    ack = json.loads(bytes(ack_body))
    e2e = now() - t_start
    if ack_seqno != args.frames or len(ack) != args.frames:
        print(f"[edge] ACK shape mismatch: seqno={ack_seqno} "
              f"entries={len(ack)}")
        ep.close()
        return 1
    bad = [s for s, c in ack.items() if c != expected_crc]
    print(f"[edge] send-side  {send_stats.summary(e2e)}")
    print(f"[edge] end-to-end frames={args.frames} "
          f"bytes={frame_bytes * args.frames / 1e6:.1f}MB "
          f"wall={e2e * 1000:.0f}ms "
          f"goodput={frame_bytes * args.frames * 8 / e2e / 1e9:.2f}Gbps")
    print(f"[edge] CRC verify: "
          f"{'OK (all ' + str(args.frames) + ' frames match)' if not bad else 'FAILED seqnos=' + str(bad[:5])}")
    ep.close()
    return 0 if not bad else 1


def run_server(args) -> int:
    ep, peer = VerbsEndpoint.serve(args.host, args.port)
    print(f"[cloud] family=verbs QP established with {peer} "
          f"(dev={ep.rc.dev_name} qpn={ep.rc.qpn} "
          f"gid_index={ep.rc.gid_index}), expecting {args.frames} frames")

    stats = FrameStats()
    crcs: dict[str, int] = {}
    t_start = now()
    for expected in range(args.frames):
        t0 = now()
        seqno, payload, aux = ep.recv_frame()
        stats.add(len(payload) + len(aux), now() - t0)
        if seqno != expected:
            print(f"[cloud] FATAL pairing error: expected {expected}, "
                  f"got {seqno}")
            ep.close()
            return 1
        crcs[str(seqno)] = zlib.crc32(payload)
        if (expected + 1) % 20 == 0:
            print(f"[cloud] received {expected + 1}/{args.frames}")

    wall = now() - t_start
    print(f"[cloud] recv-side  {stats.summary(wall)}")

    ack = json.dumps(crcs).encode()
    ep.send_frame(args.frames, ack)
    ep.drain()
    ep.close()
    return 0
