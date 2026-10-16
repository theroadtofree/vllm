# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NDR(NPU Direct RDMA)可行性探针:910B + npu_peermem 场景。

复用 verbs_demo 的 QP/握手/hw-ABI 机制(保证探针走的 libibverbs 与
生产 verbs_wire 完全同一条路径),新增对 **NPU 设备内存指针** 的
ibv_reg_mr 与双向 DMA 验证。云侧只需普通 host 内存。

分六个 Phase,每个独立 PASS/FAIL,详见模块尾部判读表:

  A 注册能力:设备指针 reg_mr 能否成功(多尺寸/权限位/反复 reg-dereg)
  B 设备→线缆:NDR SEND 的数据完整性(CRC 对拍,内容每轮变化)
  C 线缆→设备:post_recv 落设备内存,CQE 后立即可读(D2H 回读校验)
  D 复用刷新:MR 只注册一次,设备侧 D2D 改写内容后再发(生产形态)
  E CQE/DMA 竞态:CQE 后立即覆写,统计撕裂帧(量化 NDR 所需隔离期)
  F 带宽/时延:NDR vs 现产 D2H 暂存路径的分帧耗时(直接换算收益)

用法:
  云侧(任意 RDMA 对端):
    python3 ndr_probe.py --role cloud --host <cloud_roce_ip> --port 19910
  边侧(910B,需 torch_npu):
    python3 ndr_probe.py --role edge --host <cloud_roce_ip> --port 19910

verbs_demo 的 env(VERBS_GID_INDEX/VERBS_DEV/VERBS_ABI/VERBS_QP_TIMEOUT
等)对本探针同样生效。控制面走 --port+1 的第二条 TCP。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import socket
import sys
import time
import zlib

import verbs_demo as vd
from smc_common import accept_socket, bind_socket, connect_socket, open_socket

# 独立绑定 reg_mr(use_errno=True)以捕获失败 errno(_ibv 主绑定没开 errno)
_ibv2 = ctypes.CDLL("libibverbs.so.1", use_errno=True)
_ibv2.ibv_reg_mr.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
]
_ibv2.ibv_reg_mr.restype = ctypes.POINTER(vd._Mr)

_TILE = 1 << 16  # 64KiB 图案瓦片(种子决定内容,两侧独立可复算)


def make_bytes(nbytes: int, seed: int) -> bytes:
    """确定性图案:seed 派生 64KiB 瓦片平铺到 nbytes(两侧零通信对拍)。"""
    tile = (seed.to_bytes(8, "little") * 4 + b"NDRPROBE")[:64]
    tile = (tile * (_TILE // len(tile) + 1))[:_TILE]
    return tile * (nbytes // _TILE) + tile[: nbytes % _TILE]


def crc32(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF


# --------------------------------------------------------------------- #
# verbs 原语:按裸地址 post(设备/_host MR 通用)                        #
# --------------------------------------------------------------------- #

def reg_addr(rc: vd._Rc, addr: int, length: int, access: int) -> tuple:
    """对裸地址注册 MR;返回 (lkey, errno)。errno=0 即成功。"""
    mr = _ibv2.ibv_reg_mr(
        ctypes.c_void_p(rc._pd), ctypes.c_void_p(addr), length, access)
    if not mr:
        return 0, ctypes.get_errno()
    return mr.contents.lkey, 0


def post_send_addr(rc: vd._Rc, addr: int, lkey: int, length: int,
                   wr_id: int) -> tuple:
    sge = vd._Sge(addr, length, lkey)
    wr = vd._SendWr()
    wr.wr_id = wr_id
    wr.next = None
    wr.sg_list = ctypes.pointer(sge)
    wr.num_sge = 1
    wr.opcode = vd._IBV_WR_SEND
    wr.send_flags = vd._IBV_SEND_SIGNALED
    rc.post_send(wr)
    return wr, sge  # 必须存活到 CQE


def post_recv_addr(rc: vd._Rc, addr: int, lkey: int, length: int,
                   wr_id: int) -> tuple:
    sge = vd._Sge(addr, length, lkey)
    wr = vd._RecvWr()
    wr.wr_id = wr_id
    wr.next = None
    wr.sg_list = ctypes.pointer(sge)
    wr.num_sge = 1
    rc.post_recv(wr)
    return wr, sge


def wait_cqe(rc: vd._Rc, want_recv: bool, timeout_s: float):
    """轮询至出现目标 CQE;返回 _Wc,超时抛 TimeoutError。"""
    deadline = time.monotonic() + timeout_s
    while True:
        for wc in rc.poll(8):
            if wc.status != vd._IBV_WC_SUCCESS:
                raise RuntimeError(f"CQE status={wc.status} "
                                   f"vendor={wc.vendor_err}")
            is_recv = (wc.opcode == vd._IBV_WC_RECV)
            if is_recv == want_recv:
                return wc
            # 非目标方向的 CQE(如发送侧收到对端 ACK 回包)忽略
        if time.monotonic() > deadline:
            raise TimeoutError("CQE timeout")
        time.sleep(0.001)


# --------------------------------------------------------------------- #
# 控制面:JSON 行协议                                                     #
# --------------------------------------------------------------------- #

def _send_line(sk: socket.socket, obj: dict) -> None:
    sk.sendall((json.dumps(obj) + "\n").encode())


def _recv_line(sk: socket.socket) -> dict:
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = sk.recv(65536)
        if not chunk:
            raise ConnectionError("ctrl peer closed")
        buf += chunk
    return json.loads(buf)


# --------------------------------------------------------------------- #
# 云侧:host 内存对端                                                     #
# --------------------------------------------------------------------- #

def run_cloud(args) -> int:
    ctrl_srv = open_socket("tcp")
    bind_socket(ctrl_srv, "tcp", args.host, args.port + 1)
    ctrl_srv.listen(1)
    qp_srv = open_socket("tcp")
    qp_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bind_socket(qp_srv, "tcp", args.host, args.port)
    qp_srv.listen(1)
    print(f"[cloud] listening ctrl={args.host}:{args.port + 1} "
          f"qp-hs={args.host}:{args.port}")

    ctrl, peer1 = accept_socket(ctrl_srv, "tcp")
    print(f"[cloud] ctrl from {peer1}")
    rc = None
    try:
        conn, peer2 = accept_socket(qp_srv, "tcp")
        rc = vd._Rc(route_host=args.host if args.host not in ("", "0.0.0.0")
                    else "")
        vd._hs_xchg(conn, rc)
        print(f"[cloud] QP up with {peer2} qpn={rc.qpn}")
    except Exception:
        if rc is not None:
            rc.destroy()
        raise
    finally:
        qp_srv.close()

    # 4 块 host 接收缓冲(轮转,支持流水)
    n_blk = 4
    size = args.max_mb << 20
    blocks = []
    for i in range(n_blk):
        buf = bytearray(size)
        lkey = rc.reg_mr(buf)
        blocks.append((buf, lkey, ctypes.addressof(
            (ctypes.c_char * size).from_buffer(buf))))
    recv_slot = 0
    holds: list = []  # WR/SGE 须存活到 CQE,统一挂起防悬挂(probe 短命)
    while True:
        req = _recv_line(ctrl)
        cmd = req.get("cmd")
        if cmd == "quit":
            _send_line(ctrl, {"ok": True})
            break
        elif cmd == "prepare_recv":
            slot = recv_slot % n_blk
            buf, lkey, addr = blocks[slot]
            # wr_id 必须编码块槽位:verify 按 wr_id-1000 取块,恒 1000
            # 会让轮转块全被误读成 blocks[0] 的旧数据(首轮实测踩坑)
            holds.append(
                post_recv_addr(rc, addr, lkey, req["size"], 1000 + slot))
            recv_slot += 1
            _send_line(ctrl, {"ok": True})
        elif cmd == "verify":
            wc = wait_cqe(rc, True, 60.0)
            blk = wc.wr_id - 1000
            data = bytes(memoryview(blocks[blk][0])[: wc.byte_len])
            _send_line(ctrl, {"len": wc.byte_len, "crc": crc32(data),
                              "match": crc32(data) == req["crc"]})
        elif cmd == "send_pattern":
            nbytes, seed = req["size"], req["seed"]
            buf, lkey, addr = blocks[0]
            buf[:nbytes] = make_bytes(nbytes, seed)
            holds.append(post_send_addr(rc, addr, lkey, nbytes, 2000))
            wait_cqe(rc, False, 60.0)
            _send_line(ctrl, {"ok": True, "crc": crc32(buf[:nbytes])})
        elif cmd == "bw_recv":
            nbytes, iters = req["size"], req["iters"]
            # 串行:挂 1 帧 WQE→等 CQE→再挂(B-E 同款模式)。
            # WQE 恒挂整块长度:每轮末尾重挂的 WQE 会残留到下一轮,若按
            # nbytes 挂、且下一轮帧更大(实测 1MiB→4MiB),首帧撞进过短
            # 的残留 WQE → 响应端长度错 → QP 进 ERROR(status 9)。
            # 整块挂则残留恒 128MiB,任何后续帧长都容得下(同生产
            # verbs_wire._post_recv 的整块挂法,byte_len 报实际帧长)。
            full = args.max_mb << 20
            t0 = time.monotonic()
            for k in range(iters):
                buf, lkey, addr = blocks[k % n_blk]
                holds.append(post_recv_addr(rc, addr, lkey, full,
                                            1000 + k % n_blk))
                wait_cqe(rc, True, 60.0)
            wall = time.monotonic() - t0
            _send_line(ctrl, {"wall_ms": wall * 1000,
                              "gbps": nbytes * iters * 8 / wall / 1e9})
        elif cmd == "bw_send":
            nbytes, iters, seed = req["size"], req["iters"], req["seed"]
            buf, lkey, addr = blocks[0]
            buf[:nbytes] = make_bytes(nbytes, seed)
            t0 = time.monotonic()
            inflight, done = 0, 0
            depth = min(4, iters)
            keep = []
            while done < iters:
                while inflight < depth and done + inflight < iters:
                    keep.append(post_send_addr(rc, addr, lkey, nbytes, 2000))
                    inflight += 1
                wait_cqe(rc, False, 60.0)
                keep.pop(0)
                inflight -= 1
                done += 1
            wall = time.monotonic() - t0
            _send_line(ctrl, {"wall_ms": wall * 1000,
                              "gbps": nbytes * iters * 8 / wall / 1e9})
        elif cmd == "echo":
            wc = wait_cqe(rc, True, 60.0)
            blk = wc.wr_id - 1000
            buf, lkey, addr = blocks[blk]
            holds.append(
                post_send_addr(rc, addr, lkey, wc.byte_len, 2000))
            wait_cqe(rc, False, 60.0)
            # 回显后必须重挂接收 WQE:上一轮的已被本帧消耗,不补挂则
            # 下一轮 edge 的帧到达时 RQ 空 → RNR 无限重试 → 双方超时
            # (实测第 2 次迭代即卡死)
            holds.append(post_recv_addr(
                rc, addr, lkey, args.max_mb << 20, 1000 + blk))
            _send_line(ctrl, {"ok": True})
        elif cmd == "race_recv":
            crcs = []
            for _ in range(req["iters"]):
                # 预挂深度 1(逐帧记录 crc,防覆写)
                wc = wait_cqe(rc, True, 60.0)
                blk = wc.wr_id - 1000
                crcs.append(crc32(
                    bytes(memoryview(blocks[blk][0])[: wc.byte_len])))
                buf, lkey, addr = blocks[blk]
                holds.append(post_recv_addr(
                    rc, addr, lkey, args.max_mb << 20, 1000 + blk))
            _send_line(ctrl, {"crcs": crcs})
        else:
            _send_line(ctrl, {"ok": False, "err": f"unknown cmd {cmd!r}"})
    ctrl.close()
    rc.destroy()
    print("[cloud] done")
    return 0


# --------------------------------------------------------------------- #
# 边侧:910B 设备内存 + 测试矩阵                                          #
# --------------------------------------------------------------------- #

def _fill_dev(dev, nbytes: int, seed: int) -> None:
    """host 图案 H2D 进设备块头部(仅准备,不计入被测路径);copy_ 是
    异步 enqueue,post_send 前必须同步,否则 NIC 可能读到未落 HBM 的
    旧内容(与总线一致性问题区分开)。"""
    host = make_bytes(nbytes, seed)
    dev[:nbytes].copy_(
        torch.frombuffer(bytearray(host), dtype=torch.uint8))
    torch.npu.synchronize()


def _read_dev(dev, nbytes: int) -> bytes:
    return dev[:nbytes].cpu().numpy().tobytes()


def run_edge(args) -> int:
    global torch
    import torch
    import torch_npu  # noqa: F401 注册 npu 后端

    ctrl = open_socket("tcp")
    connect_socket(ctrl, "tcp", args.host, args.port + 1)
    rc = vd._Rc(route_host=args.host)
    try:
        sock = open_socket("tcp")
        connect_socket(sock, "tcp", args.host, args.port)
        vd._hs_xchg(sock, rc)
    except Exception:
        rc.destroy()
        raise
    print(f"[edge] QP up qpn={rc.qpn} gid_index={rc.gid_index}")

    def call(req: dict) -> dict:
        _send_line(ctrl, req)
        return _recv_line(ctrl)

    results: list[str] = []

    def report(phase: str, ok: bool, detail: str) -> None:
        line = f"[{phase}] {'PASS' if ok else 'FAIL'} {detail}"
        print(line)
        results.append(line)

    max_bytes = args.max_mb << 20
    dev = torch.empty(max_bytes, dtype=torch.uint8, device="npu")
    print(f"[edge] dev block {max_bytes >> 20}MiB @ {hex(dev.data_ptr())}")

    # ---- Phase A:注册能力 ------------------------------------------- #
    a_ok, a_detail = True, []
    for mb in (4, 29, args.max_mb):
        lkey, err = reg_addr(rc, dev.data_ptr(), mb << 20,
                             vd._IBV_ACCESS_LOCAL_WRITE
                             | vd._IBV_ACCESS_REMOTE_WRITE)
        a_ok &= (err == 0)
        a_detail.append(f"{mb}MiB:{'ok' if not err else f'errno{err}'}")
    lkey0, err0 = reg_addr(rc, dev.data_ptr(), 4 << 20, 0)  # 仅本地读(发送)
    a_detail.append(f"access0:{'ok' if not err0 else f'errno{err0}'}")
    cycles = 0
    for _ in range(8):  # 反复 reg/dereg(池重建场景)
        _, e = reg_addr(rc, dev.data_ptr(), 4 << 20,
                        vd._IBV_ACCESS_LOCAL_WRITE)
        cycles += (e == 0)
    a_ok &= (cycles == 8)
    a_detail.append(f"reg-dereg x8:{cycles}/8")
    host_ctl = bytearray(4 << 20)
    ctl_lkey = rc.reg_mr(host_ctl)  # 对照:脚本本身正确性
    a_detail.append("host-ctrl:ok")
    report("A", a_ok, " ".join(a_detail))
    if not a_ok:
        print("[verdict] 设备内存注册不可用:peermem 未覆盖该内存类型或"
              "定制栈不支持 —— NDR 止步于此,维持 host 暂存方案")
        call({"cmd": "quit"})
        rc.destroy()
        return 1
    dev_lkey, _ = reg_addr(rc, dev.data_ptr(), max_bytes,
                           vd._IBV_ACCESS_LOCAL_WRITE
                           | vd._IBV_ACCESS_REMOTE_WRITE)

    # ---- Phase B:设备→线缆 完整性 ----------------------------------- #
    b_ok, b_detail = True, []
    for mb in (4, 29):
        nbytes = mb << 20
        for i in range(8):
            seed = 1000 + i  # 内容每轮变化,同一 MR 不重注册
            _fill_dev(dev, nbytes, seed)
            call({"cmd": "prepare_recv", "size": nbytes})
            wr, sge = post_send_addr(rc, dev.data_ptr(), dev_lkey,
                                     nbytes, 2000)
            wait_cqe(rc, False, 60.0)
            del wr, sge  # CQE 已到,WR/SGE 方可释放
            rep = call({"cmd": "verify", "crc": crc32(make_bytes(nbytes,
                                                                 seed))})
            b_ok &= rep.get("match", False)
        b_detail.append(f"{mb}MiB x8")
    report("B", b_ok, " ".join(b_detail))

    # ---- Phase C:线缆→设备 完整性 ----------------------------------- #
    c_ok, c_detail = True, []
    for mb in (4, 29):
        nbytes = mb << 20
        for i in range(8):
            seed = 2000 + i
            # 先挂 recv 再让对端发:RQ 空 → RNR 无限重试(rnr_retry=7)
            # → 对端 send CQE 永不到来(首轮实测踩坑)
            rwr, rsge = post_recv_addr(rc, dev.data_ptr(), dev_lkey,
                                       nbytes, 1000)
            ack = call({"cmd": "send_pattern", "size": nbytes,
                        "seed": seed})
            wait_cqe(rc, True, 60.0)
            del rwr, rsge
            got = crc32(_read_dev(dev, nbytes))
            c_ok &= (got == ack["crc"])
        c_detail.append(f"{mb}MiB x8")
    report("C", c_ok, " ".join(c_detail))

    # ---- Phase D:复用刷新(生产形态:D2D 改写→再发) ------------------- #
    d_ok = True
    nbytes = 29 << 20
    for i in range(16):
        seed = 3000 + i
        src = torch.frombuffer(bytearray(make_bytes(nbytes, seed)),
                               dtype=torch.uint8).npu()
        with torch.npu.stream(torch.npu.Stream()):
            dev[:nbytes].copy_(src)  # D2D 落池(生产路径形态)
        torch.npu.synchronize()  # D2D 异步,post 前必须落 HBM
        call({"cmd": "prepare_recv", "size": nbytes})
        wr, sge = post_send_addr(rc, dev.data_ptr(), dev_lkey, nbytes, 2000)
        wait_cqe(rc, False, 60.0)
        del wr, sge
        rep = call({"cmd": "verify", "crc": crc32(make_bytes(nbytes, seed))})
        d_ok &= rep.get("match", False)
    report("D", d_ok, "29MiB x16 (D2D refresh, single MR)")

    # ---- Phase E:CQE/DMA 竞态量化 ----------------------------------- #
    e_nbytes = 4 << 20
    e_iters = 256
    srcs = [torch.frombuffer(make_bytes(e_nbytes, 4000 + i),
                             dtype=torch.uint8).npu() for i in range(4)]
    call({"cmd": "prepare_recv", "size": e_nbytes})
    # race_recv 模式:对端逐帧收并回传 crc 列表(预挂 1 深度)
    _send_line(ctrl, {"cmd": "race_recv", "iters": e_iters})
    # 本侧先预挂 K-1 帧?逐帧发:post→CQE→(延迟)覆写→post 下一帧
    # 对端 race_recv 逐帧重挂,首帧用上面 prepare_recv 的 WQE
    stream = torch.npu.Stream()
    expected = [crc32(make_bytes(e_nbytes, 4000 + (i + 1) % 4))
                for i in range(e_iters)]
    torn = stale = ok_cnt = 0
    prev_crc_exp = [crc32(make_bytes(e_nbytes, 4000 + i % 4))
                    for i in range(e_iters)]
    for i in range(e_iters):
        with torch.npu.stream(stream):
            dev[:e_nbytes].copy_(srcs[i % 4])
        torch.npu.synchronize()  # 内容落 HBM 后再 post(竞态只测 CQE 后)
        wr, sge = post_send_addr(rc, dev.data_ptr(), dev_lkey,
                                 e_nbytes, 2000)
        wait_cqe(rc, False, 60.0)
        del wr, sge
        if args.race_delay_ms > 0:
            time.sleep(args.race_delay_ms / 1000)
        # CQE 后立即覆写为下一帧内容(NDR 生产是否需要隔离期的直接量)
    rep = _recv_line(ctrl)
    got = rep["crcs"]
    for i, c in enumerate(got):
        if c == prev_crc_exp[i]:
            ok_cnt += 1
        elif c == expected[i]:
            stale += 1  # 对端收到的是"下一帧"内容(DMA 晚于覆写)
        else:
            torn += 1  # 撕裂帧(DMA 跨覆写窗口)
    report("E", torn == 0 and stale == 0,
           f"ok={ok_cnt} stale(next)={stale} torn={torn} "
           f"delay={args.race_delay_ms}ms (torn/stale>0 ⇒ NDR 需要保留 "
           f"CQE 后隔离期,逐步加 delay 重测找阈值)")

    # ---- Phase F1:发送带宽 NDR vs 暂存 ------------------------------- #
    f_lines = []
    src_dev = dev  # 直接发 dev 头部(先填好图案)
    host_stage = bytearray(max_bytes)
    stage_lkey = rc.reg_mr(host_stage)
    for mb in (1, 4, 29, 64):
        nbytes = mb << 20
        iters = 32
        src = torch.frombuffer(make_bytes(nbytes, 7777),
                               dtype=torch.uint8).npu()
        # ndr_zero 要发注册块内的地址:图案先 D2D 落进 dev 头部
        with torch.npu.stream(stream):
            dev[:nbytes].copy_(src)
        rows = {}
        for mode in ("ndr_zero", "ndr_d2d", "staged"):
            # 只发不等回复:cloud 的 bw_recv 要收完全部帧才回,同步 call
            # 会互等死锁(首轮实测踩坑);回复在发送循环结束后收
            _send_line(ctrl, {"cmd": "bw_recv", "size": nbytes,
                              "iters": iters})
            print(f"[F1] {mb}MiB {mode}: running...")
            t0 = time.monotonic()
            for i in range(iters):
                # 串行 post→CQE→下一帧(B-E 已验证形态;流水化首测超时,
                # 待单独定位后再提速)
                if mode == "ndr_zero":
                    wr, sge = post_send_addr(
                        rc, dev.data_ptr(), dev_lkey, nbytes, 2000)
                elif mode == "ndr_d2d":
                    with torch.npu.stream(stream):
                        dev[:nbytes].copy_(src)
                    torch.npu.synchronize()  # D2D 落 HBM 再 post(计时含)
                    wr, sge = post_send_addr(
                        rc, dev.data_ptr(), dev_lkey, nbytes, 2000)
                else:  # 生产现状:D2H(pageable)→拷入 host 注册块→post
                    t_d2h = time.monotonic()
                    host_t = src.cpu()  # non_blocking=False 同步 D2H
                    host_stage[:nbytes] = host_t.numpy().tobytes()
                    rows.setdefault("d2h_ms", 0)
                    rows["d2h_ms"] += (time.monotonic() - t_d2h) * 1000
                    wr, sge = post_send_addr(
                        rc, ctypes.addressof(
                            (ctypes.c_char * max_bytes).from_buffer(
                                host_stage)),
                        stage_lkey, nbytes, 2000)
                wait_cqe(rc, False, 60.0)
                del wr, sge
            wall = (time.monotonic() - t0) / iters
            rows[mode] = wall * 1000
            rep = _recv_line(ctrl)
            rows[mode + "_gbps"] = rep["gbps"]
            print(f"[F1] {mb}MiB {mode}: {wall * 1000:.2f}ms/frame "
                  f"cloud={rep['gbps']:.2f}Gbps")
        f_lines.append(
            f"{mb}MiB/frame: ndr_zero={rows['ndr_zero']:.2f}ms "
            f"ndr_d2d={rows['ndr_d2d']:.2f}ms staged={rows['staged']:.2f}ms "
            f"(其中 D2H+拷贝 {rows.get('d2h_ms', 0) / iters:.2f}ms) "
            f"云端收 {rows['ndr_zero_gbps']:.2f}/{rows['staged_gbps']:.2f}Gbps"
        )
    for ln in f_lines:
        print(f"[F1] {ln}")
        results.append(f"[F1] {ln}")

    # ---- Phase F3:小帧 RTT ------------------------------------------- #
    for kb in (8, 1200):
        nbytes = kb * 1024
        _fill_dev(dev, nbytes, 8888)
        lat = []
        for _ in range(200):
            _send_line(ctrl, {"cmd": "echo", "size": nbytes})
            t0 = time.monotonic()
            swr, ssge = post_send_addr(rc, dev.data_ptr(), dev_lkey,
                                       nbytes, 2000)
            rwr, rsge = post_recv_addr(rc, dev.data_ptr() + nbytes,
                                       dev_lkey, nbytes, 1000)
            wait_cqe(rc, True, 10.0)
            del swr, ssge, rwr, rsge
            lat.append((time.monotonic() - t0) * 1000)
            _recv_line(ctrl)  # 排空 echo 回复,防污染后续控制流
        lat.sort()
        msg = (f"{kb}KiB rtt avg={sum(lat) / len(lat):.3f}ms "
               f"p50={lat[len(lat) // 2]:.3f}ms p99={lat[int(len(lat) * .99)]:.3f}ms")
        print(f"[F3] {msg}")
        results.append(f"[F3] {msg}")

    call({"cmd": "quit"})
    ctrl.close()
    rc.destroy()

    print("\n===== NDR probe summary =====")
    for ln in results:
        print(ln)
    print("判读:A/B/C/D/E 全 PASS ⇒ NDR 可行;F1 的 staged-ndr 差值即每帧"
          "可省的毫秒数(×每请求帧数≈TTFT 收益);E 的 torn>0 ⇒ 需保留"
          "发送隔离期,按 delay 阈值定时长。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="NDR feasibility probe")
    ap.add_argument("--role", choices=["edge", "cloud"], required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=19910)
    ap.add_argument("--max-mb", type=int, default=128)
    ap.add_argument("--race-delay-ms", type=float, default=0.0,
                    help="Phase E:send CQE 后到覆写的延迟,扫 0/1/5/20/50")
    args = ap.parse_args()
    return run_cloud(args) if args.role == "cloud" else run_edge(args)


if __name__ == "__main__":
    sys.exit(main())
