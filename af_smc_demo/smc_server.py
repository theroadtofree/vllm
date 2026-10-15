# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""云侧 demo:模拟 lwd_cloud——UP 通道收 hidden 帧,回 DOWN 方向 ACK。

逐帧接收(帧长可变,头部自描述,无需预知尺寸),对每帧计算 CRC32,
收满 --frames 帧后把 {seqno: crc} 打包成一个 DOWN 帧回传,边侧比对。

真实链路里云侧收到 payload 后还有 H2D(->NPU)一跳,demo 不模拟,
延迟口径差异见 README"性能口径"一节。

用法(云侧 Linux 机器):
  python3 smc_server.py --host <cloud_roce_ip> --port 19901
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import zlib

from smc_common import (
    FrameStats,
    _set_big_buffers,
    accept_socket,
    bind_socket,
    now,
    open_socket,
    print_smc_state,
    recv_frame,
    send_frame,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="AF_SMC demo: cloud side (UP receiver)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind 地址;建议显式 bind 云侧 RoCE 网段 IP")
    ap.add_argument("--port", type=int, default=19901)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--family", choices=["smc", "tcp", "verbs"], default="smc")
    args = ap.parse_args()

    if args.family == "verbs":
        import verbs_demo

        return verbs_demo.run_server(args)

    srv = open_socket(args.family)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # AF_SMC socket 的 bind 不能走 Python socket.bind(Python 不认 AF_SMC
    # 地址族,会抛 "bad family"),改走 smc_common.bind_socket 直接调 libc。
    bind_socket(srv, args.family, args.host, args.port)
    srv.listen(1)
    print(f"[cloud] family={args.family} listening on {args.host}:{args.port}, "
          f"expecting {args.frames} frames")

    # AF_SMC socket 的 accept 同样要走 smc_common.accept_socket(Python
    # socket.accept 内部 getsockaddrlen 不认 AF_SMC)。
    conn, peer = accept_socket(srv, args.family)
    # 防御性补设:buffer 必须在 listener 上就位(open_socket 建socket 时
    # 已设,accept 子 socket 握手期继承——云侧接收窗口正确起步的关键,
    # 勿把设 buffer 挪到 accept 之后:握手期按默认小 buffer 定型初始窗口,
    # 持续排空的接收端在该内核上不再触发窗口增长,单流被钉死在初始窗口/
    # RTT,vllm 真机实录 28KB@5ms ≈ 5MB/s)。TCP_NODELAY 在部分平台不随
    # accept 继承,子 socket 重设一次;对已建立连接重设 buffer 幂等无害。
    if args.family == "tcp":
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    _set_big_buffers(conn)
    print(f"[cloud] accepted from {peer}")

    stats = FrameStats()
    crcs: dict[int, int] = {}
    t_start = now()
    for expected in range(args.frames):
        t0 = now()
        seqno, payload, aux = recv_frame(conn)
        stats.add(len(payload) + len(aux), now() - t0)
        # 字节流保序,seqno 只做防御校验:错位说明两端配对已损坏。
        if seqno != expected:
            print(f"[cloud] FATAL pairing error: expected seqno={expected}, "
                  f"got {seqno}")
            conn.close()
            srv.close()
            return 1
        crcs[seqno] = zlib.crc32(payload)
        if (expected + 1) % 20 == 0:
            print(f"[cloud] received {expected + 1}/{args.frames}")

    wall = now() - t_start
    print(f"[cloud] recv-side  {stats.summary(wall)}")

    # DOWN 方向:逐帧 CRC 摘要一次性回传。
    send_frame(conn, args.frames, json.dumps(crcs).encode())
    if args.family == "smc":
        print_smc_state("cloud")
    conn.close()
    srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
