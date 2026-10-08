# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""边侧 demo:模拟 lwd_edge——构造 UP 通道 hidden 帧(AF_SMC)发往云侧。

载荷模拟边侧 embedding 之后的 hidden states:
  payload = bf16 [rows, hidden_size]      (默认 512 x 4096 = 4MiB)
  aux     = int64 [rows, 3] mrope positions (默认 512*3*8 = 12KiB)

发送满 --frames 帧后,云侧经 DOWN 方向回传逐帧 CRC ACK,边侧比对,
即 demo 完整覆盖 UP/DOWN 双通道语义。

用法(边侧 Linux 机器,IP 用 RoCE 网段内本机地址可达云侧的地址):
  python3 smc_client.py --host <cloud_roce_ip> --port 19901
  python3 smc_client.py --family tcp ...          # TCP 对照组
  python3 smc_client.py --rows 1024 --hidden 5120 # 更大帧压吞吐

注:sendall 返回只代表数据进入本地发送缓冲/RDMA 环,不代表对端已
收到;端到端吞吐以 client 的"首帧发出->ACK 收完"与 server 的接收
统计为准,两者都打印。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import zlib

from smc_common import (
    FrameStats,
    connect_socket,
    now,
    open_socket,
    print_smc_state,
    recv_frame,
    send_frame,
)


def build_payload(rows: int, hidden: int) -> bytes:
    """bf16 hidden 帧的字节内容(数值语义与传输无关,随机即可)。"""
    return os.urandom(rows * hidden * 2)


def build_aux(rows: int) -> bytes:
    """mrope positions 帧:int64 [rows, 3]。"""
    return os.urandom(rows * 3 * 8)


def main() -> int:
    ap = argparse.ArgumentParser(description="AF_SMC demo: edge side (UP sender)")
    ap.add_argument("--host", required=True, help="云侧 RoCE 网段 IP")
    ap.add_argument("--port", type=int, default=19901)
    ap.add_argument("--family", choices=["smc", "tcp"], default="smc")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--rows", type=int, default=512, help="num_tokens")
    ap.add_argument("--hidden", type=int, default=4096, help="hidden_size (bf16)")
    ap.add_argument("--interval-ms", type=float, default=0.0,
                    help="帧间隔,模拟边侧推理节拍;0=满速压测")
    args = ap.parse_args()

    payload = build_payload(args.rows, args.hidden)
    aux = build_aux(args.rows)
    frame_bytes = len(payload) + len(aux)
    expected_crc = zlib.crc32(payload)

    sock = open_socket(args.family)
    t0 = now()
    # AF_SMC socket 的 connect 不能走 Python socket.connect(同 bind:
    # Python 不认 AF_SMC 地址族),改走 smc_common.connect_socket。
    connect_socket(sock, args.family, args.host, args.port)
    connect_ms = (now() - t0) * 1000
    # SMC 建连含 TCP 握手 + CLC 协商 + RDMA 链路建立,
    # 比 TCP 握手慢是正常现象(等价于 HCCL 方案的建组/预热成本)。
    print(f"[edge] family={args.family} connected to {args.host}:{args.port} "
          f"in {connect_ms:.1f}ms; frame={frame_bytes / 1e6:.2f}MB "
          f"({args.rows}x{args.hidden} bf16 + {args.rows}x3 i64) "
          f"frames={args.frames}")

    send_stats = FrameStats()
    t_start = now()
    for seqno in range(args.frames):
        t0 = now()
        send_frame(sock, seqno, payload, aux)
        send_stats.add(frame_bytes, now() - t0)
        if (seqno + 1) % 20 == 0:
            print(f"[edge] sent {seqno + 1}/{args.frames}")
        if args.interval_ms > 0:
            time.sleep(args.interval_ms / 1000.0)

    # DOWN 方向:收云侧逐帧 CRC ACK,校验传输完整性。
    ack_seqno, ack_body, _ = recv_frame(sock)
    ack = json.loads(ack_body)
    if ack_seqno != args.frames or len(ack) != args.frames:
        print(f"[edge] ACK shape mismatch: seqno={ack_seqno} entries={len(ack)}")
        sock.close()
        return 1
    bad = [s for s, c in ack.items() if c != expected_crc]
    e2e = now() - t_start
    print(f"[edge] send-side  {send_stats.summary(e2e)}")
    print(f"[edge] end-to-end frames={args.frames} "
          f"bytes={frame_bytes * args.frames / 1e6:.1f}MB "
          f"wall={e2e * 1000:.0f}ms "
          f"goodput={frame_bytes * args.frames * 8 / e2e / 1e9:.2f}Gbps")
    print(f"[edge] CRC verify: {'OK (all ' + str(args.frames) + ' frames match)' if not bad else 'FAILED seqnos=' + str(bad[:5])}")
    if args.family == "smc":
        print_smc_state("edge")
    sock.close()
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
