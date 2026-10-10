# AF_SMC 边云数据面双机 Demo

用 `AF_SMC` socket(Python 标准库,零第三方依赖)在两台 Linux 机器之间传输模拟的
边云 hidden states,验证 **SMC-R 数据面** 在你们网卡/内核组合上的可行性。
帧协议与 [HCCL→AF_SMC 迁移方案](HCCL_TO_AF_SMC_MIGRATION.md) 完全一致,demo 即
数据面 wire 层的参考实现。

```
边侧 smc_client.py                     云侧 smc_server.py
  hidden bf16 [512, 4096] = 4MiB  ──┐    ┌── 逐帧 CRC 校验
  aux  i64  [512, 3]     = 12KiB    ├─UP─┤
  (UP 通道,100 帧)                  ┘    └── 收满回 DOWN ACK(逐帧 CRC)
```

## 前置条件(两端机器)

1. Linux 内核 >= 4.11(6.6 已验证支持),`uname -r` 确认。
2. 有 RoCE 网卡且驱动已加载:`rdma link show` 应列出设备(如 `rxe`/`mlx5_roce`/
   华为 RoCE);`ibv_devices` 非空。
3. 两端 RoCE 网段 IP 互通(先 `ping <对端_roce_ip>` 确认路由走 RoCE 口)。
4. 可选:安装 smc-tools(`smc` 命令)便于排查降级。

## 运行

```bash
# 云侧(bind 云侧 RoCE 网段 IP)
python3 smc_server.py --host 10.0.0.2 --port 19901

# 边侧(另开一台,connect 云侧 RoCE IP)
python3 smc_client.py --host 10.0.0.2 --port 19901
```

常用变体:

```bash
# TCP 对照组(同机同参,对比 SMC 的吞吐/CPU 差异)
python3 smc_server.py --family tcp ...   # 两端都要加 --family tcp
python3 smc_client.py --family tcp ...

# 更大帧压吞吐(1024 token x 5120 hidden ~ 10MiB/帧)
python3 smc_client.py --host 10.0.0.2 --rows 1024 --hidden 5120 --frames 50

# 模拟边侧推理节拍(每 50ms 一帧,观察节拍下的延迟)
python3 smc_client.py --host 10.0.0.2 --interval-ms 50
```

预期输出(边侧,200G RoCE 参考量级,以实测为准):

```
[edge] family=smc connected to 10.0.0.2:19901 in 3.2ms; frame=4.02MB ...
[edge] end-to-end frames=100 bytes=402.3MB wall=251ms goodput=12.8Gbps
[edge] CRC verify: OK (all 100 frames match)
[edge] /proc/net/smc: header=... entries=1
```

## 确认没有降级(重要)

SMC 协商失败会**静默回退 TCP**,功能正常但拿不到 RDMA 收益。验证手段按可靠性排序:

1. demo 自带:结束时打印 `/proc/net/smc`,连接表非空(`entries>=1`)即走了 SMC-R。
2. `smc dbg` — smc-tools 输出链路/连接详情,含 fallback 计数。
3. `dmesg | grep -i smc` — CLC 协商被拒的原因都会在这里(无 RoCE 设备、
   PNET 不匹配、对端不支持)。
4. 对比法:同参数跑 `--family tcp`,SMC 的 goodput/CPU 占用应有明显优势;
   若两者几乎一样,大概率已降级。

常见降级原因:connect 源地址没走 RoCE 网段(检查路由)、防火墙拦 UDP 4791
之外的 RoCE 流量、网卡驱动未注册 RDMA 设备。

## 性能口径说明

- demo 统计的是 **host 内存到 host 内存** 的 SMC 传输(含 SMC 内部两次拷贝)。
- 真实边云链路多两跳:边侧 D2H(NPU→pinned host)、云侧 H2D(host→NPU),
  各约 PCIe 带宽量级;迁移方案文档中有完整时序与预估。
- `sendall` 返回仅代表数据进入本地发送路径;端到端吞吐以边侧
  "首帧发出→ACK 收完"与云侧接收统计为准。

## 吞吐低于 TCP 的排查(SMC-R 单流的三个限制器)

实测参考:10G 链路,ib_write_bw 9.4-9.7Gbps,demo SMC 3.3Gbps(低 RTT 环境)/
0.67Gbps(高 RTT 环境)。SMC-R 的吞吐模型与 TCP 本质不同:

| | TCP | SMC-R |
|---|---|---|
| 窗口 | 运行时 autotuning,放开 rmem/wmem_max 新连接立刻受益 | 建连时 CLC 协商 RMBE 档位(定死),受协议编码上限约束 |
| 数据推进 | 连续字节流 + TSO/GRO offload | 分块 RDMA WRITE + CDC credit 逐块确认 |

排查顺序:

1. **宿主 buffer 上限**(两端):demo 已 setsockopt 8MB,但内核会 clamp 到
   `net.core.rmem_max/wmem_max`——默认 ~208KB 时 RMBE 取最小档:
   `sudo sysctl -w net.core.rmem_max=16777216 net.core.wmem_max=16777216`
2. **autocorking**(两端,大块吞吐最直接的旋钮):SMC 发送端默认攒 64KB
   (`net.smc.autocorking_size`)才推进一次 RDMA WRITE,4MiB 帧 = 64 块,
   每块等 CDC 确认——吞吐 ≈ 块大小 / RTT,高 RTT 环境直接崩(实测 0.7ms
   RTT 下 64×0.7ms/帧 ≈ 0.67Gbps,与实测吻合):
   `sudo sysctl -w net.smc.autocorking_size=0`(禁用;或 1048576)
3. **RTT**:`ping <对端_roce_ip>`——同一调优在低/高 RTT 环境效果差异巨大,
   对比两环境数据时先对齐 RTT。
4. **确认未 fallback**:`smcss` / `/proc/net/smc`;fallback 连接 = TCP +
   SMC 层开销,特征恰好是"比纯 TCP 略低"。
5. **口径校准**:看 server 行 recv-side goodput(client 的 end-to-end 含
   DOWN ACK 回传,RTT 大时被惩罚)。

调优后预期:低 RTT 环境 SMC 单流应显著超过 TCP、达到 ib_write_bw 的
60-90%;高 RTT 环境 CDC 逐块确认的物理下限仍在——这是 SMC-R 协议模型
(拷贝 + 分块 + 块确认)相对裸 verbs 的固有代价,若高 RTT 场景最终不达标,
评估回到迁移方案文档里的 verbs 直写路线。

## verbs 模式(--family verbs,拉远场景的预期路线)

`verbs_demo.py` 提供 libibverbs(ctypes) RC SEND 传输,与正式实现
`vllm_ascend/distributed/lwd_comm/verbs_wire.py` 同款模型(帧协议、
握手消息格式、块池、RC SEND 双边),无 NPU 依赖,先在 demo 层验证
verbs 数据面的连通与吞吐:

```bash
# 云侧(--port 即握手 TCP 端口)
python3 smc_server.py --family verbs --host <云RoCE_IP> --port 19901
# 边侧
python3 smc_client.py --family verbs --host <云RoCE_IP> --port 19901
```

- 为什么:SMC-R 的 RMBE/RTT 窗口在拉远场景(实测 500km/5ms)上限
  ~0.8Gbps(协议限制);verbs 无逐块确认模型,同链路 ib_send_bw
  ~9.4Gbps。先跑 demo 对照 `ib_send_bw` 基线,再切正式实现
  (`data_plane_family=verbs`)。
- 依赖:`libibverbs.so.1`(能跑 ib_write_bw 即具备);env 可调
  `VERBS_BLOCK_MB`(默认 64)/`VERBS_N_BLOCKS`(默认 4)/
  `VERBS_GID_INDEX`(RoCE v2 gid 自动探测不准时手动指定)/
  `VERBS_DEV`(多网卡指定设备名)。
- 排障:QP 迁移报 `errno=22` 时对照部署机
  `/usr/include/infiniband/verbs.h` 的 `ibv_qp_attr_mask` 核对常量;
  选错网卡用 `VERBS_DEV` 指定;`ibv_devices` 看设备列表。

## 与正式迁移的关系

| demo 组件 | 正式方案对应(vllm-ascend lwd_comm) |
|---|---|
| `send_frame`/`recv_frame` 帧协议 | `smc_wire.py`/`verbs_wire.py` 帧编解码(同款头部) |
| UP 通道(payload+aux) | `LwdChannelType.UP` 的 tensor+aux_tensor |
| DOWN ACK 帧 | `LwdChannelType.DOWN` 云→边回传 |
| seqno 帧头校验 | per-channel seqno FIFO 配对契约(防御层) |
| `--family tcp` 对照 | SMC 自身的 TCP fallback 行为对照 |
| `--family verbs`(`verbs_demo.py`) | `verbs_wire.py`(握手格式/块池/RC SEND 同款) |

Windows 上无法运行(需 Linux AF_SMC/libibverbs);本目录代码只在
边/云 Linux 机器上执行。
