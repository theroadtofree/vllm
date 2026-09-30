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

## 与正式迁移的关系

| demo 组件 | 正式方案对应(vllm-ascend lwd_comm) |
|---|---|
| `send_frame`/`recv_frame` 帧协议 | `lwd_comm/smc_wire.py` 帧编解码(同款头部) |
| UP 通道(payload+aux) | `LwdChannelType.UP` 的 tensor+aux_tensor |
| DOWN ACK 帧 | `LwdChannelType.DOWN` 云→边回传 |
| seqno 帧头校验 | per-channel seqno FIFO 配对契约(防御层) |
| `--family tcp` 对照 | SMC 自身的 TCP fallback 行为对照 |

Windows 上无法运行(需 Linux AF_SMC);本目录代码只在边/云 Linux 机器上执行。
