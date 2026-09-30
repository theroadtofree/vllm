# 边云数据面迁移方案:HCCL hidden 传输 → AF_SMC

> 适用仓库:`C:\cann\vllm-ascend-github`(数据面 wire 层)、`C:\cann\vllm-github`
> (控制面/配置)。参考实现:[af_smc_demo](README.md)(帧协议与本方案一致)。

## 1. 现状与改动边界

### 1.1 现有数据面(HCCL)

```
边侧 LwdEdgeWorker                云侧 LwdCloudWorker/ModelRunner
  hidden(NPU)                       submit_recv(UP, num_elements<元数据推导>)
  submit_send(UP, tensor)             │
    │                                 ▼
    ▼                          LwdChannel._wire_recv
  LwdChannel._wire_send         torch.empty(n, bf16, device="npu")
  snapshot clone                dist.irecv(HCCL new_group)
  dist.isend(HCCL new_group)          │ future.wait(60)/wait_for_comm()
    │ future(reap)                    ▼
    ▼                               hidden(NPU) → 云侧继续计算
```

涉及文件(`vllm_ascend/distributed/`):

| 文件 | 职责 | SMC 方案 |
|---|---|---|
| `lwd_wire.py` | 2×`dist.new_group`(HCCL 域) + 2×NPU stream + warmup | **改造**:group → SMC 连接;stream/warmup 保留 |
| `lwd_comm/channel.py` | FIFO/seqno 重排/snapshot/stream 桥接/`isend-irecv` | **改造**:仅 `_wire_send/_wire_recv/_bridge_and_record` |
| `lwd_comm/future.py` | NPU event 完成语义 `done()/wait()/wait_for_comm()` | **零改动**(done_event 类型注释放宽) |
| `lwd_comm/service.py` | singleton、submit/poll/skip/shutdown | **零改动** |
| `lwd_comm/types.py` | `LwdCommRequest/Result` | **零改动** |
| `worker/lwd_edge_worker.py` `worker/lwd_cloud/*` | `submit_send/submit_recv/future.wait` | **零改动** |

控制面(主仓 `v1/lwd_control/control_communication/`,ZMQ PUSH/PULL over TCP)
**保持不动**:小消息场景 TCP 足够,且 ZMQ 不支持替换地址族。SMC 只替换
**数据面 wire 层**;数据面连接的 endpoint 发现新增走控制面(见 §5)。

### 1.2 目标数据面(AF_SMC)

```
边侧                                     云侧
  hidden(NPU)                              recv 到 pinned host buffer
  snapshot clone(NPU)                      H2D copy_(channel stream)
  D2H → pinned host buffer                 record NPU event = future 完成
  sock.sendall(AF_SMC) ──RoCEv2──→         hidden(NPU) → 继续计算(消费侧不变)
  HostDoneEvent = future 完成
```

对上层完全透明:`submit_send/submit_recv/future` 三件套签名与语义不变,
`lwd_edge_worker.py`、`lwd_cloud_worker.py`、`lwd_cloud_model_runner.py`
一行不改。这是本方案最大的收益——改动收敛在 wire 层两个文件 + 一个新文件。

## 2. 协议设计(与 demo 同款)

```
magic(4B "LWD1") + seqno(8B) + payload_len(8B) + aux_len(8B) + payload + aux
```

- `payload`:`num_elements`×2 字节(bf16 hidden,UP 主帧);DOWN 通道为
  combined c2e packet,同样按字节流传输。
- `aux`:UP 通道 `aux_num_elements`×8 字节(int64 mrope positions),与主帧
  同帧连续发送——对应现在"一个 seqno 管主+aux 两条 HCCL op"的语义。
- 帧头 `seqno` 是防御层:字节流天然保序,seqno 仅在错位(连接损坏/两端
  配对 bug)时立即报 `pairing error`,替代 HCCL"两端提交顺序契约"的
  隐式匹配。
- **skip/abort 语义增强**:HCCL 无法按序号丢弃已 post 的收发,只能依赖
  两端同步 `skip_seqno`;SMC 帧流是自描述的,接收端 IO 线程发现帧头
  seqno ∈ 本地 skipped 集合时可直接**物理跳过该帧字节**继续解析下一帧,
  不再要求"边侧 skip 时该帧尚未发出"(见 §6)。

## 3. 代码修改清单

### 3.1 新增 `vllm_ascend/distributed/lwd_comm/smc_wire.py`

```python
class HostDoneEvent:
    """发送侧 future 的完成事件:query() 兼容 LwdCommFuture 的
    done()/wait()/_finalize 路径;wait_event() 为防御性 no-op
    (发送 future 无 wait_for_comm 调用方,已核实)。"""
    def query(self) -> bool: ...
    def wait_event(self, _) -> None: ...  # no-op + warning
    def set(self) -> None: ...

class SmcIoThread:
    """per-(channel, op) 单线程 IO:D2H→sendall / recv→H2D。
    队列元素 (seqno, npu_tensor(s), future);socket 调用与 torch 拷贝
    均在本线程,不占 worker 计算线程。"""
    def submit_send(self, seqno, tensor, aux_tensor, future): ...
    def submit_recv(self, req, npu_buffer, aux_buffer, future): ...
    def add_skipped(self, seqno): ...      # 与 channel.skip_seqno 同步

class SmcLink:
    """一条物理通道的连接端点:云侧 listen/accept,边侧 connect。
    UP=port_base, DOWN=port_base+1,均由云侧 listen、边侧 connect,
    方向仍由谁读谁写区分(与 LwdChannelType 语义一致)。"""
    def serve(self, host, port): ...       # 云侧
    def connect(self, host, port): ...     # 边侧

def open_smc_socket(): ...                 # 同 demo smc_common.open_socket
```

- 接收用 `sock.recv_into(memoryview)` 分段填充 pinned buffer,避免拼接拷贝。
- pinned host buffer 池按帧大小档位复用(`torch.empty(n, pin_memory=True)`,
  torch_npu host pin,列为验证点 V1)。
- IO 线程内异常 → `future._finalize(error=...)` + link 标记 broken,
  后续 submit 立即失败(复用 future 现有 ERROR 路径)。

### 3.2 改造 `lwd_wire.py::init_lwd_duplex_channels`

```python
# 开关(灰度/回退):env LWD_WIRE_TRANSPORT = "hccl"(默认) | "smc"
if transport == "smc":
    # endpoint rank 解析逻辑不变(lwd_config / PP 组约定);
    # 两个 endpoint rank 各自:
    #   云 rank: SmcLink.serve(roce_ip, port_base)      # UP
    #           SmcLink.serve(roce_ip, port_base + 1)   # DOWN
    #   边 rank: SmcLink.connect(cloud_roce_ip, ...)    # 两条
    # interior rank:不参与(不再需要 world 级 new_group 同步)
else:
    dist.new_group(...)  # 现有 HCCL 路径原样保留
# NPU stream 创建保留(D2H/H2D 仍需在 channel stream 上排序)
```

- `_LWD_CHANNEL_GROUPS` 之外增加 `_LWD_CHANNEL_SMC: dict[channel, SmcIoThread]`;
  `get_lwd_channel_device_group` 仅剩 HCCL 路径调用(SMC 路径不再触达)。
- `warmup_lwd_duplex_channels`:SMC 下建连本身已含 TCP 握手+CLC 协商+
  RDMA 链路建立,warmup 简化为两端交换一帧 8B 探针(函数名与调用点不变);
  末尾 `get_world_group().barrier()` 保留。

### 3.3 改造 `lwd_comm/channel.py`(核心,~60 行净变更)

`_execute()` 中 send/recv 分支改为**向 IO 线程投递**,future 复用现有
`deferred() + _bind()` 机制(原本为乱序 seqno 设计,天然适配异步 IO):

```python
# _wire_send:替换 dist.isend
future = LwdCommFuture.deferred(req)
smc_io.submit_send(req.seqno, req.tensor, req.aux_tensor, future)
# IO 线程内: ch_stream.wait_stream(producer_stream)   # _order_after 保留
#            host_t.copy_(npu_t, non_blocking=True)
#            d2h_event.record(); d2h_event.synchronize()
#            sock.sendall(frame)                        # 帧协议见 §2
#            future._bind(handles=[], done_event=HostDoneEvent())
#            host_done_event.set()

# _wire_recv:替换 torch.empty(device="npu") + dist.irecv
# NPU 目标 buffer 仍在提交线程分配(现行为,可后续接内存池)
future = LwdCommFuture.deferred(req)
smc_io.submit_recv(req, npu_buffer, aux_buffer, future)
# IO 线程内: recv header → 校验/跳过 skipped seqno 帧
#            recv_into(pinned host_t) + recv_into(aux)
#            with torch.npu.stream(ch_stream):
#                npu_t.copy_(host_t, non_blocking=True)
#                done_event = torch.npu.Event(); done_event.record()
#            future._bind(handles=[], done_event=done_event, tensor=npu_t)
```

**完成语义映射**(与现状逐点对齐):

| | HCCL 现状 | AF_SMC |
|---|---|---|
| recv future 完成 | irecv 完成,NPU event | H2D 完成,NPU event(**同款**,`future.wait(60)`、`wait_for_comm()` 直接可用) |
| send future 完成 | isend 完成,NPU event | sendall 完成,`HostDoneEvent`(query() 兼容,`done()/reap/_finalize` 直接可用) |
| keepalive 释放 | reap→`_finalize` | 同左(sendall 完成→query True→reap) |
| 提交线程阻塞 | `handle.wait()` 即返 | 仅队列 put,即返(**不劣于现状**) |

`_snapshot_send`(clone 保持 producer 安全)、seqno FIFO、`_held` 重排、
`skip_seqno`、lazy reap、`shutdown` 全部保留;`_bridge_and_record` 的
HCCL handle 桥接逻辑随 handles=[] 自然失效,可删。

### 3.4 配置与 endpoint 发现

- Phase 1(静态):env `LWD_SMC_CLOUD_ADDR=<cloud_roce_ip>:<port_base>`,
  两端各自读取;`vllm/config/lwd.py` 增可选字段便于部署清单化管理。
- Phase 2(动态,推荐终态):控制面 ZMQ notify 增 `data_plane_endpoint`
  字段——云侧 worker 启动后经现有控制面发布自己的数据面地址,边侧收到
  后再 connect。**控制面拉起数据面**,替代原 torch.distributed
  rendezvous 在建组中的角色;云端点变化(重启/扩容)边侧自动重连。

## 4. 时序与同步关键点

1. **D2H/H2D 必须在 channel stream 上**(现约定保留):seqno FIFO 的顺序
   语义由"同一条 NPU stream 上的到达顺序"保证;`_order_after`(
   stream.wait_event(前驱 done_event))继续生效。
2. **Phase 1 正确性优先**:IO 线程 D2H 可先用同步 `npu_t.to("cpu")`,
   Phase 2 再换 `non_blocking=True + pinned 池 + event.synchronize()`
   (收益:隐藏拷贝与 socket 提交)。
3. **GIL**:`sendall/recv_into` 在 C 层释放 GIL;D2H/H2D 为 CANN op。
   worker 计算线程不受影响(与现状 `handle.wait()` 即返的体感一致)。
4. **半帧错位**:header 已收但对端中断 → `recv_exactly` 抛
   ConnectionError → 链路 broken,所有 pending future ERROR,与 HCCL
   集合通信失败的爆炸半径一致(单边云对)。

## 5. skip/abort 语义(增强,无需两端时序对赌)

| 场景 | HCCL 现状 | SMC |
|---|---|---|
| 边 abort 时帧未 post | 两端 skip,seqno 跳过 | 边 IO 队列标记取消,帧不上字节流;云 skip 后直接等下一 seqno |
| 边 abort 时帧已 post | 两端仍需同步 skip,已 post 的靠上层丢弃 | 边照发;云 IO 线程见帧头 seqno∈skipped → **物理丢帧**继续解析 |

`channel.skip_seqno` 现有逻辑保留,增加一行同步 `smc_io.add_skipped(seqno)`。

## 6. 性能预估与验证计划

单请求(UP,rows=512×hidden=7168 bf16 ≈ 7MiB,200G RoCE):

| 路径 | 量级 |
|---|---|
| D2H(PCIe ~25GB/s) | ~0.3ms |
| SMC 传输(线速 + 两次内核拷贝) | ~0.5ms |
| H2D | ~0.3ms |
| **合计 vs HCCL 网卡直达 ~0.3ms** | **多 ~0.8ms/请求** |

- 收益换成本:边侧 NPU/GPU/CPU 通吃、云侧无需 NPU RoCE 直达、摆脱 HCCL
  建组约束;TTFT 增加 ~1ms 量级,embedding 大 batch 场景可接受,待压测确认。
- 验证步骤:
  1. **V0** demo 双机跑通且 `/proc/net/smc` 确认未降级(含云侧华为 RoCE 网卡
     的 PNET 匹配——这是本方案最大的环境风险项);
  2. **V1** pinned memory 与 `non_blocking` 拷贝在 torch_npu 的行为(单元);
  3. **V2** `LWD_WIRE_TRANSPORT=smc` 1E1C 拓扑,`dump_tensor`(lwd_wire 现成)
     两端数值成对比对 = HCCL 基线一致;
  4. **V3** TTFT 探针(`[Lwd][perf]` 现成)压测对比,采样 CPU 占用差异。

## 7. 实施顺序与回退

1. **Phase 0**:demo(本目录)双机验证 SMC 链路不降级 —— *已具备*;
2. **Phase 1**:`smc_wire.py` + `channel.py/lwd_wire.py` 改造,env 静态
   endpoint,`LWD_WIRE_TRANSPORT` 开关,1E1C 数值回归(V2);
3. **Phase 2**:控制面 notify 交换 endpoint、pinned 池/non_blocking 优化、
   DOWN 通道同规格迁移;
4. **回退**:`LWD_WIRE_TRANSPORT=hccl`(默认)即回到现网行为,两条路径
   长期共存直至 Phase 3 评估下线 HCCL 路径。
