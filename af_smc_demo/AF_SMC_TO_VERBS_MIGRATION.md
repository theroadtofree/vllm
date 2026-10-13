# 边云数据面替换方案:AF_SMC socket → libibverbs RC 直连(verbs_wire)

> 适用仓库:`C:\cann\vllm-ascend-github`(数据面 wire 层)、
> `C:\cann\vllm-github`(控制面/装配配置)。参考 demo:
> [af_smc_demo](README.md) 的 `--family verbs`(`verbs_demo.py`)。
> 正式实现:`vllm_ascend/distributed/lwd_comm/verbs_wire.py`(已落地,
> 本文即该实现的设计文档,`data_plane_family=verbs` 启用)。

## 0. 摘要与动机

SMC-R 的数据推进是"RMBE 环形窗口 + CDC 逐块确认",吞吐上限 ≈
RMBE/RTT,与调参无关:拉远场景(实测 500km/5ms RTT)被压到 ~0.8Gbps;
同链路 `ib_send_bw`/`ib_write_bw` 实测 ~9.4Gbps。本方案把 wire 最底层
从 AF_SMC 字节流换成 **verbs RC QP(ctypes 直调 libibverbs,无 pyverbs
依赖)**:

- **传输模型换为 RC SEND 双边**:一帧一次 `post_send`,接收侧一帧一个
  CQE(`byte_len` 即帧长)——没有半帧/流错位问题,`_drain` 变成整块
  重新 `post_recv`;
- **注册块池替代 socket 缓冲**:每方向 N 块 × BLOCK_MB 的
  `bytearray + ibv_reg_mr` 长期锁页内存,发送 D2H 直落注册块(比 SMC
  少一次内核拷贝),接收 CQE 后对注册块做零拷贝视图直接 H2D;
- **上层零改动**:`channel.py`/`service.py`/`future.py`/三个 worker
  只依赖 wire 的方法名(`submit_send/submit_recv/add_skipped/shutdown`),
  `lwd_wire.py` 工厂按 `data_plane_family` 选择 `VerbsIoThread` 或
  `SmcIoThread`,帧头语义/seqno 配对/skip 语义/断线 ERROR 语义完全一致,
  两种 wire 可对拍;
- 控制面复用 SMC 方案的双轨寻址(静态配置建连 + HELLO additive 字段
  校验),仅握手端口从 `port_base/+1` 让出 `+2/+3` 给 verbs 的
  TCP 握手通道。

```
【SMC 现状】                                【verbs 目标】
边 NPU embeds(hidden)                       边 NPU embeds(hidden)
  │ snapshot clone(producer 流)              │ snapshot clone(producer 流)
  │ .to("cpu") 同步 D2H                       │ .to("cpu") 同步 D2H(channel 流)
  │ AF_SMC sendall                            │ memmove 落注册块 + CRC + 单 SGE post_send
  │  └ RMBE/CDC 逐块确认(RTT 上限)            │  └ RC SEND, CQE=对端 ACK,无逐块窗口
  ▼                                            ▼
云 pinned/RMB → H2D(NPU, ch_stream)          云注册块 CQE → frombuffer 零拷贝视图
  ▼                                            → H2D(NPU, ch_stream)
云内 TP broadcast(HCCL,不动)              云内 TP broadcast(HCCL,不动)
```

## 1. 代码地图(文件 × 动作)

| 位置 | 职责 | 相对 SMC 版的动作 |
|---|---|---|
| `vllm_ascend/distributed/lwd_comm/verbs_wire.py` | verbs wire 全量:ctypes 绑定、`_VerbsRc`(QP 生命周期)、块池、`VerbsIoThread`、`VerbsAcceptor`、握手编解码 | **新增**(~1700 行) |
| `vllm_ascend/distributed/lwd_wire.py` | wire 工厂:按 `data_plane_family` 分派;verbs 分支建 `VerbsIoThread`/`VerbsAcceptor`(握手端口 `base+2/+3`);启动期帧预算预检 | verbs 分支 + 预检 |
| `vllm_ascend/distributed/lwd_comm/channel.py` | per-(channel,op) FIFO、seqno 重排、snapshot、`_execute` 投递 | **零改动**(只认方法名) |
| `vllm_ascend/distributed/lwd_comm/future.py` | `done()/wait()/wait_for_comm()/_bind/_finalize` | **零改动**(完成事件仍是 `HostDoneEvent`/NPU event 双形态) |
| `vllm_ascend/distributed/lwd_comm/service.py`、`types.py` | FIFO 路由 / `LwdCommRequest` | **零改动** |
| `worker/lwd_edge_worker.py`、`worker/lwd_cloud/*` | EMBED/UNEMBED 提交与消费 | **零改动** |
| `vllm/v1/lwd_control/.../lwd_edge_assemble.py` | `LwdConfig.data_plane_family: str = "smc"`(+`"verbs"`) | 配置取值扩展,结构不动 |
| `vllm/v1/lwd_control/.../lwd_notify.py`、两个 engine | HELLO additive 字段 `data_plane_host/port_base` | 沿用 SMC 方案,零新增 |

启用方式:`lwd_config` 段(或 `VLLM_ASCEND_LWD_DATA_PLANE_FAMILY` env)
置 `data_plane_family: verbs`;`data_plane_host` 指云侧 RoCE 网段 IP。

## 2. 帧协议(v2,LWD2)与握手协议(LWDV1)

### 2.1 数据帧(与 smc_wire 的 LWD1 同构 + CRC)

```
偏移  长度  字段          语义
0     4    magic         b"LWD2"(v1 头 28B 无 CRC,升级 LWD2 后新旧两端
                          fail-fast)
4     8    seqno         uint64 网络序;UP=边侧 dispatch seqno(=控制面
                          LwdRangeNotify.seqno),DOWN=云侧 down_seqno
                          (=LwdC2eNotify.down_seqno)
12    8    payload_len   字节数(bf16;UP 含 Hash 专家 ID 段)
20    8    aux_len       字节数(int64 mrope [n,3],仅 UP;0=无)
28    4    crc32         zlib.crc32(payload+aux),发送端落块后计算回填
32    4    pad           对齐保留
36    ...  payload+aux   字节连续,单 SGE 单次 SEND
```

- 帧级 CRC 覆盖 `边 D2H+memmove → 线缆 → 云注册块` 全链;历史真机
  录案:UP 接收缓冲跨 seqno 串数据(record_stream 修复前)在无校验和时
  是**静默损坏**——UP embeds 错会漂移 KV/生成轨迹/MTP 接受率。
  `VLLM_ASCEND_LWD_VERBS_CRC = 0 关 / 1 帧级(默认) / 2 +收割点深度复核`
  (深度模式在云收割点把 NPU 数据 D2H 回算,三段定位 H2D 时错 vs
  H2D 后被换,仅诊断用)。
- 消息模式无半帧错位:接收 CQE 的 `byte_len` 必须等于
  `36+payload_len+aux_len`,不等即帧错位,broken。

### 2.2 握手(一次性小 TCP,云 listen base+2/+3,边 connect)

```
"!8sIIII16sI"  magic=b"LWDV1\0\0\0", lid, qpn, psn, mtu,
               gid[16], max_frame(=本端 _BLOCK_BYTES)
```

- SEND 双边不需要 rkey/对端内存布局,交换信息最少;交换完 TCP 即关。
- 对端 `max_frame < 本端 _BLOCK_BYTES` 时 fail-fast(两端
  `VLLM_ASCEND_LWD_VERBS_BLOCK_MB` 必须对齐)。
- 选网:路由源 IP 对齐(UDP connect trick 不发包取到对云侧地址的源
  IP → 设备+RoCE v2 gid index 一起锁定),多网卡时避免选到非路由口;
  `VLLM_ASCEND_LWD_VERBS_DEV/GID_INDEX` 可手动指定。CX5 会把同一 IPv4
  同时注册为 v1/v2 两条 gid,v1 纯以太封装跨三层必丢,必须选中 v2。

### 2.3 QP 参数

RC,MTU 4096,`sq_sig_all=1`(每个 SEND 都有 CQE),`max_send/recv_wr =
2×N_BLOCKS`,单 SGE,`timeout=16`、`retry_cnt/rnr_retry=7`(无限重试)、
`min_rnr_timer=16`、`max_rd_atomic=1`(SEND 不用 RDMA READ)。GRH:
`hop_limit=64`(拉远多跳;=1 第一跳即丢,真机实录)、`traffic_class=128`
(骨干按 DSCP=TC>>2=32 规划无损队列)。

## 3. 内存与拷贝路径(核心)

### 3.1 注册块池

- 每方向每端 `_N_BLOCKS`(默认 4)× `_BLOCK_BYTES`(默认 128MB,
  env `VLLM_ASCEND_LWD_VERBS_BLOCK_MB/N_BLOCKS` 可调),`bytearray`
  分配后 `ibv_reg_mr(LOCAL_WRITE|REMOTE_WRITE)` 长期锁页。
- **为什么不用 torch `pin_memory=True`**:NPU 环境下它分配的是 CANN
  管理的 host 内存(dvmm),`ibv_reg_mr` 注册直接失败(真机实录
  `reg_mr(64MB) failed`);普通 bytearray 由 ibv_reg_mr 自带长期锁页,
  demo 同款路径已在真机验证。
- 块池大小即天然流控/在途上限:发送侧等空闲块(`_acquire_block` 内
  边 poll CQ 边等),接收侧早到帧占块缓存(`_early`),块耗尽即发送
  端自然背压,无需显式 credit。

### 3.2 发送路径(边 UP / 云 DOWN)

```
计算线程 channel.submit:
  snapshot clone(producer 流,防 staging buffer 复用) → 队列 put
IO 线程 _do_send:
  ① 取空闲块(无块时 poll CQ 等回收)
  ② 同步 D2H:.to("cpu", non_blocking=False),包在
     torch.npu.stream(通道流) + inference_mode 内
     ——channel 流在提交侧已 wait_stream(producer),D2H 读到的
     必是完整 clone(字节即发即正确,Phase1 语义)
  ③ memmove:36B 头 + payload + aux 连续落块
  ④ CRC32(payload+aux) 回填帧头(块内自洽)
  ⑤ 单 SGE(块地址+total+lkey) post_send(SIGNALED),
     future._bind(HostDoneEvent) 先于任何可查完成
send CQE(RC 下即对端 ACK):
  HostDoneEvent.set() → future 完成;块回收进 _free
```

### 3.3 接收路径(云 UP / 边 DOWN)

```
wire up 时:N 块全部预挂 post_recv(每块整块 BLOCK_BYTES 为 SGE 长度)
IO 线程 poll CQE(opcode=RECV, byte_len=帧长):
  ① 头校验:magic / 长度自洽,失败即 broken
  ② 帧级 CRC 复核(ok/bad 计数,bad 打 error 不中断)
  ③ skipped seqno → 物理丢弃(直接重挂 post_recv)
  ④ seqno 配对:hit _expects(submit_recv 已登记)→ 立即交付;
     miss(数据面快于控制面)→ 缓存 _early 等任务
  ⑤ 交付 _complete_recv:
     尺寸对账(payload_len//2 == buffer.numel 等)
     torch.frombuffer 对注册块做零拷贝视图(bf16/int64)
     H2D:buffer.copy_(view, non_blocking=True) 于通道流 +
          torch.npu.Event().record()
     buffer.record_stream(通道流) + _lwd_seqno 标签
     (分配在计算流、写入在通道流——不登记则分配器跨流复用判定
      缺信息,存在跨 seqno 覆写窗口,真机实录即此处串数据)
     future._bind(done_event=NPU event, tensor=buffer,
                  keepalive=(blk,))
  ⑥ 块回收:NPU event query()==True 后重挂 post_recv
     (_pending_h2d 列表,每次 poll 拍检查)
```

### 3.4 帧预算预检(lwd_wire 启动期)

```
frame_bound = max_num_batched_tokens × (hidden×2 + Hash专家最坏512 + aux24)
frame_bound > _BLOCK_BYTES → 启动即报错
```

真机录案:64MB 块时代 bench 大 prompt(5013 token chunk,帧 72MB)
超块,发送线程 broken 后对端 60s unbound 才暴露——预检把这个失败
前移到启动第一秒。预算注意:预检公式用 +28B(v1 头),v2 头 36B,
差 8B 被 512B/token 的专家余量覆盖,不影响判定。

## 4. 线程模型

每个 (channel, op) 一条 IO 线程(与 SMC 相同的 FIFO 保序骨架),
提交接口永不阻塞;差异在"字节流主动读"变为"CQE 推送",IO 线程
从"收帧驱动"变为"事件循环驱动":

| 线程 | 端 | 职责 |
|---|---|---|
| worker 计算线程 | 双端 | EMBED/UNEMBED/采样;只做 `service.submit_*`(队列 put)+ snapshot clone |
| `lwd-verbs-{ch}/{op}` IO 线程 | 双端 | 5ms tick 循环:任务出队 → `_do_send/_do_recv`;`_poll_and_complete`(CQE 处理:send 回收+置事件 / recv 校验配对交付);`_pending_h2d` 块回收重挂;5s stall watchdog;5s 聚合统计 |
| `lwd-verbs-connect-*` | 边 | 握手重试(0.2s 起指数退避封顶 5s,无限;ECONNREFUSED 提示对端 listener 未就绪);成功 = TCP 握手 + QP RTR/RTS,随后 `_wire_up` 建块池、预挂 recv、启动 IO 线程 |
| `lwd-verbs-hs-{port}` acceptor | 云 | 一次性 listen+accept(每 QP 一条),accept 后 `attach_fd` 完成对称握手与 QP 迁移 |
| future 消费 | 双端 | 云 runner `future.wait(60)`(host 门);边 `wait_for_comm()`(先等 `_bind` 句柄就位,再 current_stream.wait_event(NPU event) 做设备序) |

要点:

1. **5ms 轮询拍**:IO 线程 `queue.get(timeout=0.005)` 与 poll 合一,
   完成处理平均量化延迟 ~2.5ms(25ms→2.5ms 的真机调优结论),唤醒
   200 次/s,CPU 代价可忽略;
2. **提交/完成解耦**:计算线程永不接触 verbs;`qwait` 统计段
   (提交→IO 线程取到)直接量化积压/GIL 争抢;
3. **stall watchdog**(5s 快照,error 级):`_early` 滞留(数据面快于
   控制面,块被早到帧占用)/`pending_h2d`(NPU event 不翻转,块不
   回收)/`free=0`(发送 CQE 未回收)/`expects`(帧未到达)四类
   卡点各带定位短语——future unbound 问题的第一现场;
4. GIL:`post_send/poll_cq`(ctypes C 调用)与 D2H/H2D(CANN op)
   在 C 层放锁;`.to("cpu")` 同步拷期间 worker 计算线程可继续前向
   (仅通道流设备侧被占用)。

## 5. 边云通信(协议配对、skip、错误传播、控制面)

- **seqno 配对契约不变**:UP seqno=控制面 `LwdRangeNotify.seqno`
  (边引擎派发 EMBED 批时分配,随 SchedulerOutput 到边 worker 与云
  worker);DOWN seqno=云 worker 发送时 `_lwd_next_down_seqno()` 自增,
  经 `LwdC2eNotify.down_seqno` 通告边侧 recv。verbs 下帧头 seqno 仅
  防御校验,`_expects/_early` 按 seqno 迟后匹配(数据先到是常态:
  控制面 ZMQ 与数据面 RDMA 两条路,前者跨公网更慢)。
- **skip(abort hole)**:控制面双端 abort → `channel.skip_seqno` →
  `VerbsIoThread.add_skipped` → 收到该 seqno 帧 CQE 时校验 CRC 丢弃
  并重挂块(物理丢帧,比 SMC 的 drain 更彻底);发送侧已 post 的帧
  不撤回,靠对端丢弃。
- **断线爆炸半径**:QP CQE status != SUCCESS / 帧头错位 / 尺寸不
  对账 → `_broken=True` + `_fail_all_pending`(`_expects` 全部 ERROR、
  `_early` 块重挂),后续 submit 的任务立即 ERROR;**不自动重连**
  (请求级重试由上层 abort/重派处理,与 HCCL 域失败同语义)。
- **控制面双轨沿用**:云引擎 HELLO 填 `data_plane_host/port_base`
  (lwd_cloud_engine L192-207),边引擎收 HELLO 校验本地配置
  (lwd_edge_engine L209-227,不一致告警不拦截)。寻址三源:
  `lwd_config` 段 → env 覆盖(`VLLM_ASCEND_LWD_DATA_PLANE_HOST/PORT/
  FAMILY`)→ 缺省 host 随 `pre_out_host`。

## 6. prefill(边)/decode(云)端到端流程

### 6.1 UP:边 embed → 云注入(prefill chunk 流)

```
边引擎:请求入队,按 max_num_batched_tokens 分 chunk,seqno 随
        LwdRangeNotify 通告云侧;边 worker execute_model 收
        LwdEmbedBatch:
  1. token ids 扁平化 → pinned 环(4 槽)non_blocking H2D
     (替代 pageable 同步拷,避免主流等设备排空)
  2. model.embed_input_ids(纯文本)或 +mm merge(视觉塔输出按
     mm_hash 共享缓存,同图复用不重编)
  3. DeepSeek V4:tid2eid 表查 Hash 专家 ID,pack_hash_payload
     拼进 payload;has_mrope 请求的 [n,3] 位置拼 aux
  4. service.submit_send(UP, seqno) → snapshot clone →
     verbs IO 线程:D2H 落块 → CRC → post_send
云 worker _lwd_up_post_recvs(execute_model 前段):
  5. 端点(rank 0)按控制面通告尺寸 submit_recv(UP, seqno)
     → IO 线程登记 _expects;帧 CQE 到达即校验配对交付
     (数据先到则 _early 缓存)→ H2D 落 NPU buffer + NPU event
云 runner _lwd_harvest_up_batches(_prepare_inputs 内,消费时刻):
  6. 端点 future.wait(60) 过 host 门(超时带逐请求 token/offset
     现场报错);窗口核验(chunk rows == scheduled,不等即
     fail-fast 防静默乱码)
  7. TP 组内现场 broadcast(launch+wait 原子对,torch_npu 跨流
     排序不可依赖——demo 纪律);aux(mrope)第二段独立广播
  8. 注入:embeds 直写 inputs_embeds.gpu 各请求调度窗口 +
     unpack_hash_payload 喂 lwd_hash_state;mrope 行写 req_state
  9. 云全量层 forward(prefill chunk)+ 后续 decode 步
```

### 6.2 DOWN:云采样 → 边 rank-replay(每步 decode)

```
云 runner sample_tokens → _lwd_collect_down_payload:
  1. hidden 组包(有效请求行;spec 路径按段打包含被拒行)
  2. 全局秩 ranks = (logits > thresh).sum(1)(bf16 直比);
     meta = [ranks|counts|seg_lens] 阻塞 D2H 落每步私有 CPU 张量
     (event 同步在 NPU 栈上不可靠,真机定案;仅 ~KB)
云 worker sample_tokens 尾段:
  3. seqno = _lwd_next_down_seqno();submit_send(DOWN, hidden)
     → verbs IO 线程同 UP 路径发出
  4. lwd_c2e_meta(ranks/token 直通值/num_accept)经控制面
     LwdC2eNotify 通告边引擎 → 边调度派 LwdUnembedBatch
边 worker _execute_lwd_unembed:
  5. submit_recv(DOWN, seqno) → wait_for_comm(设备序)→
     result.tensor 为 NPU hidden
  6. model.compute_logits(hidden)(边侧 lm_head,词表大头在边)
  7. select_token_batch:按云侧通告的 rank r_i 在排序 logits 中
     取名(rank-replay,免传整行 vocab 分布)
  8. spec:按 num_accept_tokens 截取有效前缀 → ModelRunnerOutput
```

一上一下各占一条独立 QP/IO 线程,互不阻塞;请求结束由控制面
finished 流转,数据面无 FIN 帧(最后一步 DOWN 包随步自然终结)。

## 7. 部署与环境

| 项 | 值/说明 |
|---|---|
| 依赖 | `libibverbs.so.1`(能跑 `ib_write_bw` 即具备);无 pyverbs/NIXL |
| family 切换 | `data_plane_family: verbs` 或 env `VLLM_ASCEND_LWD_DATA_PLANE_FAMILY=verbs` |
| 端口 | 数据 `port_base` 默认 5560(SMC 占 base/base+1);verbs 握手 = base+2(UP)/base+3(DOWN),TCP 仅握手用,完即关 |
| 块池 env | `VLLM_ASCEND_LWD_VERBS_BLOCK_MB=128`、`N_BLOCKS=4`(两端必须一致,握手校验) |
| CRC env | `VLLM_ASCEND_LWD_VERBS_CRC=0/1/2` |
| 网络选路 env | `VLLM_ASCEND_LWD_VERBS_DEV`(设备名过滤)、`VERBS_GID_INDEX`(v2 自动探测不准时) |
| QP/GRH env | `QP_TIMEOUT=16`、`TC=128`、`HOP=64` |
| ABI | `VLLM_ASCEND_LWD_VERBS_ABI=hw`(默认,华为定制 verbs.h)/`std`(rdma-core) |
| memlock | 块池走 ibv_reg_mr 自带锁页,仍建议核对 `ulimit -l` ≥ 块池总量 |

## 8. 性能分析

单帧(UP,512×7168 bf16 ≈ 7MiB + 头,200G RoCE,PCIe ~25GB/s):

| 段 | 预算 | 说明 |
|---|---|---|
| snapshot clone | ~0.3ms | producer 流,与计算重叠 |
| D2H + memmove + CRC | ~0.5ms | 同步拷在 IO 线程,worker 不等 |
| RC SEND(post→CQE) | ~0.5ms | 线速,CQE=对端 ACK(比 SMC 的 sendall 返回更精确) |
| 云校验+零拷贝视图 | <0.1ms | frombuffer 无拷贝 |
| H2D | ~0.3ms | 通道流,与云 TP 广播衔接 |

相对 SMC:省掉 RMBE/CDC 逐块确认(拉远 RTT 下 SMC 上限 ~0.8Gbps,
verbs 同链路 ~9.4Gbps)与内核 sndbuf/RMB 两次拷贝;5ms tick 引入
~2.5ms 平均完成量化延迟(可调小 tick 换 CPU)。

优化路线(按需,均不改帧格式):
P1 发送 D2H 换 non_blocking + 通道流 event 同步(消除 IO 线程同步
拷占用);P2 CRC 换表驱动/关断(线上默认开,稳态可 0);P3 大帧
分块多 WR 并行(块内分段 post_send,重叠 D2H 与线缆);P4 拉远
高 RTT 下评估 RDMA WRITE_WITH_IMM 单边(免接收侧 CQE 消耗 recv WR,
端到端延迟再降);P5 多边端时按边端建 QP 组(每 (edge, channel)
独立 QP,共享 PD/CQ)。

## 9. 已知坑(真机录案,部署/评审必读)

1. **华为定制 verbs ABI**:`ibv_qp_attr_mask` 为稀疏位图、结构体布局
   与 rdma-core 不同、部分枚举值不同;`ibv_post_send/recv/poll_cq`
   是 verbs.h static inline 不导出——hw ABI 下走 `ibv_context.ops`
   函数指针表(实测偏移 poll_cq@88/post_send@200/post_recv@208),
   cmd 层 `ibv_cmd_post_recv` 返回 EOPNOTSUPP;`modify_qp` 报
   errno=22 时对照部署机 `/usr/include/infiniband/verbs.h`。ABI
   常量与结构体在 verbs_wire.py 内按 `ABI=hw/std` 双份维护。
2. **qp_num 偏移跟用户态库走**:华为定制 libibverbs 的 `qp_num@52`,
   标准在 20;按网卡名切 offset 曾读出 0xFFFF 哨兵,握手交换垃圾
   QPN 后两端互不认包。
3. **gid 必须 RoCE v2**:v1 纯以太封装跨三层必丢;CX5 同一 IPv4 注册
   v1/v2 两条,自动探测已按 sysfs 类型过滤。
4. **hop_limit=1 在拉远多跳第一跳即被丢**,RC 无限重传表现为 send
   CQE 超时;默认 64。
5. **torch pin_memory 分配的 CANN dvmm 内存注册失败**,块池必须
   bytearray + reg_mr。
6. **帧超块**:启动预检拦截(§3.4);两端 BLOCK_MB 不一致在握手
   fail-fast。
7. **早到帧占块**:控制面慢于数据面时 `_early` 占用接收块,块耗尽
   后对端 RNR(已设无限 rnr 重试,表现为吞吐塌陷)——watchdog 的
   `early=` 快照即此症状,根因通常在控制面 ZMQ 通路。

## 10. 测试计划

1. **单元(verbs_wire)**:帧编解码 round-trip;skipped 帧 CQE 物理
   丢弃;错位/短帧 broken 传播;早到-迟配对;块回收重挂;握手
   BLOCK_MB 不一致 fail-fast。
2. **双机链路**:先 `af_smc_demo --family verbs`(verbs_demo.py 同款
   模型)对拍 `ib_send_bw` 基线;确认 gid=v2、路由口正确。
3. **数值回归(1E1C)**:`data_plane_family` 在 smc/verbs 间切换对拍,
   `dump_tensor` 边云两端成对输出 sum/mean/head10/tail10 逐请求
   一致;CRC=2 深度模式跑一轮(三段定位零 mismatch)。
4. **异常路径**:请求中途 abort(双端 skip、CQE 丢弃);云引擎重启
   (QP broken → 边侧 ERROR → 请求级重试);mrope 混批(aux 变长);
   帧预算预检(超大 max_num_batched_tokens 启动报错)。
5. **拉远压测**:高 RTT(≥5ms)链路 SMC vs verbs 吞吐对照(动机
   验证);`[Lwd][stats]` 分段(send_prep/send_wire/recv_h2d/
   recv_early/qwait)+ stall watchdog 空转零输出;MTP 接受率与
   HCCL 基线一致性(隐性数值回归指标)。

## 11. 实施清单(文件 × 动作)

| # | 文件 | 动作 |
|---|---|---|
| 1 | `vllm_ascend/distributed/lwd_comm/verbs_wire.py` | 新增(§2-§5 全量) |
| 2 | `vllm_ascend/distributed/lwd_wire.py` | verbs 工厂分支(握手端口 +2/+3)+ 启动帧预算预检 |
| 3 | `vllm_ascend/distributed/lwd_comm/smc_wire.py` | 复用其 TCP ctypes 帮手(`c_open/c_connect/c_accept/c_bind/c_listen/c_close`)、`HostDoneEvent`、`load_data_plane_config`;零结构改动 |
| 4 | `channel.py`/`future.py`/`service.py`/`types.py`/三个 worker/控制面 | **零改动**(验证性确认) |
| 5 | 部署 | `data_plane_family=verbs` + 两端 `BLOCK_MB/N_BLOCKS` 对齐 + 端口放行 base+2/+3(TCP)与 RoCE 流量 |

实施顺序:1→2(demo 已验证 verbs 数据面连通/吞吐)→ 3-4 确认 →
测试计划 2→3→4→5。
