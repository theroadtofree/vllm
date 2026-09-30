# 边云数据面替换方案:HCCL hidden 传输 → 控制面拉起的 AF_SMC(详细版)

> 适用仓库:`C:\cann\vllm-ascend-github`(数据面 wire 层 / worker)、
> `C:\cann\vllm-github`(控制面 / 装配配置)。参考实现:[af_smc_demo](README.md),
> 帧协议与 demo 完全一致。**本方案直接替换 HCCL,不保留回退路径。**

## 0. 摘要

数据面 wire 层的两个文件(`lwd_wire.py` 建组、`channel.py` 的
`_wire_send/_wire_recv`)从 HCCL `dist.isend/irecv` 替换为 AF_SMC 字节流;
控制面新增数据面 endpoint 通告(HELLO additive 字段)与静态配置双轨寻址;
`service.py`/`types.py`/`future.py`/三个 worker 的提交与消费接口**零改动**。
跨机段 SMC 化,**云内 TP 组内扩散保持 HCCL broadcast 不动**。

```
【现状】                                    【目标】
边 NPU hidden                                边 NPU hidden
  │ snapshot clone(NPU)                       │ snapshot clone(NPU)
  │ HCCL isend ──RoCE──→ 云端点 NPU buffer     │ D2H → pinned host(ch_stream)
  │                (new_group 通信域)          │ AF_SMC sendall ──RoCEv2──→
  ▼                                           ▼
云内 TP broadcast(HCCL,不动)              云 pinned host → H2D(NPU, ch_stream)
                                             ▼
                                          云内 TP broadcast(HCCL,不动)
```

## 1. 现状代码地图(修改前最后确认)

### 1.1 数据面(vllm-ascend)

| 位置 | 现职责 | 本方案 |
|---|---|---|
| `distributed/lwd_wire.py` | 2×`dist.new_group`(HCCL 域)+2×NPU stream+warmup+endpoint rank 解析 | **重写**:group→SMC 连接;stream/rank 解析保留 |
| `distributed/lwd_comm/channel.py` | per-(channel,op) FIFO、seqno 重排、snapshot、`_wire_send/_wire_recv`(`isend/irecv`)、`_bridge_and_record` | **改 3 个函数**(`_execute/_wire_send/_wire_recv`),删 `_bridge_and_record` |
| `distributed/lwd_comm/future.py` | `done()/wait()/wait_for_comm()/_bind/_finalize` | 1 处类型注解放宽 |
| `distributed/lwd_comm/service.py` | singleton、submit/poll/skip/shutdown | 零改动 |
| `distributed/lwd_comm/types.py` | `LwdCommRequest/Result/ChannelType` | 零改动 |
| `worker/lwd_edge_worker.py` | `init_device`→`init_lwd_duplex_channels`(L149);UP `submit_send`(L326-334);DOWN `submit_recv`+`wait_for_comm`(L477-488) | 零改动 |
| `worker/lwd_cloud/lwd_cloud_worker.py` | `init_device`→`init_lwd_duplex_channels`(L103);UP `submit_recv`(L192-216);DOWN `submit_send`(L244-252);down_seqno 分配(L322) | 零改动 |
| `worker/lwd_cloud/lwd_cloud_model_runner.py` | UP 消费 `future.wait(60)`(L458)+云内 broadcast(L480/497) | 零改动 |

### 1.2 控制面与配置(vllm 主仓)

| 位置 | 现职责 | 本方案 |
|---|---|---|
| `v1/lwd_control/control_communication/lwd_notify.py` | `LwdHelloNotify`(云→边发现通告,pre_out 端点)/`LwdRangeNotify.seqno`(UP 配对)/`LwdC2eNotify.down_seqno`(DOWN 配对) | `LwdHelloNotify` 加 2 个 additive 字段 |
| `v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py` | L101 构造 HELLO,首拍 `_lwd_announce()` | 填充数据面 endpoint 新字段 |
| `v1/lwd_control/control_edge_scheduler/lwd_edge_engine.py` | L162 消费 HELLO→retarget PRE_OUT | 新增数据面 endpoint 校验/日志 |
| `v1/lwd_control/control_edge_scheduler/lwd_edge_assemble.py` | `LwdConfig` 装配(additional_config `lwd_config` 段 + `VLLM_ASCEND_LWD_*` env) | 加 3 个数据面字段 |

### 1.3 必须保留的语义不变量(方案的红线)

1. `submit_send/submit_recv` **立即返回**(提交线程不阻塞)——现网
   `handle.wait()` 即返,SMC 下只做队列 put,不劣化;
2. recv future 完成 = **数据已在 NPU buffer 可用**(NPU event)——云侧
   `future.wait(60)`、边侧 `wait_for_comm()` 语义原样;
3. per-(channel, op) seqno FIFO 配对:UP `batch.seqno`、DOWN
   `down_seqno`,两端同源(控制面通告),帧头仅防御校验;
4. send snapshot(clone)保 producer buffer 复用安全;keepalive 由
   lazy reap 释放;
5. `skip_seqno` abort hole:两端同 skip(SMC 下接收端还能物理丢帧,更强);
6. 云内 TP broadcast 消费时刻现场发起(runner L480 注释:torch_npu 跨流
   排序不可依赖)——与跨机段解耦,不动。

## 2. 帧协议(demo 同款,定稿)

```
偏移  长度  字段          语义
0     4    magic         b"LWD1";版本化预留(扩展载荷结构时升 LWD2,
                          新旧两端 fail-fast 而非错解析)
4     8    seqno         uint64 网络序;UP=边侧 dispatch seqno(=控制面
                          LwdRangeNotify.seqno),DOWN=云侧 down_seqno
                          (=LwdC2eNotify.down_seqno)。接收端 IO 线程
                          对账:seqno∈skipped → 物理丢帧;≠期望且未skip
                          → pairing error,连接标记 broken
12    8    payload_len   字节数 = num_elements × 2(bf16;UP 含 Hash
                          专家 ID 段,由 hash_payload_numel 统一推导,
                          帧层不感知)
20    8    aux_len       字节数 = aux_num_elements × 8(int64 mrope
                          [n,3],仅 UP;0 = 无 aux 帧)
28    ...  payload+aux   字节连续;aux 段只含 has_mrope 标记请求的行
                          (Σ n_i×3,与边侧组帧同源同序)
```

- 收发两端各自从**控制面元数据**推导 `num_elements/aux_num_elements`
  (现网逻辑不变),帧头自描述仅用于校验与丢帧,不引入新的控制依赖。
- 字节流天然保序:`_held` 重排缓冲退化为直通(代码保留,恒空触发)。
- 头部 28B 与 demo `HEADER = struct.Struct("!4sQQQ")` 一致。

## 3. 连接拓扑与生命周期("控制面拉起数据面")

### 3.1 拓扑

```
云侧(两个 endpoint listen,均 bind data_plane_host):
  UP   = data_plane_host : data_plane_port_base      (边 connect;边写云读)
  DOWN = data_plane_host : data_plane_port_base + 1  (边 connect;云写边读)
边侧(两条都 connect);方向由谁读谁写区分,与 LwdChannelType 语义一致。
interior rank(多余云 TP rank)不建任何 socket(不再参与 world 级
new_group;init 末尾保留一次 get_world_group().barrier() 对齐)。
```

### 3.2 寻址:静态配置 + HELLO 通告双轨(与 pre_out 同构)

`pre_out_host` 的既有模式就是"云侧 bind 配置静态值,HELLO 通告给边侧"。
数据面完全同构:

- **静态轨**(建连依据):`lwd_config` 段新增字段(见 §4.1),
  云侧 worker bind、边侧 worker connect 都读它;
- **动态轨**(校验/未来动态化):`LwdHelloNotify` additive 新字段
  `data_plane_host/data_plane_port_base`,边引擎收到后与本地配置比对,
  不一致仅告警不拦截(部署错位可观测)。

### 3.3 时序(为何天然成立)

```
云: worker init_device → SMC listen(两条) → …权重/KV/图编译… →
    云引擎 _lwd_setup_zmq 构造 HELLO(从 lwd_wire 全局取实际 bind 的
    endpoint 填新字段)→ 首拍 _lwd_announce()
边: worker init_device → SMC 仅建 IO 线程(FUTURE 状态,不 connect)
    → …边引擎起 → 收 HELLO(retarget PRE_OUT,现有逻辑)→ 数据面
    endpoint 校验(新增)
数据: 边首个 LWD_EMBED batch 的 submit_send 进入 IO 线程队列 →
    IO 线程惰性 connect(失败退避重试)→ 连接成功(建连本身即
    TCP 握手+CLC 协商+RDMA 链路建立,等价原 warmup 目的)→
    首帧发出
```

边引擎等 HELLO 预算 600s(`LWD_HELLO_TIMEOUT_S_DEFAULT`,覆盖云全量
启动)——边 worker 的 IO 线程重试窗口覆盖同一区间即可(重试无上限,
`hello_event` 未 set 前 scheduler 不会派发 LWD batch,队列不会积压)。
**原 `warmup_lwd_duplex_channels` 的"把首用 rendezvous 移出 pipeline"
目的由 IO 线程 queue-until-connected 达成**,init 不再需要阻塞 warmup。

## 4. 控制面与配置改动明细

### 4.1 `lwd_edge_assemble.py`:LwdConfig 加 3 字段

```python
LWD_DATA_PLANE_PORT_DEFAULT = LWD_POST_OUT_PORT_DEFAULT + 2  # 19902 起

@dataclass(frozen=True)
class LwdConfig:
    ...
    data_plane_host: str = ""        # 缺省 = pre_out_host(同一 RoCE 网段)
    data_plane_port_base: int = LWD_DATA_PLANE_PORT_DEFAULT
    data_plane_family: str = "smc"   # "smc"|"tcp";tcp 仅排障对比用

    def lwd_data_plane_up_endpoint(self) -> tuple[str, int]:
        host = self.data_plane_host or self.pre_out_host
        return host, self.data_plane_port_base
```

`from_env_and_config` 同步读段字段
(`data_plane_host/data_plane_port_base/data_plane_family`);
`_lwd_apply_env_overrides` 加
`VLLM_ASCEND_LWD_DATA_PLANE_HOST/PORT/FAMILY` 三项。
**校验**:云侧角色下 `data_plane_host` 不得为空/`0.0.0.0`(必须边可
路由真实 IP,同 `pre_out_host` 的既有约束)。

### 4.2 `lwd_notify.py`:LwdHelloNotify additive 扩展

```python
class LwdHelloNotify(msgspec.Struct, gc=False, tag=True):
    pre_out_host: str
    pre_out_port: int
    ...  # 现有字段不动
    data_plane_host: str = ""      # 缺省 "" = 旧版云侧(additive 兼容)
    data_plane_port_base: int = 0
```

msgspec 带默认字段 additive,新旧云边混跑不破协议。

### 4.3 `lwd_cloud_engine.py`:HELLO 填充

`_lwd_setup_zmq`(L101 附近):

```python
from vllm_ascend.distributed import lwd_wire  # 仅云进程可 import
dp_host, dp_port = lwd_wire.get_lwd_data_plane_endpoint()  # listen 实际值
self._lwd_hello = LwdHelloNotify(
    ...,
    data_plane_host=dp_host,
    data_plane_port_base=dp_port,
)
```

### 4.4 `lwd_edge_engine.py`:HELLO 消费校验

L162 `isinstance(msg, LwdHelloNotify)` 分支内追加:

```python
if msg.data_plane_port_base:
    local = LwdConfig...  # 装配期已合成的边侧配置
    if (msg.data_plane_host, msg.data_plane_port_base) != local_pair:
        logger.warning("[Lwd] data-plane endpoint mismatch: hello=%s:%d "
                       "local=%s:%d(以 local 建连)", ...)
```

边引擎与边 worker 分属两个进程(multiproc executor),引擎侧**只校验
不建连**——建连发生在 worker 进程的 IO 线程(读装配配置,§5.1)。

## 5. 数据面改动明细(核心)

### 5.1 新增 `vllm_ascend/distributed/lwd_comm/smc_wire.py`

一个文件,四件事:socket 工厂、帧编解码、per-(channel,op) IO 线程、
host 完成事件。关键实现(可直接落码的骨架):

```python
"""AF_SMC 数据面 wire:连接 + 帧协议 + per-channel IO 线程。

替代 HCCL isend/irecv 的跨机段;云内 TP 扩散不受本文件影响。
线程模型:每个 (channel, op) 一条 IO 线程(FIFO 保序),worker 计算
线程只做 queue.put;socket 调用与 D2H/H2D 拷贝全部发生在 IO 线程。
"""
import socket, struct, threading, queue, time
import torch
from vllm.logger import logger
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

AF_SMC = getattr(socket, "AF_SMC", 43)
IPPROTO_SMC = getattr(socket, "IPPROTO_SMC", 265)
_HEADER = struct.Struct("!4sQQQ")     # 与 af_smc_demo/smc_common.py 一致
_MAGIC = b"LWD1"

class HostDoneEvent:
    """发送 future 的完成事件:query() 兼容 LwdCommFuture 的
    done()/wait()/_finalize 轮询路径。发送侧无 wait_for_comm 调用方
    (已核实:lwd_edge_worker.py:487 为 DOWN recv;runner:480 为
    broadcast work),wait_event 提供防御性 no-op。"""
    def __init__(self):
        self._flag = threading.Event()
    def set(self): self._flag.set()
    def query(self): return self._flag.is_set()
    def wait_event(self, _):  # 防御:no-op + warning
        logger.warning("[lwd-smc] wait_for_comm on a host-done event")

class _Task:  # 队列元素
    __slots__ = ("seqno", "tensor", "aux_tensor", "future",
                 "req", "buffer", "aux_buffer", "cancelled")
    ...

class SmcIoThread:
    def __init__(self, channel: LwdChannelType, op: str, family: str):
        self._sock = None
        self._broken = False
        self._skipped: set[int] = set()          # 与 channel.skip_seqno 同步
        self._q: "queue.Queue[_Task | None]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name=f"lwd-smc-{channel.value}-{op}",
            daemon=True)
        self._thread.start()

    # ---- 连接管理(云侧由 lwd_wire 预先 listen/accept 后注入) ----
    def attach(self, sock: socket.socket) -> None: ...   # 云侧 recv 线程
    def connect_loop(self, host: str, port: int) -> None: ...
    # 边侧 IO 线程内:退避重试(0.2s 起,封顶 5s)直至成功;
    # 成功 = TCP 握手+CLC 协商+RDMA 链路建立(即原 warmup 语义)

    # ---- 提交接口(计算线程调用,永不阻塞) ----
    def submit_send(self, seqno, tensor, aux_tensor, future, req):
        self._q.put(_Task(seqno=seqno, tensor=tensor,
                          aux_tensor=aux_tensor, future=future, req=req))
    def submit_recv(self, seqno, req, buffer, aux_buffer, future):
        self._q.put(_Task(...))

    def add_skipped(self, seqno: int) -> None:
        with self._skipped_lock: self._skipped.add(seqno)

    # ---- IO 主循环 ----
    def _run(self):
        while True:
            task = self._q.get()
            if task is None: break               # shutdown 令
            if self._broken or task.cancelled:   # 断连/取消:future 失败
                task.future._finalize(error=RuntimeError("smc link broken"))
                continue
            try:
                if task.req.op == "send": self._do_send(task)
                else: self._do_recv(task)
            except Exception as exc:             # 含半帧 ConnectionError
                self._broken = True
                task.future._finalize(error=exc)
                logger.error("[lwd-smc] %s %s broken: %r",
                             self._channel, self._op, exc)

    # ---- 发送:D2H → sendall ----
    def _do_send(self, t):
        stream = lwd_wire.get_lwd_channel_stream(self._channel)
        with torch.npu.stream(stream):
            # producer→channel 排序在提交侧已完成(见 channel._execute);
            # 同 stream 连续 copy 的先后由 stream FIFO 保证
            host = t.tensor.to("cpu", non_blocking=False)   # Phase1 同步拷
            host_aux = (t.aux_tensor.to("cpu") if t.aux_tensor is not None
                        else None)
        hdr = _HEADER.pack(_MAGIC, t.seqno,
                           host.numel() * host.element_size(),
                           0 if host_aux is None
                           else host_aux.numel() * host_aux.element_size())
        self._sock.sendall(hdr)
        self._sock.sendall(host.view(torch.uint8).numpy())  # buffer 协议零拷贝
        if host_aux is not None:
            self._sock.sendall(host_aux.view(torch.uint8).numpy())
        ev = HostDoneEvent()
        t.future._bind(handles=[], done_event=ev,
                       tensor=None, keepalive=None, aux_tensor=None)
        ev.set()

    # ---- 接收:recv_into → H2D → NPU event ----
    def _do_recv(self, t):
        # 内联循环:一次 _do_recv 可消化多个被 skip 的帧(帧自描述,
        # drain 后继续读下一帧),直到命中本 task 的 seqno
        while True:
            seqno, plen, alen = self._recv_header()
            if seqno == t.seqno:
                break
            if seqno in self._pop_skipped(seqno):   # 已 skip:物理丢弃
                self._drain(plen + alen)
                continue
            raise ConnectionError(                  # 错位:配对契约破坏
                f"pairing error on {self._channel}: expect {t.seqno}, "
                f"got {seqno}")
        host = torch.empty(plen // 2, dtype=torch.bfloat16, pin_memory=True)
        self._recv_into_tensor(host)
        aux = None
        if alen:
            aux = torch.empty(alen // 8, dtype=torch.int64, pin_memory=True)
            self._recv_into_tensor(aux)
        stream = lwd_wire.get_lwd_channel_stream(self._channel)
        with torch.npu.stream(stream):
            t.buffer.copy_(host, non_blocking=True)
            if aux is not None: t.aux_buffer.copy_(aux, non_blocking=True)
            done = torch.npu.Event(); done.record()
        t.future._bind(handles=[], done_event=done,
                       tensor=t.buffer, keepalive=(host, aux),
                       aux_tensor=t.aux_buffer)

    def _recv_into_tensor(self, host_t) -> None:
        mv = memoryview(host_t.view(torch.uint8).numpy())
        got = 0
        while got < len(mv):
            n = self._sock.recv_into(mv[got:])
            if n == 0: raise ConnectionError("peer closed mid-frame")
            got += n
```

实现注意(评审要点):

1. **skip 帧的读取不能"回队尾"**:`_do_recv` 内联循环——读到 skipped
   seqno 时直接 `_drain` 该帧字节并**继续在本循环读下一帧**,直到读到
   非 skipped 帧为止(帧是自描述的,一个 IO 循环可消化多帧);对应的
   future 是队列里的下一个 task,读到的帧 seqno 必须等于该 task 的
   seqno(或 ∈ skipped),否则 pairing error。
2. **recv 的 keepalive**:`_bind(keepalive=(host, aux))` 持有 pinned
   host 缓冲直至 future finalize——`non_blocking` H2D 的正确性由
   channel stream 上 `done.record()` 之后的 event.query()(=H2D 真实
   完成)保证,finalize 发生在 reap(query True)之后,时序闭合。
   Phase2 优化:两个尺寸档位的 pinned 环形池,finalize 归还。
3. **send 的 D2H 同步拷(Phase1)**:`.to("cpu")` 在 IO 线程阻塞至拷贝
   完成,字节即发即正确;Phase2 换 `non_blocking + pinned 池 +
   d2h_event.synchronize()`,语义不变。
4. **`view(torch.uint8).numpy()`**:bf16/int64 tensor 的字节视图走
   buffer 协议,`sendall/recv_into` 直接读写,无 tobytes 拷贝。
5. **aux 只在 UP**:与现 `_wire_recv` 的约束一致,DOWN 提交 aux 时
   `submit_send` 抛错(channel 层已有一致检查,保留)。

### 5.2 重写 `lwd_wire.py`(全量结构)

```python
"""prefill_only duplex data-plane wire layer (AF_SMC)。

不再创建任何 HCCL 通信域;云内 TP 扩散(runner 消费时刻的
broadcast)不受影响。跨机段 = 两条 SMC 连接(云 listen/边 connect,
UP=base, DOWN=base+1)+ 每 (channel,op) 一条 IO 线程 + 每通道一条
NPU stream(D2H/H2D 排序)。
"""
_LWD_CHANNEL_STREAMS / _LWD_SMC_IO / _LWD_SMC_SERVER  # 三张注册表
_LWD_ENDPOINTS / _INITIALIZED  # 语义同现在(edge/cloud rank 对)

def init_lwd_duplex_channels() -> None:
    # 1) endpoint rank 解析:与现在完全相同(lwd_config
    #    edge_npu_count>0 → (0, E);否则 PP 组两 rank)
    # 2) family/host/port:从装配 LwdConfig 读(经
    #    get_current_vllm_config().additional_config 段,worker 进程内
    #    与引擎进程读同一配置源)
    # 3) 每个 channel:
    #    - 建 NPU stream(不变)
    #    - 云 endpoint rank:open+setsockopt(REUSEADDR)+bind+listen,
    #      accept 在独立小线程(每连接一次),accept 到的 sock 注入
    #      对应 recv 方 IO 线程(UP 注云 recv 线程,DOWN 注边 recv 线程
    #      —— DOWN 云侧 listen 的 sock 注入云 send 线程方向由
    #      channel 类型决定,见 §3.1)
    #      疑难澄清:UP/DOWN 都是云 listen;云侧 UP=recv 方,
    #      DOWN=send 方;边侧 UP=send 方,DOWN=recv 方
    #    - 边 endpoint rank:两条 IO 线程标记 FUTURE,connect_loop
    #      由 IO 线程首任务触发(或 init 后即后台重试,二选一,
    #      推荐 init 后即重试:连接建立早于首帧,首帧零建连延迟)
    #    - interior rank:仅参与末尾 barrier,不建 socket
    # 4) get_world_group().barrier()(保留唯一全局对齐点)
    _INITIALIZED = True

def get_lwd_data_plane_endpoint() -> tuple[str, int]:
    """云进程实际 bind 的 (host, port_base);云引擎构造 HELLO 时取。"""

def get_lwd_channel_smc_io(channel, op) -> SmcIoThread: ...
def get_lwd_channel_stream(channel) -> torch.npu.Stream: ...  # 不变
def destroy_lwd_duplex_channels() -> None:
    # 各 IO 线程投 None shutdown 令 → join(2s) → close socket;
    # 替代原 destroy_process_group 路径
```

删除:`dist.new_group` 循环、`warmup_lwd_duplex_channels`(连接建立
即 warmup)、`get_lwd_channel_device_group`(唯一调用方 channel.py
同步修改)、`get_lwd_channel_peer`(帧协议无需 rank;日志可打印
对端地址替代)。顺手:`dump_tensor` 现文件里有**两处重复定义**
(L36 与 L204),删除后一处。

### 5.3 修改 `channel.py`(`_execute` 重构 + 两个 wire 函数替换)

**保留不动**:`submit`/`_snapshot_send`/`_submit_sequenced`/
`skip_seqno`/`_advance_past_skipped`/`reap`/`shutdown`/`_reap`/
`_finalize_many`/`_order_after`。FIFO/锁/重排结构原样。

**替换 `_execute`(原 L255-305)**:

```python
def _execute(self, req, predecessor, into=None):
    """投递 IO 线程并返回 deferred future;完成事件由 IO 线程 _bind。"""
    producer_stream = torch.npu.current_stream()
    stream = self._stream()
    future = into if into is not None else LwdCommFuture.deferred(req)
    with torch.npu.stream(stream):
        self._order_after(predecessor)      # recv 链:前驱 NPU event 排序
        if req.op == "send":
            # 快照产生于 producer 流,channel 流后于其再 D2H
            stream.wait_stream(producer_stream)
    smc = lwd_wire.get_lwd_channel_smc_io(self.channel_type, req.op)
    if req.op == "send":
        # snapshot 已在 submit 路径 clone(现逻辑);IO 队列 task 持有
        # 引用直至 sendall 完成,无需额外 keepalive
        smc.submit_send(req.seqno, req.tensor, req.aux_tensor,
                        future, req)
        return future                        # request.tensor 引用转移进 task
    buffer = torch.empty(req.num_elements, dtype=torch.bfloat16,
                         device="npu")
    aux_buffer = None
    if (self.channel_type is LwdChannelType.UP
            and req.aux_num_elements > 0):
        aux_buffer = torch.empty(
            req.aux_num_elements,
            dtype=req.aux_dtype or torch.int64, device="npu")
    smc.submit_recv(req.seqno, req, buffer, aux_buffer, future)
    return future
```

**删除**:`_wire_send`(L317-352)、`_wire_recv`(L354-390)、
`_bridge_and_record`(L392-414)——isend/irecv/handle.wait 全部随
HCCL 路径移除;`[Lwd][perf] bridge-wait` 探针移到
`smc_wire._do_send/_do_recv` 同位置(测 IO 线程内单帧耗时,口径:
send=hdr+payload 离开,recv=帧落地+H2D 完成)。

**返回值契约变化**:原 `_execute` 返回带 tensor 的 future;现在 recv
future 的 tensor 由 IO 线程 `_bind` 注入。消费方(云 runner
`future.wait(60)` 后取 result.tensor;边 `wait_for_comm()+result()`)
都在完成后访问——`_bind` 先于完成状态翻转,契约成立(见 §7 论证 4)。

**`skip_seqno` 增加一行**(L60-82 函数尾):

```python
smc = lwd_wire.get_lwd_channel_smc_io(self.channel_type, self.op)
smc.add_skipped(seqno)
```

发送 FIFO 的 skip 现逻辑(`seqno < _next_seqno` 时 return,不撤回
已 post 的)保留:边已发出的帧,靠云 IO 线程按帧头物理丢弃。

### 5.4 `future.py` 一处放宽

```python
# L39-42 done_event 类型注解
done_event: "torch.npu.Event | HostDoneEvent | None" = None
```

`done()/wait()/_finalize` 只调 `event.query()`,`HostDoneEvent`
满足;`wait_for_comm` 调 `torch.npu.current_stream().wait_event`,
只有 recv future 有调用方且 recv 恒为 NPU event(§1.3-2)。
**无逻辑改动。**

### 5.5 零改动论证(service.py / 三个 worker)

- `service.py` 只做 FIFO 路由(`(channel, op)` 键不变);
- 云 worker `_lwd_up_post_recvs`:尺寸推导(`hash_payload_numel`
  /aux Σ)与 seqno 来源全部不变,submit_recv 签名不变;
- 云 worker `sample_tokens`:DOWN submit_send + `down_seqno` 分配
  不变;`lwd_down_carrier` 的 pinned 视图语义不受影响(它挂在
  云 NPU 侧输出上,与传输层正交);
- 边 worker UP submit/DOWN recv、云 runner 的 `wait(60)`/
  broadcast——见 §1.1 表,均不触碰 wire 内部。

## 6. 同步与时序正确性论证(逐条)

1. **提交线程永不阻塞**:`_execute` 只做 stream 上下文切换 + 队列
   put + NPU buffer 分配(与现在 irecv 路径的 buffer 分配同级)。
2. **D2H 读到完整 snapshot**:snapshot clone 在 producer(计算)流,
   `_execute` 里 `stream.wait_stream(producer_stream)` 使 channel 流
   的 D2H 后于 clone;IO 线程内同步 `.to("cpu")` 落地字节(Phase1)。
3. **H2D 后的可见性**:`buffer.copy_(non_blocking)` + `done.record()`
   同在 channel stream,`event.query()=True` ⇔ H2D 完成;消费者
   `wait_for_comm()`(NPU event→计算流)或 `wait()`(host 轮询)
   两个入口语义与 HCCL irecv 完成完全同构。
4. **_bind 与完成的顺序**:IO 线程先 `_bind(done_event=...)` 再让
   event 变为可查——send:`ev.set()` 在 `_bind` 之后;recv:
   `done.record()` 在 `_bind` 之前但 NPU event 的 query 只有在
   record 之后才可能 True,而 `_bind` 是纯 host 操作先于任何 poll
   返回。`done()` 内 `with self._lock` 检查 PENDING→翻 OK,无竞态
   (future 现有锁结构保证)。
5. **跨帧顺序(同 channel 同 op)**:单 IO 线程 FIFO + 单字节流,
   发送序=seqno 序;对端单 recv IO 线程按到达序消费,字节流保序
   ⇒ 到达序=发送序。`_held` 恒空,机制无害保留。
6. **recv↔send 无交叉**:UP/DOWN 独立连接独立线程,与现在两组
   HCCL 域隔离等价;abort 只 skip 不插队,无乱序注入。
7. **GIL**:sendall/recv_into C 层放锁;D2H/H2D 为 CANN op;
   `.to("cpu")` 同步拷期间 worker 计算线程可继续前向(仅 channel
   stream 设备侧被占用——与现在 HCCL wire 占 channel stream 同构)。
8. **半帧错位**:header 后对端关闭 → `recv_into` 抛
   ConnectionError → link broken → 该 channel 后续 submit 的 future
   立即 ERROR(爆炸半径 = 单边云对,同 HCCL 域失败)。

## 7. 错误处理与生命周期

| 场景 | 行为 |
|---|---|
| 边 connect 前 submit | 任务入队,FUTURE 状态 IO 线程持有;连接成功后按序 flush |
| connect 重试 | 0.2s 起指数退避封顶 5s,无限重试(HELLO 前 scheduler 不派发,无积压风险);日志节流(每 10 次一条) |
| 云 accept 前 | listen 已在 init 完成,accept 线程等待;边重试直至 accept |
| 传输中断线 | §6-8;broken 后 `submit_*` 返回的 future 立即 ERROR;**不自动重连**(请求级重试由上层 abort/重派路径处理,与 HCCL 域失败同语义) |
| shutdown | `destroy_lwd_duplex_channels` → IO 线程 None 令 → join(2s) → close;service.shutdown 现有 drain 逻辑(`channel.shutdown`)照常等 pending future |
| SMC 协商失败 | 内核**自动降级 TCP**(业务不断,性能回退);启动日志打印 `/proc/net/smc` 摘要 + `data_plane_family` 可切 `"tcp"` 做对照排障 |

## 8. 性能分析

**新增成本**(UP, rows×hidden bf16, 200G RoCE,PCIe ~25GB/s):

| 步骤 | 7MiB(512×7168) | 说明 |
|---|---|---|
| 边 D2H | ~0.3ms | channel stream,与计算流并行度受 PCIe |
| SMC 传输 | ~0.5ms | 线速 + SMC 两跳内核拷贝(sndbuf/RMB) |
| 云 H2D | ~0.3ms | 同 D2H |
| **对比 HCCL 直达** | **~+0.8ms/请求** | HCCL ~0.3ms(网卡直达 NPU) |

DOWN 方向同量级(decode 步 hidden 小,增量 ~0.1ms 级)。

**换得**:边侧 NPU/GPU/CPU 通吃(host socket 即可);云侧不需要
NPU RoCE 直达;摆脱 HCCL 建组约束(new_group world 集合同步消失,
interior rank 零参与)。

**优化路径(按需)**:P0 Phase1 同步拷上线;P1 pinned 环形池 +
non_blocking 双向;P2 payload 与 aux 合帧单 sendall(省 syscall);
P3 帧内 zero-copy(recv_into 直接落 pinned)。均不改协议帧格式。

## 9. 测试计划

1. **单元(smc_wire)**:帧编解码 round-trip;skipped 帧物理丢弃;
   半帧 ConnectionError;HostDoneEvent query 语义;IO 线程 broken
   传播。
2. **双机链路**:af_smc_demo 先行(已交付)——确认云侧华为 RoCE
   网卡 SMC 不降级(`/proc/net/smc` 有连接、dmesg 无 CLC 拒绝);
   这是整个方案的**环境前置门槛**。
3. **数值回归(1E1C)**:`LWD_WIRE` 切换后全量跑通,
   `dump_tensor`(`[Lwd][DUMP]` 现成探针)边云两端成对输出
   sum/mean/head10/tail10 逐请求比对=HCCL 基线一致;
   `data_plane_family=tcp` 对照一轮(隔离 SMC 层问题)。
4. **异常路径**:请求 abort 中途(skip_seqno 双端)、云引擎重启
   (断线→边侧 ERROR→请求级重试)、mrope 混批(aux 变长)。
5. **压测**:`[Lwd][perf]` 全套现成探针(TTFT/cloud-step/
   worker-exec-start)对比 HCCL 基线;CPU 占用采样(预期边云两侧
   通信相关 CPU 下降——无协议栈处理)。

## 10. 实施清单(文件 × 动作)

| # | 文件 | 动作 |
|---|---|---|
| 1 | `vllm_ascend/distributed/lwd_comm/smc_wire.py` | **新增**(§5.1 全量) |
| 2 | `vllm_ascend/distributed/lwd_wire.py` | **重写**(§5.2;删 new_group/warmup/dup dump_tensor) |
| 3 | `vllm_ascend/distributed/lwd_comm/channel.py` | 替换 `_execute`;删 `_wire_send/_wire_recv/_bridge_and_record`;`skip_seqno` 加 1 行 |
| 4 | `vllm_ascend/distributed/lwd_comm/future.py` | done_event 类型注解放宽(1 行) |
| 5 | `vllm/v1/lwd_control/.../lwd_edge_assemble.py` | LwdConfig +3 字段 + env override + 校验 |
| 6 | `vllm/v1/lwd_control/.../lwd_notify.py` | LwdHelloNotify +2 additive 字段 |
| 7 | `vllm/v1/lwd_control/.../lwd_cloud_engine.py` | HELLO 填充数据面 endpoint(从 lwd_wire 取) |
| 8 | `vllm/v1/lwd_control/.../lwd_edge_engine.py` | HELLO 分支 +endpoint 校验日志 |
| 9 | `worker/lwd_edge_worker.py`、`worker/lwd_cloud/*`、`service.py`、`types.py` | **零改动**(验证性确认,不走码) |

实施顺序:1→2→3→4(数据面可独立联调,寻址先用 env 静态值)→
5→6→7→8(控制面双轨补齐)→ 测试计划 2→3→4→5。
