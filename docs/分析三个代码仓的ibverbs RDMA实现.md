# 分析三个代码仓的ibverbs RDMA实现

> **Workspace:** e:\cann

---

**User**

E:\cann\Mooncake、E:\cann\hcomm、E:\cann\ucx三个代码仓是不是都有基于ibverbs实现的rdma

**Assistant**

我来并行检查三个仓库中是否存在基于 ibverbs 的 RDMA 实现。

三个仓库都命中了。再确认各自的核心实现文件（而非仅测试/stub）。

hcomm 的文件没有直接匹配到 ibv_ 调用，确认一下它的封装方式。

是的，三个仓库都有基于 ibverbs 的 RDMA 实现，但形态不同：

## 1. Mooncake — 有，自研 RDMA 传输引擎

核心实现在 [rdma_transport](file:///e:/cann/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport) 目录：
- [rdma_transport.cpp](file:///e:/cann/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp)、[rdma_context.cpp](file:///e:/cann/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp)、[rdma_endpoint.cpp](file:///e:/cann/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp) 等，直接 `#include <infiniband/verbs.h>` 使用原生 verbs API
- 另有一套新组件 `tent`（[rdma_transport.h](file:///e:/cann/Mooncake/mooncake-transfer-engine/tent/include/tent/transport/rdma/rdma_transport.h)、[ibv_loader.h](file:///e:/cann/Mooncake/mooncake-transfer-engine/tent/include/tent/transport/rdma/ibv_loader.h)），通过 dlopen 动态加载 libibverbs

## 2. hcomm — 有，通过适配层封装 verbs

- 你打开的 [transport_ibverbs.cc](file:///e:/cann/hcomm/src/platform/resource/transport/host/transport_ibverbs.cc) 就是其实现（`TransportIbverbs` 类），另有设备侧 [transport_device_ibverbs.cc](file:///e:/cann/hcomm/src/platform/resource/transport/device/transport_device_ibverbs.cc)
- 它不直接调 `ibv_*`，而是经 [adapter_verbs.h](file:///e:/cann/hcomm/src/platform/inc/adapter/adapter_verbs.h) 中的 `hrtIbvPostSend` / `hrtIbvPollCq` 等运行时适配函数包装，底层仍是 `ibv_qp`、`ibv_cq`、`ibv_srq` 这套 verbs 对象

## 3. UCX — 有，最完整的 ibverbs 协议栈

[src/uct/ib](file:///e:/cann/ucx/src/uct/ib) 是 UCX 的 IB 传输层，覆盖面最广：
- **verbs 通用层**：[rc_verbs](file:///e:/cann/ucx/src/uct/ib/rc/verbs)、[ud_verbs](file:///e:/cann/ucx/src/uct/ib/ud/verbs)（RC/UD 可靠/不可靠连接）
- **Mellanox 加速层**：[mlx5](file:///e:/cann/ucx/src/uct/ib/mlx5)（含 dv/devx 直接 verbs，含 RC/DC/UD/GGA）
- **其他**：[rdmacm](file:///e:/cann/ucx/src/uct/ib/rdmacm)（连接管理）、[efa/srd](file:///e:/cann/ucx/src/uct/ib/efa)（AWS EFA）

**对比小结**：UCX 是通用的完整 verbs 协议栈；Mooncake 是面向 KV 传输场景的自研引擎、直接调 verbs；hcomm 是 HCCL 集合通信场景、通过 `hrtIbv*` 适配层间接使用 verbs。

---

**User**

他们的实现有什么区别

**Assistant**

我来深入看三个实现的核心架构，先并行读取关键头文件。

三个实现的核心信息已经掌握。这是一个对比类问题，我加载可视化模块来辅助呈现。

三者虽然都基于 ibverbs，但定位、抽象方式和运行机制差异很大。对核心代码的分析结论如下：

## 1. 定位与抽象层级

- **UCX**：通用通信框架，ibverbs 只是其 UCT 传输层的一个后端（与 tcp、shm、cuda_ipc 等并列）。IB 子系统自身再分三层：[base](file:///e:/cann/ucx/src/uct/ib/base)（md/iface/device 抽象）→ 协议实现（[rc_verbs](file:///e:/cann/ucx/src/uct/ib/rc/verbs)、[ud_verbs](file:///e:/cann/ucx/src/uct/ib/ud/verbs)）→ 硬件直驱（[mlx5 dv/devx](file:///e:/cann/ucx/src/uct/ib/mlx5)、[efa/srd](file:///e:/cann/ucx/src/uct/ib/efa)）。对上提供 AM、RMA、Atomic、Tag 等完整通信语义。
- **Mooncake**：面向 LLM 推理的 KV cache 专用传输引擎。直接以 [RdmaContext](file:///e:/cann/Mooncake/mooncake-transfer-engine/include/transport/rdma_transport/rdma_context.h)（每 NIC 一个：PD/CQ/MR 池）+ RdmaEndPoint（每对端一组 QP）组织 verbs 对象，对上只提供批量 `submitTransfer`（Batch 拆成 Slice）语义。
- **hcomm**：HCCL 集合通信库内部的一个 transport 类。[TransportIbverbs](file:///e:/cann/hcomm/src/platform/resource/transport/host/transport_ibverbs_pub.h) 继承 `TransportNet`，只服务于一条集合通信 Link，语义是 `TxAsync`/`RxAsync` + notify 同步。

## 2. verbs 的调用方式

- **UCX**：编译期直链 libibverbs，且双路径——标准 `ibv_post_send`（见 [rc_verbs_ep.c](file:///e:/cann/ucx/src/uct/ib/rc/verbs/rc_verbs_ep.c#L60)）与绕过 verbs 的 mlx5dv/devx 硬件直驱（直接填 WQE、敲 doorbell）。
- **Mooncake**：直接 `#include <infiniband/verbs.h>` 调原生 API；新组件 tent 还支持 [ibv_loader](file:///e:/cann/Mooncake/mooncake-transfer-engine/tent/include/tent/transport/rdma/ibv_loader.h) 运行时 dlopen，解耦部署环境。
- **hcomm**：完全不直接调 `ibv_*`，全部经 [adapter_verbs.h](file:///e:/cann/hcomm/src/platform/inc/adapter/adapter_verbs.h) 的 `hrtIbvPostSend`、`hrtIbvPollCq` 等 hrt 适配函数，便于 host/device 两侧切换到底层 hccp 驱动实现。

## 3. 连接与进度模型

- **UCX**：wireup 由 UCP 层负责（sockcm/[rdmacm](file:///e:/cann/ucx/src/uct/ib/rdmacm)），进度靠上层驱动的单线程 `uct_worker_progress` 轮询，非阻塞、无线程池。
- **Mooncake**：自建 handshake daemon（元数据走 etcd/redis/http 的 TransferMetadata），按拓扑选 NIC 建 QP；[WorkerPool](file:///e:/cann/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp) 多线程按 NIC 分片并发 `performPostSend` + `processCompletions`，支持多 rail 条带化传输和 rail 故障恢复。
- **hcomm**：QP 建链由 network manager 按 ranktable 统一编排（`InitQpConnect`），支持每连接多 QP 和多种 QP mode（普通/离线/单算子）；完成同步不靠自己 poll CQ，而是与 HCCL 的 notify/stream 机制绑定。

## 4. 硬件特色能力

- **UCX**：覆盖面最广——mlx5 dv/devx、ODP、Tag Matching、DC、GGA/GDAKI（GPU 直发）、AWS EFA。
- **Mooncake**：GPUDirect 的 dma-buf 优化（[DmabufExport](file:///e:/cann/Mooncake/mooncake-transfer-engine/include/transport/rdma_transport/rdma_context.h#L97-L100)：单一 fd 跨 NIC 共享，节省 GPU BAR1 窗口）、超 `max_mr_size` 大 buffer 自动切分 MR。
- **hcomm**：华为 NPU 体系独有——**RDMA Write with Reduce** 网内归约（int8~bf16，max/min/sum，见 [transport_ibverbs_pub.h](file:///e:/cann/hcomm/src/platform/resource/transport/host/transport_ibverbs_pub.h#L123-L138)），以及把 WQ/CQ 下沉到 device 侧（`AiDataPlaneWq` + SW_DB/HW_DB doorbell 直通），AICPU/算子核可直接发包。

下面的卡片概括了核心差异：

**一句话总结**：三者不是替代关系，而是处在不同层次——UCX 是面向 HPC 的通用 verbs 协议栈（最全、最底层，靠 mlx5dv/devx 榨性能）；Mooncake 是为 KV cache 搬运特化的传输引擎（多线程、多轨、批量切片，握手依赖外部元数据服务）；hcomm 是 HCCL 集合通信链路的一层薄封装（verbs 收敛进 hrt 适配层，价值在 NPU 生态特有能力：网内归约和 WQ 下沉到 device）。

如果想进一步深入某个方向，比如 hcomm 的网内归约 WQE 格式、Mooncake 的多 rail 调度、或 UCX 的 devx 直驱路径，我可以继续展开。

---

