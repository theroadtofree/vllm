"""边侧调度器:chunk 化 prefill 组批 + 控制面发布(notify/abort/seqno)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结。边侧只做 embedding(执行层无 decode),故需
专用调度器接管请求的边侧生命周期:

  入队 -> EMBEDDING(chunk 组批/范围预告)
       -> 嵌入完结:走原生 finish_requests 清出调度器(释放边侧 KV
          簿记、通知 worker 释放缓存),登记 awaiting
       -> AWAITING(等待云结果;前端未收到输出继续等待)
       -> 云结果终结(lwd_edge_deliver_tokens);迟到结果幂等丢弃。

纯 prefill 的实现依据:schedule() 全量复用原生——边侧请求从不产生
输出 token(num_tokens_with_spec 恒等于 num_prompt_tokens),且嵌入
完结当步即被清出调度器,原生 RUNNING 段每步只会调度剩余 prefill,
decode 分支不可达。

chunk 组批约束(lwd_chunked_prefill_design.md):prompt 按
c = min(剩余, chunk 单元 C) 切块,批 = 续传 chunk 优先 + waiting FCFS
前缀,Σc ≤ cap(云侧预算 − 在途数 × spec 系数,边云同值前提下读本地
配置);续传队首装不下时的 cap 截断块独占一批(云侧按批内最大条目
复现 chunk 边界,设计 §2.4)。一个批 = 一条批量 RangeNotify = 一个
seqno = 一次 UP send,数据面 chunk 流按批连续,跨请求交错/重组复杂度
仍为零。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdAbortNotify,
    LwdC2eNotify,
    LwdRangeItem,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)
from vllm.v1.lwd_debug import LwdControlLog
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
        LwdControlPublisher,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)

# add 预告发布重试:次数 x 递增间隔(共约 3s),耗尽即请求级报错
_LWD_ADD_RETRY_STEPS = 5
_LWD_ADD_RETRY_INTERVAL_S = 0.2

# 云侧 pending 上限,按未消费 chunk(seqno)计:一条 seqno = 一个云步
# 消费单位。边侧记录 seqno -> 批内请求列表,批内全部请求收到首 token
# 即该 chunk 已消费;未消费 chunk 数达到上限即停止新开 prefill——请求
# 在边侧 waiting 攒批(picker FCFS 前缀天然组多请求批),云侧 pending
# 排空后闸门自开。
_LWD_MAX_PENDING_CHUNKS = 3


class LwdEdgeScheduler(LwdBaseScheduler):
    """纯 prefill 调度语义 + 控制面出口(notify/abort/seqno)。"""

    def __init__(
        self,
        *args,
        publisher: LwdControlPublisher | None = None,
        **kwargs,
    ) -> None:
        """publisher 经构造注入(与调度器同生命周期)。"""
        super().__init__(*args, **kwargs)
        self.lwd_edge_publisher = publisher
        self._lwd_seqno = 0
        # 前缀缓存:manager 级关命中,配置级保留使能。两级拆分的原因:
        # - 必须关命中:命中会跳过 token 排程,首条 RangeNotify 的
        #   offset != 0,云侧按 offset==0 识别首块的约定失效;且被
        #   "命中"的块在 EMBED 批下从未写入 KV(边侧不落 KV),
        #   缓存登记残留会持续占用块池。关闭后命中恒空、free 直接
        #   归还块池。
        # - 不能在配置级关:request_block_hasher 只在配置级使能时
        #   创建,关掉则 Request.block_hashes 恒空,LwdRequestNotify
        #   带不出哈希链——云侧 prompt 是占位零值 token,前缀缓存
        #   只能靠边侧按真实内容算出的哈希链命中。
        self.kv_cache_manager.enable_caching = False
        # 引擎初始化对 0 KV 组的模型强制关 chunked prefill
        # (core.py:136-141,"model without KVCache")——边侧恰是 0 KV 组
        # (worker get_kv_cache_spec={}),该开关一关,原生 waiting 段对
        # 超预算 prompt 直接 break 不截断(scheduler.py:696-702),chunk
        # 切片无从谈起(逃生通道的"允许原生截断"同样依赖它)。原生关它
        # 的理由(encoder 模型无 KV 不能分块前向)对边侧不成立:边侧
        # forward 被 LWD 劫持,chunk 是纯调度记账,执行层只 embed 切片。
        # 此处重新打开;flag 在 schedule() 时才被读取,构造期生效。
        self.scheduler_config.enable_chunked_prefill = True
        # awaiting:嵌入完待云结果的 request_id -> 登记时刻(单调钟)。
        # 请求本体已清出调度器,此表是结果路径的唯一生命周期台账。
        self._lwd_awaiting: dict[str, float] = {}
        # 已收到首个 token 的请求集合:awaiting 中不在此集合的请求 =
        # 云侧尚未消费到其 prefill(pending_notify 的边侧本地估计,
        # 免新增协议字段);首个 c2e 到达即标记,完结/abort 时清除。
        self._lwd_first_token_seen: set[str] = set()
        # seqno -> 批内 request_id 列表(chunk 粒度 pending 跟踪):
        # 批内全部请求收到首 token 即整条移除;abort 时按请求摘除,
        # 摘空即移除。云侧 pending_notify 的边侧本地镜像,免协议改动。
        self._lwd_batch_requests: dict[int, list[str]] = {}
        # pending 闸门状态(仅作开/关转换日志用)
        self._lwd_prefill_gate_held: bool = False
        # MTP 预算系数:decode 每请求每步消耗 1+k 个 token(草稿 token
        # 同样占预算与 KV);k = num_speculative_tokens(配置解析期已
        # 求值,缺省 None 按 0),无 spec 配置即 1。
        # 边云 max_num_batched_tokens/max_num_seqs 部署对齐(同值),故
        # cap 直接读本地配置(设计 §2.3,配置对齐为部署约束)
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        num_spec = (
            getattr(spec_config, "num_speculative_tokens", None) or 0
        )
        self._lwd_spec_factor: int = 1 + num_spec
        # mixed 开关:与云侧调度器选择同源(scheduler_name,部署双侧同值)。
        # 非 mixed(prefill_first/decode_first 逃生通道)= 旧单请求组批语义:
        # 只取队首、允许原生截断成 chunk、不做整 prompt 核验
        from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
            LwdConfig,
        )

        self._lwd_mixed: bool = (
            LwdConfig.from_env_and_config(self.vllm_config).scheduler_name
            == "mixed"
        )
        # 多模态请求的整 prompt mrope positions([3,N] CPU 张量,准入时
        # 一次性算好,按 chunk 切片随 LwdEmbedBatch 下发 worker);嵌入
        # 完结/abort 时摘除。key 在 = 该请求各 chunk 的 has_mrope 恒真。
        self._lwd_mrope_positions_dict: dict[str, torch.Tensor] = {}
        # 惰性解析的模型类静态方法 _get_mrope_input_positions(直接缓存
        # 方法本身而非模型类,语义直达;首用即验,不支持的族 fail-fast)。
        self._lwd_mrope_positions_fn = None
        # chunk 单元 C:prompt 按 min(剩余, C) 切块下发
        # (lwd_chunked_prefill_design.md)。C 取本地
        # long_prefill_token_threshold(原生调度本就按它截断 prompt 排程,
        # scheduler.py:408/690,双侧天然复现同一边界);未配置(≤0,默认)
        # 时退化为 max_num_scheduled_tokens(= 边云对齐预算,整 prompt 不
        # 超预算不切块,与整体组批形态连续过渡)。云侧不读本值,按每条
        # RangeNotify 的 max(items) 临时覆写阈值复现(设计 §2.3)。
        threshold = self.scheduler_config.long_prefill_token_threshold
        self._lwd_chunk_unit: int = (
            threshold if threshold > 0 else self.max_num_scheduled_tokens
        )
        # 活性 fail-fast:最坏在途(满员)下 cap 恒 < 1 即死锁配置
        # (设计 §2.6),拉起时直接拒绝
        if self._lwd_mixed and (
            self.max_num_scheduled_tokens
            - self.max_num_running_reqs * self._lwd_spec_factor
            < 1
        ):
            raise RuntimeError(
                f"[LWD] deadlock config: max_num_batched_tokens("
                f"{self.max_num_scheduled_tokens}) - max_num_seqs("
                f"{self.max_num_running_reqs}) * spec_factor("
                f"{self._lwd_spec_factor}) < 1, no schedulable budget; "
                "请增大 max_num_batched_tokens 或减小 max_num_seqs"
            )

    def schedule(self) -> SchedulerOutput:
        """chunk 化 prefill 组批 + 原生准入。

        EMBED 批的 LwdBatch(seqno/token 片段)由 lwd_edge_notify 在
        发布成功后挂批——seqno 必须与发布成功绑定。

        开新准入:云侧在途满员(lwd_edge_max_num_seqs_check 为 False)
        时本步空排、新开请求留 waiting 等云侧排水;running 尚有未发完
        embed 的续传不受闸门约束(先收尾再开新,亦防上限=1 时自锁)。"""
        LwdControlLog.flight(len(self.running), len(self._lwd_awaiting))
        if (
            not self.lwd_edge_max_num_seqs_check()
            and not self._lwd_has_prefill_chunk_inflight()
        ):
            return SchedulerOutput.make_empty()
        # 云侧 pending 闸门(组批的产生机制):awaiting 中尚未收到首个
        # token 的请求数 = 云侧尚未消费到的 prefill(notify 积压的边侧
        # 本地估计,免协议改动)。超过上限即停止新开 prefill,请求在边
        # 侧 waiting 攒批(picker FCFS 前缀天然组多请求批),云侧
        # pending 排空后闸门自开。续传豁免(先收尾再开新)。
        if self._lwd_mixed and not self._lwd_has_prefill_chunk_inflight():
            pending_chunks = len(self._lwd_batch_requests)
            if pending_chunks >= _LWD_MAX_PENDING_CHUNKS:
                if not self._lwd_prefill_gate_held:
                    self._lwd_prefill_gate_held = True
                    logger.info(
                        "[Lwd][edge-sched] prefill gate HOLD: "
                        "pending_chunks=%d >= %d",
                        pending_chunks, _LWD_MAX_PENDING_CHUNKS,
                    )
                return SchedulerOutput.make_empty()
            if self._lwd_prefill_gate_held:
                self._lwd_prefill_gate_held = False
                logger.info(
                    "[Lwd][edge-sched] prefill gate OPEN: pending_chunks=%d",
                    pending_chunks,
                )

        return self._lwd_schedule_batch()

    def _lwd_pick_prefill_batch(
        self,
    ) -> tuple[list[str], dict[str, int], int]:
        """选下一步 embed 批:续传优先 + FCFS 前缀组批,按 chunk 单元切块。

        返回 (req_ids, expected, cap):expected 为逐请求的本步应排 token
        数(步后 chunk 核验用,不等即配置发散 fail-fast);cap 为本步
        token 预算上界(云侧预算 − 在途数 × spec 系数,边云同值前提下
        本地配置即云侧预算),供 _lwd_schedule_batch 做预算钳制。

        chunk 语义(lwd_chunked_prefill_design.md §3.1):
        - 每个 chunk 自限:c = min(剩余 prompt, chunk 单元 C),云侧原生
          调度在阈值复现(T=批内最大)下逐请求推出同一边界;
        - 续传(先收尾再开新):running 中未发完的 prefill 按 FCFS 序取
          c = min(剩余, C),Σ ≤ cap;
        - 开新:waiting(skipped 优先,原生准入失败回插者)按 FCFS
          迭代序取前缀,c = min(prompt, C),满足 Σchunk ≤ cap 且
          新增请求数 ≤ free_seq;不跳过队首(保 FCFS 与 seqno 链次序);
        - DeepSeek V4 总 prompt 长度 >=8000 的请求独占 prefill 批
          (含续传);短请求允许合批,不限制云侧 decode 混排;
        - 末位截断(续传/新开共用):批内最后一个条目允许被剩余额度
          截尾成 cap−used,与集中式 chunked prefill 的"预算耗尽截断"
          逐语义对齐——截断块恒为末位,边侧(§2.5)与云侧(§2.3)的
          预算钳制都靠预算耗尽精确复现它,非末位块由 R/C/T 定界与
          截断块尺寸无关。无此逃生,在途 decode 把 cap 压到 C 以下时
          大 prompt 永不可开新,流水线全程串行;
        priority 调度策略下 waiting 迭代序非弹出序,本前缀语义未适配
        (LWD 部署恒 FCFS,设计 M-10)。"""
        # 逃生通道(旧相位语义):单请求、不 cap、允许原生截断成 chunk、
        # 不做核验——两侧同配 prefill_first/decode_first 时逐字节
        # 复现旧行为
        if not self._lwd_mixed:
            candidates = list(self.skipped_waiting) + list(self.waiting)
            picked = [candidates[0].request_id] if candidates else []
            return picked, {}, self.max_num_scheduled_tokens
        # cap 语义:批 Σchunk ≤ 预算 − 在途数 × spec 系数(云混排步里
        # decode 每请求占 1+k token,MTP 下 k = num_spec_tokens;在途数
        # 是 decode 人口的恒成立上界,设计 §2.2)。边云预算部署对齐,
        # 本地 max_num_scheduled_tokens 即云侧预算
        inflight = len(self.running) + len(self._lwd_awaiting)
        cap = self.max_num_scheduled_tokens - inflight * self._lwd_spec_factor
        free_seq = self.max_num_running_reqs - inflight
        picked: list[str] = []
        expected: dict[str, int] = {}
        used = 0
        if cap < 1:
            return picked, expected, cap
        unit = self._lwd_chunk_unit
        # Apply the limit before scheduling/publishing so RangeNotify and the
        # embeddings payload retain identical request sets and token counts.
        is_deepseek_v4 = (
            self.vllm_config.model_config.hf_text_config.model_type == "deepseek_v4"
        )
        single_prefill = False
        # 1) 续传优先(先收尾再开新)
        for request in self.running:
            if single_prefill and picked:
                break
            remaining = (
                request.num_prompt_tokens - request.num_computed_tokens
            )
            if remaining <= 0:
                continue
            # Use the full prompt length, not this chunk's remaining length.
            # If a long request follows short ones, close the batch here and
            # leave it for the next batch; do not bypass it with waiting work.
            single_prefill = is_deepseek_v4 and request.num_prompt_tokens >= 8000
            if single_prefill and picked:
                break
            chunk = min(remaining, unit)
            if used + chunk > cap:
                # 末位截断:批内最后一个条目允许被剩余额度截尾,
                # 批收尾(预算耗尽,与集中式 chunked prefill 逐语义
                # 对齐)。复现依据:截断块恒为末位,边侧预算钳制
                # (§2.5)与云侧预算钳制(§2.3)都靠"预算耗尽"复现它;
                # 非末位块由 R/C/T 定界,与截断块尺寸无关
                c = cap - used
                if c >= 1:
                    picked.append(request.request_id)
                    expected[request.request_id] = c
                    used += c  # 截断块同样占预算,漏记会被新开循环超批
                break
            picked.append(request.request_id)
            expected[request.request_id] = chunk
            used += chunk
        # 2) 开新:FCFS 前缀,自限 chunk;装不下时同上末位截断。
        # 无截断逃生时,任一在途请求(含 decode 中的 awaiting)把 cap
        # 压到 C 以下会让大 prompt 永不可开新、流水线全程串行(真机
        # 已现:decode 阶段云侧只剩 1-2 个请求在跑)
        new_count = 0
        candidates = list(self.skipped_waiting) + list(self.waiting)
        for request in candidates:
            if single_prefill and picked:
                break
            single_prefill = is_deepseek_v4 and request.num_prompt_tokens >= 8000
            if single_prefill and picked:
                break
            chunk = min(request.num_prompt_tokens, unit)
            if new_count >= free_seq:
                break
            if used + chunk > cap:
                c = cap - used
                if c >= 1:
                    picked.append(request.request_id)
                    expected[request.request_id] = c
                    used += c
                break
            picked.append(request.request_id)
            expected[request.request_id] = chunk
            used += chunk
            new_count += 1
        # 组批自洽:截断分支同样记账,任何路径下 Σexpected == used ≤ cap
        # (真机实录:截断分支漏记 used,新开循环按陈旧余量超批 2×cap)
        total = sum(expected.values())
        assert total == used and total <= cap, (
            f"[Lwd][edge-sched] picker budget overflow: "
            f"Σexpected={total} used={used} cap={cap}"
        )
        return picked, expected, cap

    def _lwd_schedule_batch(self) -> SchedulerOutput:
        """chunk 组批:picker 选批,预算钳制后经基类可见集机制调度 +
        逐 chunk 核验。

        选择规则见 _lwd_pick_prefill_batch;队列剔除/隔离/拼回复用基类
        _lwd_schedule_for_visible_reqs(waiting/skipped 来源走原生准入
        窗口,running 来源走续跑)。

        预算钳制(设计 §2.5):可见集调用期间把 max_num_scheduled_tokens
        临时压到 cap(原生 token 预算即读该字段,
        scheduler.py:360),原生 budget/threshold 双重截断逐请求复现
        picker 的 chunk 划分。语义注记:与旧手写容器交换不同,本步被
        抢占的请求回 waiting 尾部、被跳过的回 skipped 队首,均取基类
        统一语义,不再做队首回插。"""
        req_ids, expected, cap = self._lwd_pick_prefill_batch()
        if req_ids:
            logger.info(
                "[Lwd][edge-sched] pick batch reqs=%s tokens=%d cap=%d",
                req_ids, sum(expected.values()), cap,
            )
        if req_ids and cap < self.max_num_scheduled_tokens:
            saved_budget = self.max_num_scheduled_tokens
            self.max_num_scheduled_tokens = cap
            try:
                out = self._lwd_schedule_for_visible_reqs(req_ids)
            finally:
                self.max_num_scheduled_tokens = saved_budget
        else:
            out = self._lwd_schedule_for_visible_reqs(req_ids)
        # chunk 核验:picker 的 chunk 划分必须被原生调度逐请求复现
        # (不等即 cap/threshold 失守或原生行为发散;云侧注入窗口
        # 看门狗也会把它变成 RuntimeError——这里先炸,错在边侧调度,
        # 不等到数据面)
        for req_id, want in expected.items():
            got = out.num_scheduled_tokens.get(req_id, 0)
            if got != want:
                raise RuntimeError(
                    f"[Lwd][edge-sched] chunk schedule mismatch: "
                    f"req={req_id} scheduled={got} != expected={want} "
                    f"(cap/threshold breach or native behavior divergence)"
                )
        return out

    def lwd_edge_max_num_seqs_check(self) -> bool:
        """max_num_seqs 适配检查:云侧在途水位(running + awaiting)是否
        还有名额,True=可开新请求。

        awaiting 请求已清出调度器,原生准入只数 running(边侧恒在飞批
        深度以内)永远拦不住;以 running+awaiting 对账云侧在途数,达到
        max_num_running_reqs(边云部署对齐,即云侧名额)即满员。续传
        豁免不在本判断(schedule 闸门经 _lwd_has_prefill_chunk_inflight
        放行收尾);请求到达时的 announce 亦不受约束(云只登记不计算)。"""
        return (
            len(self.running) + len(self._lwd_awaiting)
            < self.max_num_running_reqs
        )

    def _lwd_has_prefill_chunk_inflight(self) -> bool:
        """running 中是否存在未发完的 embed 请求(续传收尾中)。

        与 _lwd_pick_prefill_batch 的续传分支同判据,两处需保持一致:
        闸门放行收尾的前提是 picker 必然挑中该续传请求而非开新。"""
        return any(
            req.num_computed_tokens < req.num_prompt_tokens
            for req in self.running
        )

    def lwd_edge_add_request(self, request: Request) -> None:
        """请求入口:边界校验 -> 云预告 -> 本地入队。

        预告先于本地登记:云侧视图领先本地工作,云只能准备不能提前
        计算(没有 chunk 预告就不会开算)。abort_immediately 请求走
        finish + abort 出口,与原生语义一致。"""
        self._lwd_validate_request(request)
        self._lwd_compute_mrope_positions(request)
        self.lwd_edge_notify_request(
            request_id=request.request_id,
            num_prompt_tokens=len(request.prompt_token_ids),
            sampling_params=request.sampling_params,
            block_hashes=list(request.block_hashes),
        )
        super().add_request(request)
        if request.abort_immediately:
            self.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
            self.lwd_edge_abort([request.request_id])

    @staticmethod
    def _lwd_request_has_mm(request: Request) -> bool:
        """请求是否带真实多模态数据(prompt_embeds passthrough 不算——
        它没有 grid 元数据,mrope 按文本位置处理,与原生口径一致)。"""
        return any(
            feature.modality != "prompt_embeds"
            for feature in request.mm_features
        )

    def _lwd_compute_mrope_positions(self, request: Request) -> None:
        """准入时为多模态请求一次性算全 prompt 的 [3,N] mrope positions。

        设计分工(对照 latest_lwd 边云):token_id 不上线路,云侧拿
        不到 grid 元数据,无法从第一性原理构造 3D 位置——由边侧(唯一
        持有真实 prompt + mm_features 的一侧)算好,逐 chunk 切片随
        EMBED 批经数据面发给云;delta 云侧收齐末 chunk 后自推
        (max+1-N),不上 wire。

        模型类静态方法直接以 hf_config 调用(Qwen3VL 族,含 Qwen3.5),
        无需模型实例/权重;模型族不提供该静态入口即拒绝(报错信息给出
        白名单口径),不静默按文本位置错算。"""
        if not self.vllm_config.model_config.uses_mrope:
            return
        if not self._lwd_request_has_mm(request):
            return
        if self._lwd_mrope_positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_cls, _ = ModelRegistry.resolve_model_cls(
                self.vllm_config.model_config.architectures,
                self.vllm_config.model_config,
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions (Qwen3VL family, incl. "
                    f"Qwen3.5); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._lwd_mrope_positions_fn = fn
        mrope_features = [
            f for f in request.mm_features if f.modality != "prompt_embeds"
        ]
        _t_mrope = time.monotonic()
        positions, _delta = self._lwd_mrope_positions_fn(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mrope_features,
            config=self.vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._lwd_mrope_positions_dict[request.request_id] = positions
        # [Lwd][perf] mrope 全 prompt 计算逐请求耗时(在边引擎主线程上,
        # 期间调度循环停转;MM 劣化归因)
        logger.info(
            "[Lwd][perf] mrope-compute req=%s prompt=%d dur=%.2fms",
            request.request_id, request.num_prompt_tokens,
            (time.monotonic() - _t_mrope) * 1000,
        )

    def lwd_edge_notify(self, scheduler_output: SchedulerOutput) -> bool:
        """对新调度的 prefill 批发一条批量 LwdRangeNotify(seqno 先行)。

        seqno 无空洞契约:UP 数据通道按连续号序配对(通道层对超前号
        扣留等待,一个空洞即永久挂死整条链),因此号只能分配给真正
        上 wire 的批:
        - peek-then-advance:发布成功才进位计数器;
        - 单 notify 前提:本方法每步至多发一条(批量条目全部装在同
          一条里),发布原子性即整批原子性,不存在部分成功。

        前提:控制面发布通道不丢消息,publish 恒成功——步末回退
        对账已按此前提移除。若前提被破坏返回 False,调用方本步不
        派发但进度不回退,该批永久丢失;重复预告在云侧按
        (request_id, offset) 幂等登记。"""
        publisher = self.lwd_edge_publisher
        scheduled = scheduler_output.num_scheduled_tokens
        items: list[LwdRangeItem] = []
        req_ids: list[str] = []
        token_slices: list[list[int]] = []
        prompt_offsets: list[int] = []
        has_mrope: list[bool] = []
        mrope_slices: list[list[list[int]]] = []
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            # _update_after_schedule 已乐观推进 num_computed,起点需回退本步量
            offset = request.num_computed_tokens - num_tokens
            # 多模态:准入时预算了整 prompt positions 的请求,各 chunk
            # 恒携带 mrope 行(key 在即恒真);[3,N] 切本 chunk 列 ->
            # [n,3] 行序与 token 轴对齐,随批下发 worker 组 aux 帧
            positions = self._lwd_mrope_positions_dict.get(request_id)
            req_has_mrope = positions is not None
            items.append(
                LwdRangeItem(
                    request_id=request_id,
                    offset=offset,
                    num_tokens=num_tokens,
                    has_mrope=req_has_mrope,
                )
            )
            req_ids.append(request_id)
            token_slices.append(
                list(request.prompt_token_ids[offset : offset + num_tokens])
            )
            prompt_offsets.append(offset)
            has_mrope.append(req_has_mrope)
            if req_has_mrope:
                assert positions is not None
                mrope_slices.append(
                    positions[:, offset : offset + num_tokens].t().tolist()
                )
            else:
                mrope_slices.append([])
        if not items:
            return True
        seqno = self._lwd_seqno
        first = items[0]
        # 顶层单请求字段 = 首条目:旧版云侧(不读 items)仍可消费单请求批
        if not publisher.publish(
            LwdRangeNotify(
                request_id=first.request_id,
                offset=first.offset,
                num_tokens=first.num_tokens,
                seqno=seqno,
                items=items,
            )
        ):
            return False
        self._lwd_seqno = seqno + 1
        self._lwd_batch_requests[seqno] = list(req_ids)
        logger.info(
            "[Lwd][edge-notify] batch reqs=%d tokens=%d seqno=%d mrope=%d",
            len(items), sum(item.num_tokens for item in items), seqno,
            sum(has_mrope),
        )
        # 发布成功即组 EMBED 批挂 SO:seqno 是数据面发云张量的
        # 配对键(与云侧 RangeNotify 登记同值),embed 载荷为本
        # 批逐请求的 token 片段(边 worker 按 req_ids 序扁平化);
        # mrope 行同序携带(仅 MM 请求非空),worker 据此组 aux 帧
        scheduler_output.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=req_ids,
                token_ids=token_slices,
                prompt_offsets=prompt_offsets,
                has_mrope=has_mrope,
                mrope_positions=mrope_slices,
            ),
        )
        return True

    def lwd_edge_notify_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        sampling_params: SamplingParams | None = None,
        block_hashes: list[bytes] | None = None,
    ) -> None:
        """发 LwdRequestNotify(请求元数据预告)。

        block_hashes = prompt 全量满块哈希链(自位置 0 起)。云侧
        prompt token 是占位零值,本地算不出真实内容哈希,前缀缓存
        命中只能靠这条链;缺省空链 = 不提供,云侧回退占位链。

        sampling_params 只透传影响云侧 token 选择的字段(采样核/惩罚/
        EOS 策略/min_tokens);stop 字符串等 detokenizer 层参数留在
        边侧前端原生处理,不上 wire。

        失败语义 fail-fast:发布队满时短退避重试(瞬态背压几乎必在
        秒级窗口内腾出),耗尽即抛 RuntimeError——异常沿 add_request
        调用链回前端 error 通道,用户立即得到失败;不做无限阻塞
        重试(发布点在引擎主线程,云宕机会把整个引擎卡死在 add)。
        此时请求未入队、云侧零残留,无需补发 abort。"""
        publisher = self.lwd_edge_publisher
        if publisher is None:
            return
        sp = sampling_params
        message = LwdRequestNotify(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=(
                sp.max_tokens if sp is not None and sp.max_tokens is not None else 16
            ),
            block_hashes=block_hashes if block_hashes is not None else [],
            temperature=sp.temperature if sp is not None else 1.0,
            top_p=sp.top_p if sp is not None else 1.0,
            top_k=sp.top_k if sp is not None else 0,
            min_p=sp.min_p if sp is not None else 0.0,
            seed=sp.seed if sp is not None else None,
            repetition_penalty=sp.repetition_penalty if sp is not None else 1.0,
            presence_penalty=sp.presence_penalty if sp is not None else 0.0,
            frequency_penalty=sp.frequency_penalty if sp is not None else 0.0,
            ignore_eos=sp.ignore_eos if sp is not None else False,
            stop_token_ids=(
                list(sp.stop_token_ids) if sp is not None and sp.stop_token_ids else []
            ),
            min_tokens=sp.min_tokens if sp is not None else 0,
            eos_token_id=sp.eos_token_id if sp is not None else None,
        )
        for attempt in range(_LWD_ADD_RETRY_STEPS):
            if publisher.publish(message):
                logger.info(
                    "[Lwd][edge-notify] request meta announced: req=%s "
                    "prompt=%d",
                    request_id, num_prompt_tokens,
                )
                return
            time.sleep(_LWD_ADD_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] add-request notify for {request_id} dropped: publish "
            f"queue full after {_LWD_ADD_RETRY_STEPS} retries "
            f"(cloud PRE_OUT consumption stalled?)"
        )

    def lwd_edge_abort(self, request_ids: list[str]) -> None:
        """发 LwdAbortNotify + 摘除 awaiting;调度器内清理走原生路径。

        awaiting 请求已不在调度器视野(嵌入完结时清出),原生
        finish_requests 触不到它,须在此显式摘除,防迟到云结果被误认领。"""
        publisher = self.lwd_edge_publisher
        for request_id in request_ids:
            self._lwd_awaiting.pop(request_id, None)
            self._lwd_first_token_seen.discard(request_id)
            self._lwd_mrope_positions_dict.pop(request_id, None)
            seen = self._lwd_first_token_seen
            for seqno, rids in list(self._lwd_batch_requests.items()):
                rids = [rid for rid in rids if rid != request_id]
                if not rids or all(rid in seen for rid in rids):
                    del self._lwd_batch_requests[seqno]
                else:
                    self._lwd_batch_requests[seqno] = rids
            if publisher is None:
                continue
            if not publisher.publish(LwdAbortNotify(request_id=request_id)):
                logger.warning(
                    "[Lwd] drop abort signal for %s: publish queue full", request_id
                )
            else:
                logger.info(
                    "[Lwd][edge-notify] AbortNotify req=%s", request_id
                )

    def lwd_edge_update_progress(self, executed: dict[str, int]) -> None:
        """步末登记执行量;嵌入完结即转入 awaiting。

        notify 恒成功前提:排程即派发即执行(同步执行),executed 与
        本步排程集恒等,原生 _update_after_schedule 的乐观推进即真实
        水位,无需回退对账。前提被破坏(发布队满未派发)时进度将
        静默虚高、该 chunk 永久丢失,由发布通道不丢消息保证不发生。

        嵌入完结 = 边侧工作结束而非请求结束:走原生 finish_requests
        做全套簿记(移出 running/requests、释放边侧 KV、进
        finished_req_ids 通知 worker 释放缓存——该通知随下一个排程步
        下发,引擎纯睡眠期滞后),同时登记 awaiting;前端未收到任何
        输出会继续等待,终结由云结果驱动(lwd_edge_deliver_tokens)。"""
        finished_ids: list[str] = []
        for request_id in executed:
            request = self.requests.get(request_id)
            if request is None:
                # 本步内已终结(abort/更早完成):迟到的登记无对象
                continue
            if request.num_computed_tokens >= request.num_prompt_tokens:
                finished_ids.append(request_id)
        if finished_ids:
            self.finish_requests(finished_ids, RequestStatus.FINISHED_STOPPED)
            now = time.monotonic()
            for request_id in finished_ids:
                self._lwd_awaiting[request_id] = now
                self._lwd_mrope_positions_dict.pop(request_id, None)
                logger.info(
                    "[Lwd][edge-progress] req=%s embed done -> awaiting",
                    request_id,
                )

    def lwd_edge_deliver_tokens(
        self, request_id: str, token_ids: list[int], finished: bool
    ) -> bool:
        """云结果投递(awaiting 消费点,引擎步内调用)。

        token_ids 由边侧 worker unembedding 产生,本层不消费内容,
        仅做生命周期对账:
        - 请求在 awaiting:认领;finished=True 时出 awaiting(请求本体
          已在嵌入完结时清出调度器,无需再 finish);
        - 请求不在(已 abort/更早完结/未知):迟到结果,返回 False,
          调用方丢弃告警(幂等,不复活)。

        token_ids/finished 的输出组包归引擎层(EngineCoreOutputs)。"""
        if request_id not in self._lwd_awaiting:
            logger.warning(
                "[Lwd][edge-deliver] stale result for req=%s (not awaiting)",
                request_id,
            )
            return False
        self._lwd_first_token_seen.add(request_id)
        seen = self._lwd_first_token_seen
        for seqno, rids in list(self._lwd_batch_requests.items()):
            if all(rid in seen for rid in rids):
                del self._lwd_batch_requests[seqno]
        if finished:
            del self._lwd_awaiting[request_id]
            self._lwd_first_token_seen.discard(request_id)
        logger.info(
            "[Lwd][edge-deliver] req=%s tokens=%d finished=%s",
            request_id, len(token_ids), finished,
        )
        return True

    def _lwd_validate_request(self, request: Request) -> None:
        """模式边界校验;违规抛 ValueError,在入队之前拒绝,错误经
        add_request 调用链回到客户端 error 路径。

        边界 = 边侧能力面:边是 embedding 属主(拒绝客户端自带
        prompt_embeds),只处理纯文本补全(拒 pooling/结构化
        输出),prompt 非空;不上 wire 的采样参数(logit_bias/
        allowed_token_ids/logprobs)缺省即拒,不静默丢约束。

        多模态白名单:仅 image;video/audio 等其余模态与 EVS
        (multimodal pruning)拒绝——merge/mrope 链路按 image 语义
        实现,其余模态未适配,静默放过即错算。"""
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        if request.mm_features:
            unsupported_mm = sorted(
                {
                    feature.modality
                    for feature in request.mm_features
                    if feature.modality not in ("image", "prompt_embeds")
                }
            )
            if unsupported_mm:
                raise ValueError(
                    f"[LWD] prefill-only mode supports image-only "
                    f"multimodal requests; got modality "
                    f"{', '.join(unsupported_mm)} "
                    f"(request {request.request_id})"
                )
            if (
                self.vllm_config.model_config.multimodal_config
                .is_multimodal_pruning_enabled()
            ):
                raise ValueError(
                    "[LWD] prefill-only mode does not support multimodal "
                    f"pruning (EVS) (request {request.request_id})"
                )
        if request.pooling_params is not None:
            raise ValueError(
                "[LWD] prefill-only mode does not support pooling requests "
                f"(request {request.request_id})"
            )
        if request.use_structured_output:
            raise ValueError(
                "[LWD] prefill-only mode does not support structured output "
                f"(request {request.request_id})"
            )
        sp = request.sampling_params
        unsupported = [
            name
            for name, value in (
                ("logit_bias", sp.logit_bias),
                ("allowed_token_ids", sp.allowed_token_ids),
                ("logprobs", sp.logprobs),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "[LWD] prefill-only mode does not support sampling "
                f"param(s) {', '.join(unsupported)} "
                f"(request {request.request_id}): not carried on the "
                "edge->cloud wire, cloud would silently sample without them"
            )

def lwd_build_unembed_batch(notify: LwdC2eNotify) -> SchedulerOutput:
    """组 UNEMBED 批(引擎步内调用,云载荷派发给边 worker 做 lm_head)。

    云结果不经过原生 schedule,无原生排程产物可用——以 make_empty
    为骨架:
    - lwd_batch 携带 LwdUnembedBatch:req_ids(隐藏行序)/
      num_accept_tokens/top_id_ths 逐请求透传自 c2e;批序号将随 c2e
      通告携带(规划),当前占位 0;recv_num_elements
      (DOWN 通道每请求接收元素数)与 out_token_idxs(生成序号)控制面
      不可知,留空由数据面按 DOWN 张量实收推导;
    - 请求集合同步镜像到 num_scheduled_tokens:值 = 该请求本步
      hidden 行数(num_accepted_tokens 对位,spec 步可 >1),保持
      原生管道字段语义一致,不作 token 预算解释。
    """
    scheduler_output = SchedulerOutput.make_empty()
    # 每请求本步还原的 token 数 = hidden 行数 = num_accepted_tokens
    # (spec 步一请求可多行,非 spec 恒 1);与 req_ids 按位对齐,错配
    # fail-fast
    assert len(notify.num_accepted_tokens) == len(notify.req_ids)
    scheduler_output.num_scheduled_tokens = dict(
        zip(notify.req_ids, notify.num_accepted_tokens)
    )
    scheduler_output.total_num_scheduled_tokens = sum(
        notify.num_accepted_tokens
    )
    scheduler_output.lwd_batch = LwdBatch(
        batch_type=LwdBatchType.LWD_UNEMBED,
        seqno=notify.down_seqno,
        batch_meta=LwdUnembedBatch(
            req_ids=list(notify.req_ids),
            num_accept_tokens=list(notify.num_accepted_tokens),
            recv_num_elements=notify.hidden_num_elements,
            out_token_idxs=[],
            top_id_ths=list(notify.top_id_ths),
            token_ids=[list(t) for t in notify.token_ids],
        ),
    )
    scheduler_output.lwd_c2e_notify = [notify]
    return scheduler_output
