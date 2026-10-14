"""云侧混合调度器:prefill/decode 同批下发(设计:lwd_mixed_batch_design.md
+ lwd_chunked_prefill_design.md)。

chunk 混排语义:边侧按 c = min(剩余 prompt, chunk 单元) 切块组批
(续传优先 + FCFS 前缀),云侧每步至多消费 prefill_notify_queue 队首
一条批量 RangeNotify,与全部 decode 态请求混排一个批下发:

  - preflight 三检(预算/名额/KV 保守预估)任一不过 -> 本步 decode-only,
    预告留队首原位,decode 泄压后下步重试;
  - 放行先做 offset 对账(item.offset == 云侧 num_computed,chunk
    断流/重复/乱序即 fail-fast),再按 T = max(items) 临时覆写
    long_prefill_token_threshold——原生调度的 threshold 截断逐请求
    复现边侧 chunk 边界(自限 chunk,设计 §2.3 证明与处理顺序无关);
  - 步后核验批内每请求 num_scheduled == 预告量(chunk 精确准入),
    不等即 fail-fast(一次 UP recv 对应整批张量,错排会留残余行错配)。

前置约束:spec decode(MTP)预算已按 1+k 系数对齐(边 cap 与本类
preflight),但端到端未经真机验证(eagle 纯度判据失真风险,台账
P3-3/设计 M-8);
不兼容云侧抢占重算(占位 embeds 无法本地重 prefill)与 prefix caching
(命中使 num_computed 起点 >0,触发 offset 对账 fail-fast;且被命中
chunk 的 UP send 无人消费 = 通道打洞,chunk 化后硬前提,设计 §8-C3)
——部署须保证 KV 充足并关闭缓存,本类的核验只负责把违背变成当场
报错而非静默错算。
"""

from __future__ import annotations

import time
from collections import deque

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdRangeItem,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)

logger = init_logger(__name__)


class LwdCloudMixedScheduler(LwdBaseScheduler):
    """notify 门控的 prefill/decode 混排调度器(相位机的继任者)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # prefill 通知队列:边侧批量范围预告(RangeNotify)逐条入队,每步
        # 至多消费队首一条;预告自带 seqno 即本批 UP 链配对号
        self.prefill_notify_queue: deque[LwdRangeNotify] = deque()
        # [Lwd][sched] 调度批日志步计数(饿死分析:RangeNotify 到达 ->
        # 混排步消费的间隔与中间插入的 decode-only 步数)
        self._lwd_sched_step = 0
        # MTP 预算系数:decode 每请求每步消耗 1+k 个 token(草稿 token
        # 占预算与 KV);k = num_speculative_tokens(配置解析期已求值,
        # 缺省 None 按 0),无 spec 配置即 1
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        num_spec = (
            getattr(spec_config, "num_speculative_tokens", None) or 0
        )
        self._lwd_spec_factor: int = 1 + num_spec
        logger.info(
            "[Lwd] cloud mixed scheduler: chunked-prefill batches mixed "
            "with decode (one notify consumed per step at most, "
            "spec_factor=%d)",
            self._lwd_spec_factor,
        )
        # spec 接受率累计器(引擎进程内直出):LWD 云前端没有注册请求,
        # 前端 stats 链(output_processor 丢弃未知请求输出 + lifespan
        # 周期任务)在云侧不可靠;接受率数据的源头本来就在调度器里,
        # 此处按原生口径(scheduler.py:1414-1424)累计并周期直出,
        # 与集中式的 Prometheus 计数器/前端日志同公式可直接对比
        self._lwd_spec_drafts = 0
        self._lwd_spec_draft_tokens = 0
        self._lwd_spec_accepted = 0
        self._lwd_spec_last_log = time.monotonic()

    def update_from_output(self, scheduler_output, model_runner_output):
        """先累计 spec 接受率(镜像原生 update_from_output 的统计分支),
        再走原生输出处理。"""
        self._lwd_observe_spec_acceptance(scheduler_output, model_runner_output)
        return super().update_from_output(scheduler_output, model_runner_output)

    _LWD_SPEC_LOG_INTERVAL_S = 10.0

    def _lwd_observe_spec_acceptance(
        self, scheduler_output, model_runner_output
    ) -> None:
        """逐请求累计草稿/接受数并周期打运行总计。

        口径与原生完全一致:num_draft_tokens = len(scheduled_spec_token_ids),
        num_accepted = len(generated_token_ids) - 1(含 bonus,减一剔除);
        跳过已终结/未知请求(与原生 :1403-1412 同款过滤)。"""
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not spec_tokens:
            return
        sampled = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index
        for req_id, spec_ids in spec_tokens.items():
            if not spec_ids:
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue
            generated = sampled[req_index] if sampled else []
            if not generated:
                continue
            self._lwd_spec_draft_tokens += len(spec_ids)
            self._lwd_spec_accepted += len(generated) - 1
            self._lwd_spec_drafts += 1
        now = time.monotonic()
        if now - self._lwd_spec_last_log < self._LWD_SPEC_LOG_INTERVAL_S:
            return
        self._lwd_spec_last_log = now
        if not self._lwd_spec_draft_tokens:
            return
        rate = self._lwd_spec_accepted / self._lwd_spec_draft_tokens * 100
        mean_accept_len = 1 + self._lwd_spec_accepted / max(
            self._lwd_spec_drafts, 1
        )
        logger.info(
            "[Lwd][spec] cumulative: rate=%.1f%% mean_accept_len=%.2f "
            "accepted=%d draft_tokens=%d drafts=%d",
            rate, mean_accept_len, self._lwd_spec_accepted,
            self._lwd_spec_draft_tokens, self._lwd_spec_drafts,
        )

    @staticmethod
    def _lwd_is_decode(request) -> bool:
        """prompt 已算完 = decode 态(可采样)。chunk 化后 running 可含
        prefill 半途请求(num_computed 爬坡中)——它们不进 decode 集,
        留在隐藏集等自己的下一条 chunk 预告驱动续排。"""
        return request.num_computed_tokens >= request.num_prompt_tokens

    def _lwd_collect_decode_requests(self) -> list[str]:
        """收集所有 decode 态(prompt 已算完)请求的 req_id。

        按 running/waiting/skipped 顺序遍历三队列,输出可直接作为
        _lwd_schedule_for_visible_reqs 的入参。"""
        return [
            req.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for req in queue
            if self._lwd_is_decode(req)
        ]

    # ------------------------------------------------------------------ #
    # notify 条目分类与 preflight                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lwd_notify_items(notify: LwdRangeNotify) -> list[LwdRangeItem]:
        """取批条目:优先 items(mixed 组批);空则回退旧版单请求语义。"""
        if notify.items:
            return list(notify.items)
        return [
            LwdRangeItem(
                request_id=notify.request_id,
                offset=notify.offset,
                num_tokens=notify.num_tokens,
            )
        ]

    def _lwd_classify_items(
        self, items: list[LwdRangeItem]
    ) -> tuple[list[LwdRangeItem], bool]:
        """条目三分:ready(待 prefill)/ stale(已完结或重复预告,剔除)
        / unknown(ADD 尚未落地)。返回 (ready, 是否有 unknown)。

        unknown 存在即整批 defer(all-or-nothing:一次 UP recv 对应整批
        张量,不能只放行子集);abort 清扫(引擎侧)负责把永不落地的
        条目摘除,队首不会被永久堵住。"""
        ready: list[LwdRangeItem] = []
        unknown = False
        for item in items:
            request = self.requests.get(item.request_id)
            if request is None:
                unknown = True
                continue
            if self._lwd_is_decode(request):
                logger.warning(
                    "[Lwd][cloud-sched] drop stale notify item req=%s "
                    "num=%d (already prefilled; duplicate notify?)",
                    item.request_id, item.num_tokens,
                )
                continue
            ready.append(item)
        return ready, unknown

    def _lwd_preflight_ok(
        self, items: list[LwdRangeItem], num_decode: int
    ) -> bool:
        """放行三检:token 预算 / 并发名额 / KV 保守预估。

        预算与名额本由边侧 cap 保证(设计 §2),此处是双保险;KV 预估
        忽略前缀缓存命中(命中只降占用,高估是安全方向)。decode 占用
        按 1+k 计(MTP 草稿 token 同样吃预算;KV 块预留同系数上取整)。"""
        need_tokens = (
            sum(item.num_tokens for item in items)
            + num_decode * self._lwd_spec_factor
        )
        if need_tokens > self.max_num_scheduled_tokens:
            logger.info(
                "[Lwd][cloud-sched] preflight hold: tokens %d > budget %d",
                need_tokens, self.max_num_scheduled_tokens,
            )
            return False
        # 名额只数新增准入:续传 chunk 的请求已在 running,再按条目数
        # 计会双重占用名额造成假性 hold(chunk 化前批内恒为新增,
        # len(items) 与新增数相等)
        running_ids = {req.request_id for req in self.running}
        new_items = sum(
            1 for item in items if item.request_id not in running_ids
        )
        if len(self.running) + new_items > self.max_num_running_reqs:
            logger.info(
                "[Lwd][cloud-sched] preflight hold: seqs %d+%d > max %d",
                len(self.running), new_items, self.max_num_running_reqs,
            )
            return False
        need_blocks = (
            sum(cdiv(item.num_tokens, self.block_size) for item in items)
            + num_decode * cdiv(self._lwd_spec_factor, self.block_size)
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if need_blocks > free_blocks:
            logger.info(
                "[Lwd][cloud-sched] preflight hold: kv blocks %d > free %d",
                need_blocks, free_blocks,
            )
            return False
        return True

    @staticmethod
    def _lwd_verify_full_admission(
        out: SchedulerOutput, notify: LwdRangeNotify, items: list[LwdRangeItem]
    ) -> None:
        """步后核验:批内每请求排程量 == 预告量(chunk 精确准入)。
        不等即配置发散/KV 失守(错排会让整批 UP 张量留残余行),
        fail-fast 不当场炸就会静默错算。"""
        for item in items:
            scheduled = out.num_scheduled_tokens.get(item.request_id, 0)
            if scheduled != item.num_tokens:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] batch admission mismatch "
                    f"req={item.request_id} seqno={notify.seqno}: "
                    f"scheduled={scheduled} != notify={item.num_tokens} "
                    f"(mixed mode forbids partial admission)"
                )

    # ------------------------------------------------------------------ #
    # 调度主体                                                            #
    # ------------------------------------------------------------------ #
    def _schedule_impl(self) -> SchedulerOutput:
        decode_ids = self._lwd_collect_decode_requests()
        notify = (
            self.prefill_notify_queue[0] if self.prefill_notify_queue else None
        )
        if notify is not None:
            items, unknown = self._lwd_classify_items(
                self._lwd_notify_items(notify)
            )
            if not items and not unknown:
                # 全部条目失效(完结/重复):弹队,本步纯 decode。
                # 按对象身份弹(abort 清扫可能整体替换过 deque,见
                # _schedule_mixed 同款处理)
                queue = self.prefill_notify_queue
                if queue and queue[0] is notify:
                    queue.popleft()
                notify = None
            elif unknown:
                # 请求 ADD 未落地(input_queue 次序滞后):预告留队首,
                # 本步先 decode;abort 清扫兜底防永久堵队首
                logger.info(
                    "[Lwd][cloud-sched] notify seqno=%d deferred: "
                    "request(s) not added yet",
                    notify.seqno,
                )
                notify = None
            elif not self._lwd_preflight_ok(items, len(decode_ids)):
                # 预算/名额/KV 不足:预告留队首,decode 泄压后重试
                notify = None
            else:
                return self._schedule_mixed(notify, items, decode_ids)
        return self._lwd_schedule_for_visible_reqs(decode_ids)

    def _schedule_mixed(
        self,
        notify: LwdRangeNotify,
        items: list[LwdRangeItem],
        decode_ids: list[str],
    ) -> SchedulerOutput:
        """混排步:decode 集 + 批内 chunk 同批下发。

        offset 对账:item.offset 必须 == 云侧当前 num_computed(async
        关闭前提下调度时刻即真实水位)——不等即 chunk 断流(>)/
        重复(<)/乱序,协议级事故 fail-fast,不当场炸就会把
        embeddings 注入错位窗口静默错算。
        双钳制复现(设计 §2.3/§2.4 修订):① 阈值复现——临时覆写
        long_prefill_token_threshold = max(items.num_tokens),非末位
        自限块由 R/C/T 定界逐请求复现;② 预算钳制——临时把
        max_num_scheduled_tokens 压到 decode 精确消耗 + Σ chunk,
        末位 cap 截断块由"预算耗尽"复现(与集中式 chunked prefill
        的截断语义逐字对齐)。两个 finally 均恢复。
        注意 decode 请求 num_new = 1+k 远小于正常量级的 T,不受覆写
        影响(病态小 chunk 配置见设计 §8-C9)。"""
        logger.info(
            "[Lwd][cloud-sched] mixed notify seqno=%d reqs=%s tokens=%d",
            notify.seqno, [item.request_id for item in items],
            sum(item.num_tokens for item in items),
        )
        for item in items:
            request = self.requests[item.request_id]
            if request.num_computed_tokens != item.offset:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] chunk offset mismatch "
                    f"req={item.request_id} seqno={notify.seqno}: "
                    f"notify offset={item.offset} != cloud computed="
                    f"{request.num_computed_tokens} "
                    f"(chunk gap/dup/reorder)"
                )
        req_ids = decode_ids + [item.request_id for item in items]
        # decode 精确消耗:预算钳制的精确性依赖逐字复刻原生 running 段
        # (scheduler.py:381-416)的 num_new 推导——1+len(spec) 不是准确式:
        # ① placeholder(spec 拒绝遗留的非零值)计入排程窗口;
        # ② max_tokens 守卫(占位>0 且距 max_tokens 不足 2)与
        #   next_decode_eligible_step 会让请求整步跳过、消耗 0。
        # 不逐字复刻就会留下余量,使末位截断块被超排(真机实录:
        # decode 请求被守卫跳过消耗 0,末位块 604 被排成 608)。
        # threshold 不约束 decode(正常量级 T ≫ num_new,病态小 chunk
        # 配置见设计 §8-C9);预算由构造保证充足,不影响本推导。
        decode_exact = 0
        for rid in decode_ids:
            req = self.requests[rid]
            placeholders = req.num_output_placeholders
            if (
                placeholders > 0
                and req.num_computed_tokens + 2 - placeholders
                >= req.num_prompt_tokens + req.max_tokens
            ):
                continue  # 原生 max_tokens 守卫:本步跳过,消耗 0
            if self.current_step < req.next_decode_eligible_step:
                continue  # V2+PP 节拍守卫:本步跳过,消耗 0
            decode_exact += max(
                0,
                min(
                    req.num_tokens_with_spec
                    + placeholders
                    - req.num_computed_tokens,
                    self.max_model_len - 1 - req.num_computed_tokens,
                ),
            )
        saved_threshold = self.scheduler_config.long_prefill_token_threshold
        saved_budget = self.max_num_scheduled_tokens
        self.scheduler_config.long_prefill_token_threshold = max(
            item.num_tokens for item in items
        )
        self.max_num_scheduled_tokens = decode_exact + sum(
            item.num_tokens for item in items
        )
        try:
            out = self._lwd_schedule_for_visible_reqs(req_ids)
        finally:
            self.scheduler_config.long_prefill_token_threshold = (
                saved_threshold
            )
            self.max_num_scheduled_tokens = saved_budget
        self._lwd_verify_full_admission(out, notify, items)
        # 消费队首:abort 清扫(IO 线程)可能整体替换过 deque,按对象
        # 身份弹队首,避免错弹替换后的新队首
        queue = self.prefill_notify_queue
        if queue and queue[0] is notify:
            queue.popleft()
        # UP 链 seqno 随批下发云 worker:批配对号直接取预告自带 seqno(与
        # 边侧 EMBED 批派发号同源同值)。batch_meta 承载 worker 的 recv
        # 尺寸与注入切行信息:req_ids 取批内逐请求,token_ids 为占位
        # 列表——长度必须等于边侧实际发送的逐请求 token 数,recv numel
        # 才能与边侧 isend 严格相等(HCCL P2P 要求两端 numel 匹配)。
        # prompt_offsets/has_mrope 逐 item 透传:worker 据此求 aux 帧
        # 尺寸(Σ 有 mrope 条目的 n_i×3),runner 据此对 MM 请求注入
        # mrope 行(文本请求零 aux 流量、零路径变化)。
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[item.request_id for item in items],
                token_ids=[[0] * item.num_tokens for item in items],
                prompt_offsets=[item.offset for item in items],
                has_mrope=[item.has_mrope for item in items],
            ),
        )
        return out

    def schedule(self) -> SchedulerOutput:
        # [Lwd][perf] 云侧每步 LWD 税分段之一:调度(容器交换)时长
        _t = time.monotonic()
        out = self._schedule_impl()
        self._lwd_sched_step += 1
        self._lwd_log_sched_batch(out)
        logger.info(
            "[Lwd][perf] cloud-sched dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        return out

    def _lwd_log_sched_batch(self, out: SchedulerOutput) -> None:
        """[Lwd][sched] 每步调度批结构化日志(lwd_backlog_probe ⑤ 段锚点)。

        phase 判定:lwd_batch=EMBED 且批内还有 decode 请求即 MIXED;EMBED
        独占为 PREFILL;有排程 token 为 DECODE;否则 EMPTY。pending_notify
        给出待吃量——EMPTY/DECODE 连跑且 pending_notify>0 = 通知已到但
        preflight 不过或 ADD 未落地。reqs 超 12 个截断,防大 decode 批
        刷屏。"""
        batch = getattr(out, "lwd_batch", None)
        if batch is not None and batch.batch_type == LwdBatchType.LWD_EMBED:
            n_prefill = len(batch.batch_meta.req_ids)
            n_reqs = len(out.num_scheduled_tokens)
            phase = "MIXED" if n_reqs > n_prefill else "PREFILL"
            seqno: object = batch.seqno
        else:
            phase = "DECODE" if out.total_num_scheduled_tokens else "EMPTY"
            seqno = ""
        reqs = list(out.num_scheduled_tokens)
        reqs_str = ",".join(reqs[:12]) + (
            f",+{len(reqs) - 12}more" if len(reqs) > 12 else ""
        )
        logger.info(
            "[Lwd][sched] cloud step=%d phase=%s seqno=%s reqs=[%s] tokens=%d "
            "pending_notify=%d decode_ready=%d",
            self._lwd_sched_step, phase, seqno, reqs_str,
            out.total_num_scheduled_tokens, len(self.prefill_notify_queue),
            len(self._lwd_collect_decode_requests()),
        )
