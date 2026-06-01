# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import BatchType, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class SchedulingPhase(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class PDSeparatedScheduler(Scheduler):
    """Scheduler that separates prefill and decode into distinct steps."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chunk_prefill: list[Request] = []
        self._step_counter: int = 0

    def schedule(self) -> SchedulerOutput:
        return self._schedule_pd_separated()

    def _schedule_pd_separated(self) -> SchedulerOutput:
        phase = self._select_scheduling_phase()
        self._step_counter += 1
        print(
            f"\r\n[PD] Step{self._step_counter}, phase is {phase.value},    "
            f"waiting[]: {len(self.waiting)}, chunk_prefill[]: "
            f"{len(self.chunk_prefill)}, running[]: {len(self.running)}"
        )
        for req in self.chunk_prefill:
            print(
                f"[PD] chunk_prefill[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}"
            )
        for req in self.running:
            print(
                f"[PD] running[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}"
            )
        if phase == SchedulingPhase.PREFILL:
            if not self.chunk_prefill and not self.waiting:
                print(
                    "[PD] prefill phase but no prefill work, "
                    "auto-switch to decode"
                )
                return self._pick_decode_batch()
            return self._pick_prefill_batch()
        else:
            if not self.running:
                print(
                    "[PD] decode phase but no decode work, "
                    "auto-switch to prefill"
                )
                return self._pick_prefill_batch()
            return self._pick_decode_batch()

    def _select_scheduling_phase(self) -> SchedulingPhase:
        policy = self.scheduler_config.pd_scheduling_policy
        if policy == "prefill_first":
            if self.chunk_prefill or self.waiting:
                return SchedulingPhase.PREFILL
            if self.running:
                return SchedulingPhase.DECODE
            return SchedulingPhase.PREFILL
        elif policy == "decode_first":
            if self.running:
                return SchedulingPhase.DECODE
            if self.chunk_prefill or self.waiting:
                return SchedulingPhase.PREFILL
            return SchedulingPhase.DECODE
        elif policy == "strict_alternation":
            return (
                SchedulingPhase.PREFILL
                if self._step_counter % 2 == 0
                else SchedulingPhase.DECODE
            )
        else:
            raise ValueError(f"Unknown PD scheduling policy: {policy}")

    def _pick_prefill_batch(self) -> SchedulerOutput:
        saved_running = self.running
        saved_chunk_prefill = self.chunk_prefill
        saved_max_num_running_reqs = self.max_num_running_reqs

        self.running = list(saved_chunk_prefill)
        self.chunk_prefill = []
        self.max_num_running_reqs -= len(saved_running)

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            self.max_num_running_reqs = saved_max_num_running_reqs
            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                else:
                    scheduler_output.batch_type = BatchType.PURE_PREFILL
                new_chunk_prefill = [
                    req for req in self.running if req.is_prefill_chunk
                ]
                new_running = [
                    req for req in self.running if not req.is_prefill_chunk
                ]
                for req in self.chunk_prefill:
                    if req not in new_chunk_prefill:
                        new_chunk_prefill.append(req)
                self.chunk_prefill = new_chunk_prefill
                self.running = saved_running + new_running
                print(
                    f"[PD] _pick_prefill_batch done: chunk_prefill[]: "
                    f"{len(self.chunk_prefill)}, running[]: {len(self.running)}"
                )
                for (
                    req_id,
                    num_scheduled_token,
                ) in scheduler_output.num_scheduled_tokens.items():
                    req = self.requests[req_id]
                    print(
                        f"[PD] Scheduled[{req_id}],    "
                        f"num_tokens: {req.num_tokens}, "
                        f"num_scheduled_token: {num_scheduled_token}, "
                        f"num_computed_tokens: {req.num_computed_tokens}, "
                        f"is_prefill_chunk: {req.is_prefill_chunk}, "
                        f"chunk_num: {req.chunk_num}"
                    )
            else:
                self.chunk_prefill = saved_chunk_prefill
                self.running = saved_running

        return scheduler_output  # type: ignore[return-value]

    def _pick_decode_batch(self) -> SchedulerOutput:
        saved_chunk_prefill = self.chunk_prefill
        saved_waiting = self.waiting
        saved_skipped = self.skipped_waiting

        self.chunk_prefill = []
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                else:
                    scheduler_output.batch_type = BatchType.PURE_DECODE
                for req in list(self.waiting):
                    saved_waiting.prepend_request(req)
                self.chunk_prefill = saved_chunk_prefill
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped
                print(
                    f"[PD] _pick_decode_batch done: running: {len(self.running)}, "
                    f"chunk_prefill: {len(self.chunk_prefill)}"
                )
                for (
                    req_id,
                    num_scheduled_token,
                ) in scheduler_output.num_scheduled_tokens.items():
                    req = self.requests[req_id]
                    print(
                        f"[PD] Scheduled[{req_id}],    "
                        f"num_tokens: {req.num_tokens}, "
                        f"num_scheduled_token: {num_scheduled_token}, "
                        f"num_computed_tokens: {req.num_computed_tokens}, "
                        f"is_prefill_chunk: {req.is_prefill_chunk}, "
                        f"chunk_num: {req.chunk_num}"
                    )
            else:
                self.chunk_prefill = saved_chunk_prefill
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped

        return scheduler_output  # type: ignore[return-value]

    def _migrate_prefill_to_running(self) -> None:
        completed = [req for req in self.chunk_prefill if not req.is_prefill_chunk]
        if completed:
            print(
                f"[PD] _migrate_prefill_to_running: moving {len(completed)} "
                f"requests from chunk_prefill to running"
            )
        for req in completed:
            self.chunk_prefill.remove(req)
            self.running.append(req)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_preemptions += 1
        if request.spec_token_ids:
            request.spec_token_ids = []
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        if request.is_prefill_chunk:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"stays in chunk_prefill (computed={request.num_computed_tokens})"
            )
            self.chunk_prefill.append(request)
        else:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"goes back to waiting (decode or finished prefill)"
            )
            request.num_computed_tokens = 0
            self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        was_prefill_map = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            was_prefill_map[req_id] = self.requests[req_id].is_prefill_chunk

        super()._update_after_schedule(scheduler_output)

        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            if was_prefill_map[req_id] and num_scheduled_token > 0:
                self.requests[req_id].chunk_num += 1
                print(
                    f"[PD] _update_after_schedule: request {req_id} "
                    f"chunk_num={self.requests[req_id].chunk_num} "
                    f"tokens={self.requests[req_id].num_tokens} "
                    f"scheduled={num_scheduled_token} "
                    f"computed={self.requests[req_id].num_computed_tokens} "
                    f"is_prefill_chunk={self.requests[req_id].is_prefill_chunk}"
                )

        self._migrate_prefill_to_running()
        self.finished_req_ids = set()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self.chunk_prefill = [
            req for req in self.chunk_prefill if not req.is_finished()
        ]
        return outputs

    def get_request_counts(self) -> tuple[int, int]:
        num_running, num_waiting = super().get_request_counts()
        return num_running + len(self.chunk_prefill), num_waiting

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        return super().get_num_unfinished_requests() + len(self.chunk_prefill)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        result = super().finish_requests(request_ids, finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        to_remove = set()
        for req_id in request_ids:
            req = self.requests.get(req_id)
            if req and req.is_finished():
                to_remove.add(req)

        if to_remove:
            self.chunk_prefill = remove_all(self.chunk_prefill, to_remove)

        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        if reset_running_requests:
            timestamp = time.monotonic()
            while self.chunk_prefill:
                request = self.chunk_prefill.pop()
                self.kv_cache_manager.free(request)
                self.encoder_cache_manager.free(request)
                request.status = RequestStatus.PREEMPTED
                request.num_computed_tokens = 0
                if request.spec_token_ids:
                    request.spec_token_ids = []
                request.num_preemptions += 1
                if self.log_stats:
                    request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True
                self.waiting.prepend_request(request)

        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def make_stats(self, *args, **kwargs):
        stats = super().make_stats(*args, **kwargs)
        if stats is not None:
            stats.num_running_reqs += len(self.chunk_prefill)
        return stats

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        saved_running = self.running
        self.running = list(self.running) + [
            r for r in self.chunk_prefill if r not in self.running
        ]
        try:
            result = super()._handle_invalid_blocks(invalid_block_ids)
        finally:
            self.running = saved_running
        return result


class AsyncPDSeparatedScheduler(AsyncScheduler, PDSeparatedScheduler):
    """Async scheduler with PD separation."""
    pass
