# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive scheduler for non-leader PP ranks.

A `PassiveScheduler` does not make scheduling decisions. It receives
SchedulerOutputs that have already been decided by the leader rank (rank 0)
over a ZMQ subscriber, classifies them by `batch_type`, and emits ready-to-
dispatch payloads — optionally splitting prefill / PD-mix batches into N
layer slices when `VLLM_LAYER_SLICE_SIZE` is set.

The class is intentionally minimal: it shares no implementation with
`vllm.v1.core.sched.scheduler.Scheduler` and depends only on the public
`SchedulerOutput` / `BatchType` types.
"""
import enum
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.sched.output import BatchType, SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.core import PPSchedulerZmqSubscriber

logger = init_logger(__name__)


class DispatchPolicy(enum.Enum):
    """Order in which phase queues are drained inside :meth:`PassiveScheduler.step`.

    EMPTY batches are always drained first (cheap sync messages, must not be
    starved). After that, the three phase queues — PURE_PREFILL, PD_MIX,
    PURE_DECODE — are polled in the order encoded by the policy. One
    SchedulerOutput is picked per non-empty queue per call.
    """
    PREFILL_FIRST = "prefill_first"   # P  → PD-mix → D
    DECODE_FIRST = "decode_first"     # D  → PD-mix → P
    PDMIX_FIRST = "pdmix_first"       # PD-mix → P → D


@dataclass
class LayerSliceInfo:
    """Metadata for a single layer slice in layerwise-disaggregated execution.

    When VLLM_LAYER_SLICE_SIZE > 0, the PassiveScheduler splits the local
    layer range of a single SchedulerOutput into N slices. Each slice carries
    this info so the worker / model_runner can run only the assigned layer
    range and decide whether to perform PP communication.
    """
    slice_index: int       # 0, 1, 2, ...
    total_slices: int      # N
    start_layer: int       # local start layer (0-based within local layers)
    end_layer: int         # local end layer
    is_first_slice: bool   # slice_index == 0
    is_last_slice: bool    # slice_index == total_slices - 1


@dataclass
class ScheduledBatch:
    """Output of `PassiveScheduler.schedule()`: one SchedulerOutput plus the
    plan for how to slice it across layer ranges.

    - For PURE_DECODE / EMPTY batches, or when slicing is disabled:
      ``slices == [None]`` (single full-layer execution, no slice metadata).
    - For PURE_PREFILL / PD_MIX batches with slicing enabled:
      ``slices == [LayerSliceInfo(0), ..., LayerSliceInfo(N-1)]``.

    An empty instance (``slices == []``) signals that no SchedulerOutput was
    available to dispatch this round; the caller should typically idle.
    """
    scheduler_output: SchedulerOutput
    slices: list["LayerSliceInfo | None"]

    @classmethod
    def empty(cls) -> "ScheduledBatch":
        return cls(scheduler_output=None, slices=[])  # type: ignore[arg-type]

    def is_empty(self) -> bool:
        return not self.slices


class PassiveScheduler:
    """Receive → classify → schedule, for non-leader PP ranks.

    Lifecycle (each tick of the engine-core main loop):

        passive_scheduler.poll_and_classify()
        batch = passive_scheduler.schedule()
        if not batch.is_empty():
            for slice_info in batch.slices:
                executor.rpc_broadcast_mq.enqueue(...)

    `schedule()` returns a `ScheduledBatch` with 1 SchedulerOutput plus
    the slice plan; a single PURE_PREFILL / PD_MIX batch may carry N
    layer slices, while PURE_DECODE / EMPTY batches always carry
    `[None]` (single full-layer execution).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        pp_subscriber: "PPSchedulerZmqSubscriber",
        dispatch_policy: DispatchPolicy = DispatchPolicy.PREFILL_FIRST,
        run_subscriber_thread: bool = True,
    ) -> None:
        self.pp_subscriber = pp_subscriber
        self.dispatch_policy = dispatch_policy

        # Per-phase ready queues. EMPTY batches get their own queue so they
        # never block higher-priority P/D work and can be drained in bulk.
        self.ready_prefills: deque[SchedulerOutput] = deque()
        self.ready_pdmixes: deque[SchedulerOutput] = deque()
        self.ready_decodes: deque[SchedulerOutput] = deque()
        self.ready_empties: deque[SchedulerOutput] = deque()

        # Bridge queue between the (optional) subscriber thread and the
        # main loop. When the thread is enabled, it drains
        # `pp_subscriber.consume_new_outputs()` and pushes each SchedulerOutput
        # into `_inbox`; `poll_and_classify` drains `_inbox` instead of
        # touching the subscriber directly.
        self._inbox: queue.Queue[SchedulerOutput] = queue.Queue()
        self._subscriber_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Precompute layer-slice plan once. Mirrors the logic previously
        # inlined in `run_passive_engine_core`.
        self._layer_slice_size = envs.VLLM_LAYER_SLICE_SIZE
        self._num_local_layers = 0
        self._total_slices = 0
        if self._layer_slice_size > 0:
            from vllm.distributed.utils import get_pp_indices
            num_hidden_layers = (
                vllm_config.model_config.hf_config.num_hidden_layers
            )
            pp_size = vllm_config.parallel_config.pipeline_parallel_size
            # PassiveEngineCore is always the non-leader rank (rank 1 for
            # PP=2). Determine its local layer count.
            start_layer_pp, end_layer = get_pp_indices(
                num_hidden_layers, pp_size - 1, pp_size
            )
            self._num_local_layers = end_layer - start_layer_pp
            self._total_slices = math.ceil(
                self._num_local_layers / self._layer_slice_size
            )

        if run_subscriber_thread:
            self.start_subscriber_thread()

    # ------------------------------------------------------------------ #
    # Subscriber thread lifecycle                                        #
    # ------------------------------------------------------------------ #
    def start_subscriber_thread(self) -> None:
        """Spawn a daemon thread that pulls from the ZMQ subscriber and
        pushes SchedulerOutputs into `_inbox`. Idempotent.
        """
        if self._subscriber_thread is not None:
            return
        self._shutdown_event.clear()
        self._subscriber_thread = threading.Thread(
            target=self._subscriber_loop,
            name="PassiveScheduler-Subscriber",
            daemon=True,
        )
        self._subscriber_thread.start()
        logger.debug("PassiveScheduler subscriber thread started.")

    def _subscriber_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                new_outputs = self.pp_subscriber.consume_new_outputs()
            except Exception:
                if self._shutdown_event.is_set():
                    return
                logger.exception(
                    "PassiveScheduler subscriber thread failed to consume."
                )
                return
            if not new_outputs:
                # Avoid a tight spin when the subscriber returns nothing.
                self._shutdown_event.wait(0.001)
                continue
            for _seq, scheduler_output in new_outputs:
                self._inbox.put(scheduler_output)

    def shutdown(self) -> None:
        """Signal the subscriber thread to stop and join it."""
        self._shutdown_event.set()
        thread = self._subscriber_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._subscriber_thread = None

    # ------------------------------------------------------------------ #
    # Inbox draining + classification                                    #
    # ------------------------------------------------------------------ #
    def poll_and_classify(self) -> None:
        """Drain SchedulerOutputs from the inbox (fed by the subscriber
        thread, or directly by `_drain_subscriber_inline` when the thread
        is disabled) and route each into its phase-specific ready queue.
        """
        if self._subscriber_thread is None:
            # Inline mode: pull from the subscriber directly into _inbox.
            self._drain_subscriber_inline()

        while True:
            try:
                scheduler_output = self._inbox.get_nowait()
            except queue.Empty:
                break
            bt = scheduler_output.batch_type
            if bt == BatchType.EMPTY:
                self.ready_empties.append(scheduler_output)
            elif bt == BatchType.PURE_PREFILL:
                self.ready_prefills.append(scheduler_output)
            elif bt == BatchType.PURE_DECODE:
                self.ready_decodes.append(scheduler_output)
            else:  # PD_MIX (or anything unrecognized — treat as mix)
                self.ready_pdmixes.append(scheduler_output)
            logger.debug(
                "PassiveScheduler classified batch_type=%s "
                "(prefills=%d, pdmixes=%d, decodes=%d, empties=%d)",
                bt.value if bt is not None else "<none>",
                len(self.ready_prefills),
                len(self.ready_pdmixes),
                len(self.ready_decodes),
                len(self.ready_empties),
            )

    def _drain_subscriber_inline(self) -> None:
        """Used only when the subscriber thread is disabled (e.g. tests)."""
        new_outputs = self.pp_subscriber.consume_new_outputs()
        for _seq, scheduler_output in new_outputs:
            self._inbox.put(scheduler_output)

    # ------------------------------------------------------------------ #
    # Slicing                                                            #
    # ------------------------------------------------------------------ #
    def _make_slice_info(self, slice_idx: int) -> LayerSliceInfo:
        slice_start = slice_idx * self._layer_slice_size
        slice_end = min(
            slice_start + self._layer_slice_size, self._num_local_layers
        )
        return LayerSliceInfo(
            slice_index=slice_idx,
            total_slices=self._total_slices,
            start_layer=slice_start,
            end_layer=slice_end,
            is_first_slice=(slice_idx == 0),
            is_last_slice=(slice_idx == self._total_slices - 1),
        )

    def _slice_for(
        self, so: SchedulerOutput
    ) -> list["LayerSliceInfo | None"]:
        # Pure decode and empty batches are never sliced.
        if so.batch_type in (BatchType.PURE_DECODE, BatchType.EMPTY):
            return [None]
        # Slicing disabled or trivially 1 slice.
        if self._total_slices <= 1:
            return [None]
        # PURE_PREFILL / PD_MIX → expand into N slice payloads.
        return [self._make_slice_info(i) for i in range(self._total_slices)]

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    _POLICY_ORDER: dict[DispatchPolicy, tuple[str, str, str]] = {
        DispatchPolicy.PREFILL_FIRST: (
            "ready_prefills", "ready_pdmixes", "ready_decodes",
        ),
        DispatchPolicy.DECODE_FIRST: (
            "ready_decodes", "ready_pdmixes", "ready_prefills",
        ),
        DispatchPolicy.PDMIX_FIRST: (
            "ready_pdmixes", "ready_prefills", "ready_decodes",
        ),
    }

    def schedule(self) -> ScheduledBatch:
        """Pick the next SchedulerOutput to dispatch, with its slice plan.

        Policy:
          1. EMPTY batches go first (cheap sync messages, never starved).
          2. Otherwise scan the three phase queues in the order encoded by
             ``self.dispatch_policy`` and pop from the first non-empty one.
          3. If all queues are empty, return :py:meth:`ScheduledBatch.empty`.

        Per call this picks **one** SchedulerOutput. Callers that want to
        drain multiple EMPTY batches per tick should loop until
        ``ScheduledBatch.is_empty()`` and treat EMPTY specially.
        """
        if self.ready_empties:
            so = self.ready_empties.popleft()
            return self._build_batch(so)

        for queue_name in self._POLICY_ORDER[self.dispatch_policy]:
            q: deque[SchedulerOutput] = getattr(self, queue_name)
            if q:
                return self._build_batch(q.popleft())

        return ScheduledBatch.empty()

    def _build_batch(self, so: SchedulerOutput) -> ScheduledBatch:
        batch = ScheduledBatch(scheduler_output=so, slices=self._slice_for(so))
        logger.debug(
            "PassiveScheduler.schedule[%s] picked batch_type=%s slices=%d; "
            "pending=(prefills=%d, pdmixes=%d, decodes=%d, empties=%d)",
            self.dispatch_policy.value,
            so.batch_type.value if so.batch_type is not None else "<none>",
            len(batch.slices),
            len(self.ready_prefills),
            len(self.ready_pdmixes),
            len(self.ready_decodes),
            len(self.ready_empties),
        )
        return batch

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def has_pending(self) -> bool:
        return bool(
            self.ready_prefills
            or self.ready_pdmixes
            or self.ready_decodes
            or self.ready_empties
        )

    @property
    def num_pending(self) -> int:
        return (
            len(self.ready_prefills)
            + len(self.ready_pdmixes)
            + len(self.ready_decodes)
            + len(self.ready_empties)
        )
