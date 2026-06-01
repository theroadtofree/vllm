# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PassiveScheduler (rank1 PP non-leader scheduler).

These tests intentionally avoid spinning up a real VllmConfig or ZMQ
subscriber. The PassiveScheduler only touches `vllm_config` to compute
the layer-slice plan, and only touches the subscriber's
`consume_new_outputs` method, so both are replaced with lightweight
fakes.
"""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _install_fake_distributed_utils() -> None:
    """Inject a fake `vllm.distributed.utils.get_pp_indices` so that
    importing PassiveScheduler in environments without a fully-built
    `vllm.distributed` module does not pull in CUDA/torch custom-op
    registrations.

    This mirrors the real signature:
        get_pp_indices(num_hidden_layers, pp_rank, pp_size) -> (start, end)
    """
    if "vllm.distributed.utils" in sys.modules:
        return

    def get_pp_indices(num_hidden_layers: int, pp_rank: int,
                       pp_size: int) -> tuple[int, int]:
        layers_per_rank = num_hidden_layers // pp_size
        start = pp_rank * layers_per_rank
        end = (
            num_hidden_layers
            if pp_rank == pp_size - 1
            else start + layers_per_rank
        )
        return start, end

    fake_distributed = ModuleType("vllm.distributed")
    fake_utils = ModuleType("vllm.distributed.utils")
    fake_utils.get_pp_indices = get_pp_indices
    fake_distributed.utils = fake_utils
    sys.modules.setdefault("vllm.distributed", fake_distributed)
    sys.modules["vllm.distributed.utils"] = fake_utils


_install_fake_distributed_utils()

from vllm.v1.core.sched.output import BatchType, SchedulerOutput  # noqa: E402
from vllm.v1.core.sched.passive_scheduler import (  # noqa: E402
    DispatchPolicy,
    LayerSliceInfo,
    PassiveScheduler,
    ScheduledBatch,
)


def _drain_schedule(scheduler: PassiveScheduler) -> list[ScheduledBatch]:
    """Loop `schedule()` until empty, returning all picked batches in order.
    Helper for tests that want to compare multi-batch dispatch sequences.
    """
    out: list[ScheduledBatch] = []
    while True:
        batch = scheduler.schedule()
        if batch.is_empty():
            break
        out.append(batch)
    return out


# ---------------------------------------------------------------------- #
# Fakes                                                                  #
# ---------------------------------------------------------------------- #
class FakeSubscriber:
    """Minimal stand-in for PPSchedulerZmqSubscriber.

    `consume_new_outputs` returns whatever the test pre-loaded via
    `feed`. Each call drains the buffer (matches the real subscriber's
    "consume once" semantics).
    """

    def __init__(self) -> None:
        self._buffer: list[tuple[int, SchedulerOutput]] = []
        self._seq = 0

    def feed(self, *scheduler_outputs: SchedulerOutput) -> None:
        for so in scheduler_outputs:
            self._buffer.append((self._seq, so))
            self._seq += 1

    def consume_new_outputs(self) -> list[tuple[int, SchedulerOutput]]:
        out = self._buffer
        self._buffer = []
        return out

    def shutdown(self) -> None:
        pass


def _fake_vllm_config(num_hidden_layers: int = 8, pp_size: int = 2):
    """Return a SimpleNamespace duck-typed to look like a VllmConfig
    in the eyes of `PassiveScheduler.__init__`.
    """
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=num_hidden_layers),
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
    )


def _make_so(batch_type: BatchType) -> SchedulerOutput:
    so = SchedulerOutput.make_empty()
    so.batch_type = batch_type
    return so


def _make_scheduler(
    *,
    dispatch_policy: DispatchPolicy = DispatchPolicy.PREFILL_FIRST,
    layer_slice_size: int = 0,
    num_hidden_layers: int = 8,
    pp_size: int = 2,
) -> tuple[PassiveScheduler, FakeSubscriber]:
    sub = FakeSubscriber()
    cfg = _fake_vllm_config(num_hidden_layers, pp_size)
    with patch("vllm.envs.VLLM_LAYER_SLICE_SIZE", layer_slice_size):
        scheduler = PassiveScheduler(
            cfg,
            sub,
            dispatch_policy=dispatch_policy,
            run_subscriber_thread=False,
        )
    return scheduler, sub


# ---------------------------------------------------------------------- #
# Classification                                                         #
# ---------------------------------------------------------------------- #
def test_classify_routes_by_batch_type():
    scheduler, sub = _make_scheduler()
    sub.feed(
        _make_so(BatchType.PURE_PREFILL),
        _make_so(BatchType.PD_MIX),
        _make_so(BatchType.PURE_DECODE),
        _make_so(BatchType.EMPTY),
    )
    scheduler.poll_and_classify()
    assert len(scheduler.ready_prefills) == 1
    assert len(scheduler.ready_pdmixes) == 1
    assert len(scheduler.ready_decodes) == 1
    assert len(scheduler.ready_empties) == 1


def test_classify_unknown_falls_into_pdmix():
    """An untagged batch_type (None) should land in the PD-mix queue."""
    scheduler, sub = _make_scheduler()
    so = SchedulerOutput.make_empty()
    so.batch_type = None  # type: ignore[assignment]
    sub.feed(so)
    scheduler.poll_and_classify()
    assert len(scheduler.ready_pdmixes) == 1
    assert len(scheduler.ready_prefills) == 0
    assert len(scheduler.ready_decodes) == 0


# ---------------------------------------------------------------------- #
# Slicing                                                                #
# ---------------------------------------------------------------------- #
def test_pure_decode_never_sliced():
    scheduler, sub = _make_scheduler(layer_slice_size=2, num_hidden_layers=8)
    # With 8 hidden layers, PP=2 → 4 local layers, slice_size=2 → 2 slices.
    assert scheduler._total_slices == 2
    sub.feed(_make_so(BatchType.PURE_DECODE))
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert batch.scheduler_output.batch_type == BatchType.PURE_DECODE
    assert batch.slices == [None]


def test_empty_never_sliced():
    scheduler, sub = _make_scheduler(layer_slice_size=2, num_hidden_layers=8)
    sub.feed(_make_so(BatchType.EMPTY))
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert batch.scheduler_output.batch_type == BatchType.EMPTY
    assert batch.slices == [None]


def test_pure_prefill_sliced_into_n_slices():
    scheduler, sub = _make_scheduler(layer_slice_size=2, num_hidden_layers=8)
    assert scheduler._total_slices == 2
    sub.feed(_make_so(BatchType.PURE_PREFILL))
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert batch.scheduler_output.batch_type == BatchType.PURE_PREFILL
    assert len(batch.slices) == 2
    info0, info1 = batch.slices
    assert isinstance(info0, LayerSliceInfo)
    assert info0.slice_index == 0
    assert info0.total_slices == 2
    assert info0.start_layer == 0
    assert info0.end_layer == 2
    assert info0.is_first_slice is True
    assert info0.is_last_slice is False
    assert info1.slice_index == 1
    assert info1.start_layer == 2
    assert info1.end_layer == 4
    assert info1.is_first_slice is False
    assert info1.is_last_slice is True


def test_pdmix_sliced_like_pure_prefill():
    scheduler, sub = _make_scheduler(layer_slice_size=2, num_hidden_layers=8)
    sub.feed(_make_so(BatchType.PD_MIX))
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert len(batch.slices) == 2
    assert all(isinstance(info, LayerSliceInfo) for info in batch.slices)


def test_no_slicing_when_disabled():
    scheduler, sub = _make_scheduler(layer_slice_size=0)
    assert scheduler._total_slices == 0
    sub.feed(_make_so(BatchType.PURE_PREFILL))
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert batch.slices == [None]


def test_slice_size_larger_than_local_layers_yields_one_slice():
    # 8 hidden layers, PP=2 → 4 local; slice_size=16 → 1 slice → no slicing.
    scheduler, sub = _make_scheduler(layer_slice_size=16, num_hidden_layers=8)
    assert scheduler._total_slices == 1
    sub.feed(_make_so(BatchType.PURE_PREFILL))
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert batch.slices == [None]


def test_uneven_slice_tail_is_clamped():
    # 4 local layers, slice_size=3 → ceil(4/3)=2 slices: [0,3) and [3,4).
    scheduler, sub = _make_scheduler(layer_slice_size=3, num_hidden_layers=8)
    assert scheduler._total_slices == 2
    sub.feed(_make_so(BatchType.PURE_PREFILL))
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert len(batch.slices) == 2
    info1 = batch.slices[1]
    assert info1.start_layer == 3
    assert info1.end_layer == 4
    assert info1.is_last_slice is True


# ---------------------------------------------------------------------- #
# Dispatch policy                                                        #
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "policy, expected_order",
    [
        (
            DispatchPolicy.PREFILL_FIRST,
            [BatchType.PURE_PREFILL, BatchType.PD_MIX, BatchType.PURE_DECODE],
        ),
        (
            DispatchPolicy.DECODE_FIRST,
            [BatchType.PURE_DECODE, BatchType.PD_MIX, BatchType.PURE_PREFILL],
        ),
        (
            DispatchPolicy.PDMIX_FIRST,
            [BatchType.PD_MIX, BatchType.PURE_PREFILL, BatchType.PURE_DECODE],
        ),
    ],
)
def test_dispatch_policy_order(policy, expected_order):
    scheduler, sub = _make_scheduler(dispatch_policy=policy)
    sub.feed(
        _make_so(BatchType.PURE_PREFILL),
        _make_so(BatchType.PURE_DECODE),
        _make_so(BatchType.PD_MIX),
    )
    scheduler.poll_and_classify()
    batches = _drain_schedule(scheduler)
    assert [b.scheduler_output.batch_type for b in batches] == expected_order


def test_empty_drained_before_phase_queues_regardless_of_policy():
    scheduler, sub = _make_scheduler(dispatch_policy=DispatchPolicy.DECODE_FIRST)
    sub.feed(
        _make_so(BatchType.PURE_DECODE),
        _make_so(BatchType.EMPTY),
        _make_so(BatchType.EMPTY),
    )
    scheduler.poll_and_classify()
    batches = _drain_schedule(scheduler)
    types = [b.scheduler_output.batch_type for b in batches]
    assert types == [BatchType.EMPTY, BatchType.EMPTY, BatchType.PURE_DECODE]


def test_schedule_picks_one_at_a_time():
    scheduler, sub = _make_scheduler()
    sub.feed(
        _make_so(BatchType.PURE_PREFILL),
        _make_so(BatchType.PURE_PREFILL),
    )
    scheduler.poll_and_classify()
    batch = scheduler.schedule()
    assert not batch.is_empty()
    assert batch.scheduler_output.batch_type == BatchType.PURE_PREFILL
    assert len(scheduler.ready_prefills) == 1

    batch2 = scheduler.schedule()
    assert not batch2.is_empty()
    assert batch2.scheduler_output.batch_type == BatchType.PURE_PREFILL
    assert len(scheduler.ready_prefills) == 0

    batch3 = scheduler.schedule()
    assert batch3.is_empty()


# ---------------------------------------------------------------------- #
# Introspection                                                          #
# ---------------------------------------------------------------------- #
def test_has_pending_and_num_pending():
    scheduler, sub = _make_scheduler()
    assert scheduler.has_pending() is False
    assert scheduler.num_pending == 0

    sub.feed(
        _make_so(BatchType.PURE_PREFILL),
        _make_so(BatchType.EMPTY),
    )
    scheduler.poll_and_classify()
    assert scheduler.has_pending() is True
    assert scheduler.num_pending == 2


# ---------------------------------------------------------------------- #
# Subscriber thread (T2)                                                 #
# ---------------------------------------------------------------------- #
def test_subscriber_thread_bridges_inbox():
    import time

    sub = FakeSubscriber()
    cfg = _fake_vllm_config()
    with patch("vllm.envs.VLLM_LAYER_SLICE_SIZE", 0):
        scheduler = PassiveScheduler(
            cfg, sub, run_subscriber_thread=True
        )

    try:
        sub.feed(
            _make_so(BatchType.PURE_PREFILL),
            _make_so(BatchType.PURE_DECODE),
        )
        deadline = time.time() + 1.0
        while time.time() < deadline:
            scheduler.poll_and_classify()
            if (scheduler.ready_prefills
                    and scheduler.ready_decodes):
                break
            time.sleep(0.005)
        assert len(scheduler.ready_prefills) == 1
        assert len(scheduler.ready_decodes) == 1
    finally:
        scheduler.shutdown()
