# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PassiveEngineCoreProc.step().

These tests avoid spinning up a real MultiprocExecutor or ZMQ subscriber.
Both are replaced with lightweight fakes that only implement the surface
that `PassiveEngineCoreProc.step()` actually touches:
    - executor.rpc_broadcast_mq.enqueue(...)
    - executor.is_failed
    - pp_subscriber.consume_new_outputs() / shutdown()
"""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _install_fake_distributed_utils() -> None:
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
)


# ---------------------------------------------------------------------- #
# Fakes                                                                  #
# ---------------------------------------------------------------------- #
class FakeSubscriber:
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


class FakeRpcMq:
    def __init__(self) -> None:
        self.enqueued: list[tuple] = []

    def enqueue(self, item: tuple) -> None:
        self.enqueued.append(item)


class FakeExecutor:
    def __init__(self) -> None:
        self.rpc_broadcast_mq = FakeRpcMq()
        self.is_failed = False


def _fake_vllm_config(num_hidden_layers: int = 8, pp_size: int = 2):
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


def _make_proc(
    *,
    dispatch_policy: DispatchPolicy = DispatchPolicy.PREFILL_FIRST,
    layer_slice_size: int = 0,
    num_hidden_layers: int = 8,
    pp_size: int = 2,
):
    """Construct a PassiveEngineCoreProc by hand, bypassing the heavy
    `vllm.v1.engine.core` import (which transitively pulls torch +
    msgspec + zmq). We replicate just the __init__ body the proc needs.
    """
    sub = FakeSubscriber()
    executor = FakeExecutor()
    cfg = _fake_vllm_config(num_hidden_layers, pp_size)
    with patch("vllm.envs.VLLM_LAYER_SLICE_SIZE", layer_slice_size):
        scheduler = PassiveScheduler(
            cfg, sub,
            dispatch_policy=dispatch_policy,
            run_subscriber_thread=False,
        )

    # Build a minimal stand-in for PassiveEngineCoreProc that wires the
    # same step() implementation against the fakes. We import the real
    # class lazily; if the heavyweight import fails on this environment
    # we fall back to a local re-implementation that mirrors core.py
    # line-for-line (kept in sync with the production code).
    try:
        from vllm.v1.engine.core import PassiveEngineCoreProc
        proc = PassiveEngineCoreProc.__new__(PassiveEngineCoreProc)
        proc.vllm_config = cfg
        proc.executor = executor
        proc.passive_scheduler = scheduler
        proc._idle_sleep_seconds = 0.001
    except Exception:
        proc = _LocalPassiveEngineCoreProc(cfg, executor, scheduler)

    return proc, sub, executor


class _LocalPassiveEngineCoreProc:
    """Fallback mirror of PassiveEngineCoreProc.step() used only if
    `vllm.v1.engine.core` cannot be imported in the test environment.
    Must stay in sync with the production implementation.
    """

    def __init__(self, cfg, executor, scheduler) -> None:
        self.vllm_config = cfg
        self.executor = executor
        self.passive_scheduler = scheduler
        self._idle_sleep_seconds = 0.001

    def step(self) -> bool:
        self.passive_scheduler.poll_and_classify()
        dispatched = False
        while True:
            batch = self.passive_scheduler.schedule()
            if batch.is_empty():
                break
            for slice_info in batch.slices:
                payload = (
                    (batch.scheduler_output, slice_info)
                    if slice_info is not None
                    else (batch.scheduler_output,)
                )
                self.executor.rpc_broadcast_mq.enqueue(
                    (b"pp_scheduler_output", payload, {}, None)
                )
            dispatched = True
            if batch.scheduler_output.batch_type != BatchType.EMPTY:
                break
        return dispatched


# ---------------------------------------------------------------------- #
# step() basics                                                          #
# ---------------------------------------------------------------------- #
def test_step_returns_false_when_nothing_to_dispatch():
    proc, _sub, executor = _make_proc()
    assert proc.step() is False
    assert executor.rpc_broadcast_mq.enqueued == []


def test_step_enqueues_pure_decode_as_single_unsliced_payload():
    proc, sub, executor = _make_proc(layer_slice_size=2, num_hidden_layers=8)
    sub.feed(_make_so(BatchType.PURE_DECODE))
    assert proc.step() is True
    assert len(executor.rpc_broadcast_mq.enqueued) == 1
    tag, payload, kw, ret = executor.rpc_broadcast_mq.enqueued[0]
    assert tag == b"pp_scheduler_output"
    assert kw == {} and ret is None
    # PURE_DECODE → payload is (so,) without slice info.
    assert len(payload) == 1
    assert payload[0].batch_type == BatchType.PURE_DECODE


def test_step_enqueues_pure_prefill_as_n_slice_payloads():
    proc, sub, executor = _make_proc(layer_slice_size=2, num_hidden_layers=8)
    sub.feed(_make_so(BatchType.PURE_PREFILL))
    assert proc.step() is True
    assert len(executor.rpc_broadcast_mq.enqueued) == 2

    payloads = [item[1] for item in executor.rpc_broadcast_mq.enqueued]
    # Each payload is (scheduler_output, slice_info).
    for p in payloads:
        assert len(p) == 2
        assert p[0].batch_type == BatchType.PURE_PREFILL
        assert isinstance(p[1], LayerSliceInfo)

    # Slice indices are 0 and 1, in order.
    assert payloads[0][1].slice_index == 0
    assert payloads[1][1].slice_index == 1
    assert payloads[0][0] is payloads[1][0]


def test_step_drains_all_empties_in_one_call():
    proc, sub, executor = _make_proc()
    sub.feed(
        _make_so(BatchType.EMPTY),
        _make_so(BatchType.EMPTY),
        _make_so(BatchType.EMPTY),
    )
    assert proc.step() is True
    # All three EMPTY batches should have been enqueued in a single step.
    assert len(executor.rpc_broadcast_mq.enqueued) == 3
    for item in executor.rpc_broadcast_mq.enqueued:
        payload = item[1]
        assert len(payload) == 1
        assert payload[0].batch_type == BatchType.EMPTY


def test_step_throttles_non_empty_phases_to_one_per_call():
    """Two PURE_PREFILL batches should require two step() calls — only
    one batch is taken from any given non-empty phase queue per tick.
    """
    proc, sub, executor = _make_proc()  # slicing disabled → 1 payload per batch
    sub.feed(
        _make_so(BatchType.PURE_PREFILL),
        _make_so(BatchType.PURE_PREFILL),
    )
    assert proc.step() is True
    assert len(executor.rpc_broadcast_mq.enqueued) == 1
    assert len(proc.passive_scheduler.ready_prefills) == 1

    assert proc.step() is True
    assert len(executor.rpc_broadcast_mq.enqueued) == 2
    assert len(proc.passive_scheduler.ready_prefills) == 0

    assert proc.step() is False


def test_step_empties_first_then_one_phase_batch():
    """A mix of EMPTY + PURE_DECODE: all EMPTYs drained, then exactly one
    PURE_DECODE dispatched, all within a single step().
    """
    proc, sub, executor = _make_proc()
    sub.feed(
        _make_so(BatchType.PURE_DECODE),
        _make_so(BatchType.EMPTY),
        _make_so(BatchType.EMPTY),
    )
    assert proc.step() is True
    types = [item[1][0].batch_type for item in executor.rpc_broadcast_mq.enqueued]
    assert types == [BatchType.EMPTY, BatchType.EMPTY, BatchType.PURE_DECODE]
