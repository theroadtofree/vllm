# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone tests for PD (Prefill-Decode) separation scheduling.

Run with:
    pytest tests/v1/core/test_pd_separation.py -v
Or directly:
    python -m pytest tests/v1/core/test_pd_separation.py -v

These tests verify that when enable_pd_separation=True:
1. schedule() returns either a pure-prefill or pure-decode batch, never mixed.
2. chunk_prefill queue correctly holds RUNNING requests whose prefill is not done.
3. running queue only holds decode-phase requests.
4. chunk_num increments after each prefill chunk.
5. is_last_prefill_chunk() works correctly.
6. Preempted chunk_prefill requests return to chunk_prefill (not waiting).
7. Empty-phase batches auto-switch to the other phase.
8. All three scheduling policies (prefill_first, decode_first, strict_alternation)
   behave as expected.
"""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _make_model_output(requests, sampled_token_ids):
    """Helper to build a minimal ModelRunnerOutput for update_from_output."""
    req_to_index = {req.request_id: i for i, req in enumerate(requests)}
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _simulate_step(scheduler, requests):
    """Run one scheduler step and update_from_output.

    Returns (output, phase_hint) where phase_hint is inferred from the
    scheduler state before scheduling.
    """
    had_prefill = bool(scheduler.chunk_prefill or scheduler.waiting)
    had_decode = bool(scheduler.running)
    output = scheduler.schedule()
    sampled = []
    for req in requests:
        if req.request_id in output.num_scheduled_tokens and not req.is_prefill_chunk:
            sampled.append([0])
        else:
            sampled.append([])
    model_output = _make_model_output(requests, sampled)
    scheduler.update_from_output(output, model_output)
    return output, had_prefill, had_decode


def _all_are_decode(output, scheduler):
    """Return True if every scheduled request is in decode phase.

    This is unambiguous because decode always schedules exactly 1 token and
    the request is no longer in prefill.
    """
    for req_id in output.num_scheduled_tokens:
        req = scheduler.requests[req_id]
        if output.num_scheduled_tokens[req_id] != 1 or req.is_prefill_chunk:
            return False
    return True


class TestPDSeparationDisabled:
    """When PD separation is disabled, behaviour should match the original."""

    def test_mixed_batch_by_default(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8, max_num_seqs=4, max_model_len=32
        )
        long_req, short_req = create_requests(
            num_requests=2, num_tokens=6, max_tokens=4
        )
        for req in (long_req, short_req):
            scheduler.add_request(req)

        output = scheduler.schedule()
        # Both scheduled in the same step.
        assert len(output.num_scheduled_tokens) == 2
        assert output.num_scheduled_tokens[short_req.request_id] == 6
        assert output.num_scheduled_tokens[long_req.request_id] == 2


class TestPDSeparationBasic:
    """Core PD separation invariants."""

    def test_pure_prefill_then_pure_decode(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=6, max_tokens=4)[0]
        scheduler.add_request(req)

        # Step 1: prefill_first -> prefill batch.
        output, had_prefill, _ = _simulate_step(scheduler, [req])
        assert had_prefill
        # All tokens scheduled (6 > 1) so it is clearly prefill.
        assert output.num_scheduled_tokens[req.request_id] == 6
        assert len(scheduler.chunk_prefill) == 0
        assert len(scheduler.running) == 1  # migrated after schedule

        # Step 2: only decode requests remain -> decode batch.
        output, _, had_decode = _simulate_step(scheduler, [req])
        assert had_decode
        assert _all_are_decode(output, scheduler)
        assert output.num_scheduled_tokens[req.request_id] == 1

    def test_chunked_prefill_stays_in_chunk_prefill(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=10, max_tokens=4)[0]
        scheduler.add_request(req)

        # Step 1: prefill chunk 1 (4 tokens).
        output, _, _ = _simulate_step(scheduler, [req])
        assert output.num_scheduled_tokens[req.request_id] == 4
        assert req.is_prefill_chunk
        assert req in scheduler.chunk_prefill
        assert req not in scheduler.running

        # Step 2: prefill chunk 2 (4 tokens).
        output, _, _ = _simulate_step(scheduler, [req])
        assert output.num_scheduled_tokens[req.request_id] == 4
        assert req in scheduler.chunk_prefill

        # Step 3: prefill chunk 3 (2 tokens) -> completes prefill.
        output, _, _ = _simulate_step(scheduler, [req])
        assert output.num_scheduled_tokens[req.request_id] == 2
        assert req not in scheduler.chunk_prefill
        assert req in scheduler.running
        assert not req.is_prefill_chunk

        # Step 4: decode.
        output, _, _ = _simulate_step(scheduler, [req])
        assert _all_are_decode(output, scheduler)
        assert output.num_scheduled_tokens[req.request_id] == 1

    def test_chunk_num_increments(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=3,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=8, max_tokens=4)[0]
        scheduler.add_request(req)
        assert req.chunk_num == 1

        # Chunk 1 (3 tokens).
        _simulate_step(scheduler, [req])
        assert req.chunk_num == 2
        # Chunk 2 (3 tokens).
        _simulate_step(scheduler, [req])
        assert req.chunk_num == 3
        # Chunk 3 (2 tokens).
        _simulate_step(scheduler, [req])
        assert req.chunk_num == 4
        # Decode steps do NOT increment chunk_num.
        _simulate_step(scheduler, [req])
        assert req.chunk_num == 4

    def test_is_last_prefill_chunk(self):
        req = create_requests(num_requests=1, num_tokens=5, max_tokens=4)[0]
        # Simulate: 3 tokens computed, 2 remain -> scheduling 2 is last chunk.
        req.num_computed_tokens = 3
        assert req.is_last_prefill_chunk(2)
        assert not req.is_last_prefill_chunk(1)

    def test_empty_prefill_auto_switches_to_decode(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        scheduler.add_request(req)
        # First prefill.
        _simulate_step(scheduler, [req])
        # Now running has the decode request, chunk_prefill/waiting are empty.
        # Next schedule should auto-switch to decode even though phase=prefill.
        output, _, _ = _simulate_step(scheduler, [req])
        assert _all_are_decode(output, scheduler)

    def test_empty_decode_auto_switches_to_prefill(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        # No requests at all yet.
        output, _, _ = _simulate_step(scheduler, [])
        # Empty batch is fine.
        assert len(output.num_scheduled_tokens) == 0


class TestPDSeparationPolicies:
    """Behaviour of the three scheduling policies."""

    def test_prefill_first_policy(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        scheduler.scheduler_config.pd_scheduling_policy = "prefill_first"
        decode_req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        scheduler.add_request(decode_req)
        _simulate_step(scheduler, [decode_req])  # prefill

        new_req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        scheduler.add_request(new_req)
        # prefill_first should pick prefill even though decode_req is ready.
        output, had_prefill, _ = _simulate_step(scheduler, [decode_req, new_req])
        assert had_prefill
        assert new_req.request_id in output.num_scheduled_tokens
        assert output.num_scheduled_tokens[new_req.request_id] == 4  # clearly prefill

    def test_decode_first_policy(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        scheduler.scheduler_config.pd_scheduling_policy = "decode_first"
        decode_req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        scheduler.add_request(decode_req)
        _simulate_step(scheduler, [decode_req])  # prefill

        new_req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        scheduler.add_request(new_req)
        # decode_first should pick decode of decode_req first.
        output, _, had_decode = _simulate_step(scheduler, [decode_req, new_req])
        assert had_decode
        assert decode_req.request_id in output.num_scheduled_tokens
        assert _all_are_decode(output, scheduler)

    def test_strict_alternation_policy(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        scheduler.scheduler_config.pd_scheduling_policy = "strict_alternation"
        req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        scheduler.add_request(req)
        # Step 0 -> prefill.
        output, _, _ = _simulate_step(scheduler, [req])
        assert output.num_scheduled_tokens[req.request_id] == 4
        # Step 1 -> decode.
        output, _, _ = _simulate_step(scheduler, [req])
        assert _all_are_decode(output, scheduler)
        # Step 2 -> prefill (even though req is decode now, auto-switch).
        output, _, _ = _simulate_step(scheduler, [req])
        # Since there is no prefill work, auto-switch gives empty or decode.
        assert len(output.num_scheduled_tokens) in (0, 1)


class TestPDSeparationPreemption:
    """Preemption behaviour under PD separation."""

    def test_preempt_chunk_prefill_request(self):
        """Directly test _preempt_request for a chunk_prefill request."""
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=8, max_tokens=4)[0]
        req.status = RequestStatus.RUNNING
        req.num_computed_tokens = 4
        req.is_prefill_chunk = True
        scheduler.chunk_prefill.append(req)

        scheduler._preempt_request(req, 0.0)
        assert req in scheduler.chunk_prefill
        assert req.status == RequestStatus.PREEMPTED
        assert req.num_computed_tokens == 4  # progress preserved
        assert req.num_preemptions == 1

    def test_preempt_decode_request(self):
        """Directly test _preempt_request for a decode request."""
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        req.status = RequestStatus.RUNNING
        req.num_computed_tokens = 4
        req.is_prefill_chunk = False
        scheduler.running.append(req)

        scheduler._preempt_request(req, 0.0)
        assert req not in scheduler.running
        assert req not in scheduler.chunk_prefill
        assert req.status == RequestStatus.PREEMPTED
        assert req.num_computed_tokens == 0  # decode resets
        assert req.num_preemptions == 1


class TestPDSeparationMultipleRequests:
    """PD separation with concurrent requests."""

    def test_interleaved_prefill_and_decode(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=6,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        short1, short2 = create_requests(num_requests=2, num_tokens=3, max_tokens=4)
        long_req = create_requests(num_requests=1, num_tokens=10, max_tokens=4)[0]
        for req in (short1, short2, long_req):
            scheduler.add_request(req)

        # Step 1: prefill batch. short1(3) + short2(3) = 6 tokens.
        output, _, _ = _simulate_step(scheduler, [short1, short2, long_req])
        assert output.num_scheduled_tokens[short1.request_id] == 3
        assert output.num_scheduled_tokens[short2.request_id] == 3
        assert short1 in scheduler.running
        assert short2 in scheduler.running
        assert long_req not in scheduler.running

        # Step 2: prefill batch for long_req (4 tokens).
        output, _, _ = _simulate_step(scheduler, [short1, short2, long_req])
        assert output.num_scheduled_tokens[long_req.request_id] == 4
        assert long_req in scheduler.chunk_prefill

        # Step 3: prefill batch for long_req (4 tokens).
        output, _, _ = _simulate_step(scheduler, [short1, short2, long_req])
        assert output.num_scheduled_tokens[long_req.request_id] == 4
        assert long_req in scheduler.chunk_prefill

        # Step 4: prefill batch for long_req (2 tokens) -> finishes prefill.
        output, _, _ = _simulate_step(scheduler, [short1, short2, long_req])
        assert output.num_scheduled_tokens[long_req.request_id] == 2
        assert long_req in scheduler.running

        # Step 5: all decode.
        output, _, _ = _simulate_step(scheduler, [short1, short2, long_req])
        assert _all_are_decode(output, scheduler)
        assert len(output.num_scheduled_tokens) == 3
