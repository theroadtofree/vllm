# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for edge-cloud async scheduling output types."""

import pytest

from vllm.v1.core.sched.output import BatchType, FirstStageContext, SchedulerOutput


def test_batch_type_enum():
    """Test BatchType enum values."""
    assert BatchType.FULL.value == "full"
    assert BatchType.FIRST.value == "first"
    assert BatchType.LAST.value == "last"
    assert BatchType.EMPTY.value == "empty"


def test_scheduler_output_default_batch_type():
    """Test that SchedulerOutput defaults to FULL batch_type."""
    so = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SchedulerOutput.__dataclass_fields__[
            "scheduled_cached_reqs"
        ].type(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    assert so.batch_type == BatchType.FULL
    assert so.first_stage_context is None
    assert so.step_id == 0


def test_scheduler_output_empty():
    """Test SchedulerOutput.make_empty creates EMPTY batch."""
    so = SchedulerOutput.make_empty()
    assert so.batch_type == BatchType.EMPTY
    assert so.total_num_scheduled_tokens == 0
    assert so.step_id == 0


def test_first_stage_context():
    """Test FirstStageContext construction."""
    orig_so = SchedulerOutput.make_empty()
    orig_so.step_id = 5

    ctx = FirstStageContext(
        orig_scheduler_output=orig_so,
        enqueue_timestamp=12345.0,
    )
    assert ctx.orig_scheduler_output.step_id == 5
    assert ctx.enqueue_timestamp == 12345.0


def test_scheduler_output_with_batch_type():
    """Test constructing SchedulerOutput with explicit batch_type."""
    so = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SchedulerOutput.__dataclass_fields__[
            "scheduled_cached_reqs"
        ].type(),
        num_scheduled_tokens={"req0": 10},
        total_num_scheduled_tokens=10,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        batch_type=BatchType.FIRST,
        step_id=3,
    )
    assert so.batch_type == BatchType.FIRST
    assert so.step_id == 3
    assert so.first_stage_context is None


def test_scheduler_output_last_with_context():
    """Test LAST SchedulerOutput with FirstStageContext."""
    orig_so = SchedulerOutput.make_empty()
    ctx = FirstStageContext(orig_scheduler_output=orig_so)

    so = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SchedulerOutput.__dataclass_fields__[
            "scheduled_cached_reqs"
        ].type(),
        num_scheduled_tokens={"req0": 10},
        total_num_scheduled_tokens=10,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        batch_type=BatchType.LAST,
        first_stage_context=ctx,
        step_id=2,
    )
    assert so.batch_type == BatchType.LAST
    assert so.first_stage_context is not None
    assert so.first_stage_context.orig_scheduler_output == orig_so
