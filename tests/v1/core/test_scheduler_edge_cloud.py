# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for edge-cloud async head/tail scheduling."""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.sched.output import BatchType
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID, create_requests, get_request_block_hasher, sha256

pytestmark = pytest.mark.cpu_test


_NONE_HASH_INITIALIZED = False


def _init_none_hash():
    global _NONE_HASH_INITIALIZED
    if not _NONE_HASH_INITIALIZED:
        from vllm.v1.core.kv_cache_utils import init_none_hash

        init_none_hash(sha256)
        _NONE_HASH_INITIALIZED = True


def create_edge_scheduler(
    is_edge_node: bool = True,
    enable_edge_cloud_async_sched: bool = True,
    max_batch_last_depth: int = 2,
    num_requests: int = 10,
    num_tokens: int = 10,
):
    """Create scheduler with edge-cloud async scheduling enabled."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    parallel_config = ParallelConfig(
        enable_edge_cloud=True,
        edge_npu_count=2,
        cloud_npu_count=4,
        is_edge_node=is_edge_node,
        enable_edge_cloud_async_sched=enable_edge_cloud_async_sched,
        max_batch_last_depth=max_batch_last_depth,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = 10000

    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )

    # Create and add requests
    _init_none_hash()
    block_hasher = get_request_block_hasher(16, sha256)
    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(ignore_eos=True, max_tokens=16)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    requests = []
    for i in range(num_requests):
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher,
        )
        scheduler.add_request(request)
        requests.append(request)

    return scheduler, requests


class TestEdgeCloudAsyncDisabled:
    """Tests when edge-cloud async scheduling is disabled."""

    def test_disabled_on_cloud_node(self):
        """Cloud node should NOT have async scheduling enabled."""
        scheduler, _ = create_edge_scheduler(
            is_edge_node=False,
            enable_edge_cloud_async_sched=True,
        )
        assert not scheduler.enable_edge_cloud_async_sched

    def test_disabled_flag_false(self):
        """Edge node with flag=False should have disabled."""
        scheduler, _ = create_edge_scheduler(
            is_edge_node=True,
            enable_edge_cloud_async_sched=False,
        )
        assert not scheduler.enable_edge_cloud_async_sched

    def test_standard_schedule_when_disabled(self):
        """When disabled, schedule() should return FULL batches."""
        scheduler, _ = create_edge_scheduler(
            is_edge_node=True,
            enable_edge_cloud_async_sched=False,
        )
        output = scheduler.schedule()
        assert output.batch_type == BatchType.FULL
        assert output.total_num_scheduled_tokens > 0


class TestEdgeCloudAsyncEnabled:
    """Tests when edge-cloud async scheduling is enabled on edge node."""

    def test_scheduler_has_batch_last(self):
        """Scheduler should have batch_last queue."""
        scheduler, _ = create_edge_scheduler()
        assert hasattr(scheduler, "batch_last")
        assert len(scheduler.batch_last) == 0
        assert scheduler.max_batch_last_depth == 2
        assert scheduler.enable_edge_cloud_async_sched

    def test_first_schedule_returns_batch_first(self):
        """First schedule() should return batch_first."""
        scheduler, _ = create_edge_scheduler()
        output = scheduler.schedule()
        assert output.batch_type == BatchType.FIRST
        assert output.total_num_scheduled_tokens > 0

    def test_second_schedule_returns_batch_first(self):
        """Second schedule() with depth < 2 should also return batch_first."""
        scheduler, _ = create_edge_scheduler()

        # First batch_first
        out1 = scheduler.schedule()
        assert out1.batch_type == BatchType.FIRST

        # Push to batch_last (simulate EngineCore step)
        scheduler.push_batch_last(out1)
        assert len(scheduler.batch_last) == 1
        assert scheduler.get_batch_last_depth() == 1

        # Second batch_first (depth=1 < 2)
        out2 = scheduler.schedule()
        assert out2.batch_type == BatchType.FIRST

    def test_third_schedule_returns_batch_last(self):
        """Third schedule() with depth=2 should return batch_last."""
        scheduler, _ = create_edge_scheduler()

        # Two batch_first, push both to batch_last
        out1 = scheduler.schedule()
        scheduler.push_batch_last(out1)

        out2 = scheduler.schedule()
        scheduler.push_batch_last(out2)

        assert scheduler.get_batch_last_depth() == 2

        # Third schedule: depth == max, must return batch_last
        out3 = scheduler.schedule()
        assert out3.batch_type == BatchType.LAST
        assert out3.first_stage_context is not None
        assert out3.first_stage_context.orig_scheduler_output == out1
        assert scheduler.get_batch_last_depth() == 1

    def test_batch_last_fifo_order(self):
        """batch_last[] should be FIFO."""
        scheduler, _ = create_edge_scheduler()

        out1 = scheduler.schedule()
        out1.step_id = 100
        scheduler.push_batch_last(out1)

        out2 = scheduler.schedule()
        out2.step_id = 200
        scheduler.push_batch_last(out2)

        # First batch_last should be out1 (FIFO)
        out_last1 = scheduler.schedule()
        assert out_last1.batch_type == BatchType.LAST
        assert out_last1.step_id == 100

        # Second batch_last should be out2
        out_last2 = scheduler.schedule()
        assert out_last2.batch_type == BatchType.LAST
        assert out_last2.step_id == 200

    def test_batch_last_then_batch_first(self):
        """After batch_last, if depth < 2, can schedule batch_first again."""
        scheduler, _ = create_edge_scheduler()

        # Fill batch_last to depth 2
        out1 = scheduler.schedule()
        scheduler.push_batch_last(out1)
        out2 = scheduler.schedule()
        scheduler.push_batch_last(out2)

        # batch_last pops one
        out_last = scheduler.schedule()
        assert out_last.batch_type == BatchType.LAST
        assert scheduler.get_batch_last_depth() == 1

        # Now can schedule batch_first again
        out3 = scheduler.schedule()
        assert out3.batch_type == BatchType.FIRST

    def test_empty_when_no_work(self):
        """When no requests and batch_last empty, return EMPTY."""
        scheduler, _ = create_edge_scheduler(num_requests=0)
        output = scheduler.schedule()
        assert output.batch_type == BatchType.EMPTY
        assert output.total_num_scheduled_tokens == 0

    def test_push_batch_last_marks_head_done(self):
        """push_batch_last should mark requests as HEAD_DONE."""
        scheduler, requests = create_edge_scheduler()

        output = scheduler.schedule()
        # Requests should be RUNNING after standard schedule
        for req_id in output.num_scheduled_tokens:
            assert scheduler.requests[req_id].status == RequestStatus.RUNNING

        scheduler.push_batch_last(output)

        # Requests should now be HEAD_DONE
        for req_id in output.num_scheduled_tokens:
            assert scheduler.requests[req_id].status == RequestStatus.HEAD_DONE

    def test_has_head_done_requests(self):
        """has_head_done_requests should reflect batch_last state."""
        scheduler, _ = create_edge_scheduler()
        assert not scheduler.has_head_done_requests()

        output = scheduler.schedule()
        scheduler.push_batch_last(output)
        assert scheduler.has_head_done_requests()

    def test_get_num_unfinished_includes_batch_last(self):
        """Unfinished count should include HEAD_DONE requests in batch_last."""
        scheduler, _ = create_edge_scheduler()

        output = scheduler.schedule()
        scheduler.push_batch_last(output)

        # All requests are HEAD_DONE (in batch_last), none finished
        num_unfinished = scheduler.get_num_unfinished_requests()
        assert num_unfinished == len(output.num_scheduled_tokens)

    def test_finish_requests_clears_batch_last(self):
        """Aborting requests should clear them from batch_last."""
        scheduler, _ = create_edge_scheduler()

        output = scheduler.schedule()
        req_ids = list(output.num_scheduled_tokens.keys())
        scheduler.push_batch_last(output)

        assert scheduler.get_batch_last_depth() == 1

        # Abort one request
        scheduler.finish_requests(req_ids[0], RequestStatus.FINISHED_ABORTED)

        # batch_last should be cleared (since it contained aborted request)
        assert scheduler.get_batch_last_depth() == 0
        # Aborted request should be gone
        assert req_ids[0] not in scheduler.requests


class TestUpdateFromOutputEdgeCloud:
    """Tests for update_from_output with edge-cloud batch types."""

    def test_batch_first_skips_sampling(self):
        """batch_first should skip sampling and output generation."""
        scheduler, _ = create_edge_scheduler()

        output = scheduler.schedule()
        assert output.batch_type == BatchType.FIRST

        # Simulate model execution (return empty output)
        model_output = EMPTY_MODEL_RUNNER_OUTPUT

        # update_from_output should not generate EngineCoreOutputs
        engine_outputs = scheduler.update_from_output(output, model_output)
        assert len(engine_outputs) == 0

        # Requests should still be RUNNING (before push_batch_last)
        for req_id in output.num_scheduled_tokens:
            assert scheduler.requests[req_id].status == RequestStatus.RUNNING

    def test_batch_last_restores_head_done(self):
        """batch_last should restore HEAD_DONE requests to RUNNING."""
        scheduler, _ = create_edge_scheduler()

        # batch_first
        out_first = scheduler.schedule()
        scheduler.push_batch_last(out_first)

        # batch_last
        out_last = scheduler.schedule()
        assert out_last.batch_type == BatchType.LAST

        # Simulate model execution with tokens
        req_ids = list(out_last.num_scheduled_tokens.keys())
        model_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[[1] for _ in req_ids],
        )

        # update_from_output
        engine_outputs = scheduler.update_from_output(out_last, model_output)

        # Requests should be restored to RUNNING or FINISHED
        for req_id in req_ids:
            status = scheduler.requests[req_id].status
            assert status in (RequestStatus.RUNNING, RequestStatus.FINISHED_STOPPED)
