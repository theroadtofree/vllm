# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for edge-cloud async scheduling config validation."""

import pytest

from vllm.config import ParallelConfig


class TestEdgeCloudAsyncConfig:
    """Tests for edge-cloud async scheduling configuration."""

    def test_default_disabled(self):
        """By default, edge-cloud async scheduling should be disabled."""
        config = ParallelConfig()
        assert not config.enable_edge_cloud_async_sched
        assert config.max_batch_last_depth == 2

    def test_enable_requires_edge_cloud(self):
        """enable_edge_cloud_async_sched=True requires enable_edge_cloud=True."""
        with pytest.raises(ValueError, match="requires enable_edge_cloud=True"):
            ParallelConfig(
                enable_edge_cloud=False,
                enable_edge_cloud_async_sched=True,
            )

    def test_positive_depth_required(self):
        """max_batch_last_depth must be >= 1 when async is enabled."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ParallelConfig(
                enable_edge_cloud=True,
                edge_npu_count=2,
                cloud_npu_count=4,
                enable_edge_cloud_async_sched=True,
                max_batch_last_depth=0,
            )

    def test_valid_edge_async_config(self):
        """Valid edge-cloud async config should not raise."""
        config = ParallelConfig(
            enable_edge_cloud=True,
            edge_npu_count=2,
            cloud_npu_count=4,
            enable_edge_cloud_async_sched=True,
            max_batch_last_depth=2,
        )
        assert config.enable_edge_cloud_async_sched
        assert config.max_batch_last_depth == 2

    def test_valid_edge_async_config_depth_3(self):
        """Depth can be > 2."""
        config = ParallelConfig(
            enable_edge_cloud=True,
            edge_npu_count=2,
            cloud_npu_count=4,
            enable_edge_cloud_async_sched=True,
            max_batch_last_depth=3,
        )
        assert config.max_batch_last_depth == 3
