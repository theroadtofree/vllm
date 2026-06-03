# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for edge-cloud async scheduling in EngineCore step()."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.core.sched.output import BatchType, SchedulerOutput
from vllm.v1.engine.core import EngineCore


class TestEngineCoreStepEdgeCloud:
    """Tests for EngineCore.step() with edge-cloud batch types."""

    def test_empty_batch_returns_early(self):
        """EMPTY batch should return ({}, False) immediately."""
        mock_scheduler = MagicMock()
        mock_scheduler.has_requests.return_value = True
        mock_scheduler.schedule.return_value = SchedulerOutput.make_empty()

        mock_executor = MagicMock()

        engine = EngineCore.__new__(EngineCore)
        engine.scheduler = mock_scheduler
        engine.model_executor = mock_executor
        engine.async_scheduling = False
        engine.use_spec_decode = False

        outputs, model_executed = engine.step()

        assert outputs == {}
        assert not model_executed
        mock_executor.execute_model.assert_not_called()

    def test_batch_first_calls_push_batch_last(self):
        """FIRST batch should call push_batch_last and return empty outputs."""
        mock_scheduler = MagicMock()
        so = SchedulerOutput.make_empty()
        so.batch_type = BatchType.FIRST
        so.total_num_scheduled_tokens = 10
        mock_scheduler.has_requests.return_value = True
        mock_scheduler.schedule.return_value = so

        mock_executor = MagicMock()
        future = MagicMock()
        future.result.return_value = MagicMock()
        mock_executor.execute_model.return_value = future

        mock_grammar = MagicMock()
        mock_scheduler.get_grammar_bitmask.return_value = mock_grammar

        engine = EngineCore.__new__(EngineCore)
        engine.scheduler = mock_scheduler
        engine.model_executor = mock_executor
        engine.async_scheduling = False
        engine.use_spec_decode = False
        engine._process_aborts_queue = MagicMock()
        engine.log_error_detail = MagicMock()
        engine.log_iteration_details = MagicMock()

        with (
            engine.log_error_detail(),
            engine.log_iteration_details(),
        ):
            outputs, model_executed = engine.step()

        assert outputs == {}
        assert model_executed
        mock_scheduler.update_from_output.assert_called_once()
        mock_scheduler.push_batch_last.assert_called_once_with(so)

    def test_batch_last_normal_processing(self):
        """LAST batch should call update_from_output normally."""
        mock_scheduler = MagicMock()
        so = SchedulerOutput.make_empty()
        so.batch_type = BatchType.LAST
        so.total_num_scheduled_tokens = 10
        mock_scheduler.has_requests.return_value = True
        mock_scheduler.schedule.return_value = so

        mock_executor = MagicMock()
        future = MagicMock()
        future.result.return_value = MagicMock()
        mock_executor.execute_model.return_value = future

        mock_grammar = MagicMock()
        mock_scheduler.get_grammar_bitmask.return_value = mock_grammar

        mock_outputs = {0: MagicMock()}
        mock_scheduler.update_from_output.return_value = mock_outputs

        engine = EngineCore.__new__(EngineCore)
        engine.scheduler = mock_scheduler
        engine.model_executor = mock_executor
        engine.async_scheduling = False
        engine.use_spec_decode = False
        engine._process_aborts_queue = MagicMock()
        engine.log_error_detail = MagicMock()
        engine.log_iteration_details = MagicMock()

        with (
            engine.log_error_detail(),
            engine.log_iteration_details(),
        ):
            outputs, model_executed = engine.step()

        assert outputs == mock_outputs
        assert model_executed
        mock_scheduler.push_batch_last.assert_not_called()

    def test_full_batch_normal_processing(self):
        """FULL batch should behave normally (not edge-cloud mode)."""
        mock_scheduler = MagicMock()
        so = SchedulerOutput.make_empty()
        so.batch_type = BatchType.FULL
        so.total_num_scheduled_tokens = 10
        mock_scheduler.has_requests.return_value = True
        mock_scheduler.schedule.return_value = so

        mock_executor = MagicMock()
        future = MagicMock()
        future.result.return_value = MagicMock()
        mock_executor.execute_model.return_value = future

        mock_grammar = MagicMock()
        mock_scheduler.get_grammar_bitmask.return_value = mock_grammar

        mock_outputs = {0: MagicMock()}
        mock_scheduler.update_from_output.return_value = mock_outputs

        engine = EngineCore.__new__(EngineCore)
        engine.scheduler = mock_scheduler
        engine.model_executor = mock_executor
        engine.async_scheduling = False
        engine.use_spec_decode = False
        engine._process_aborts_queue = MagicMock()
        engine.log_error_detail = MagicMock()
        engine.log_iteration_details = MagicMock()

        with (
            engine.log_error_detail(),
            engine.log_iteration_details(),
        ):
            outputs, model_executed = engine.step()

        assert outputs == mock_outputs
        assert model_executed
        mock_scheduler.push_batch_last.assert_not_called()
