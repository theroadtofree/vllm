# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for edge-cloud async scheduling request status."""

from vllm.v1.request import RequestStatus


def test_head_done_status_exists():
    """Test that HEAD_DONE status is defined."""
    assert hasattr(RequestStatus, "HEAD_DONE")
    assert RequestStatus.HEAD_DONE.name == "HEAD_DONE"


def test_is_finished_excludes_head_done():
    """Test that HEAD_DONE is NOT considered a finished status."""
    assert not RequestStatus.is_finished(RequestStatus.HEAD_DONE)


def test_is_finished_for_actual_finished_statuses():
    """Test that actual finished statuses are still recognized."""
    assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED)
    assert RequestStatus.is_finished(RequestStatus.FINISHED_IGNORED)


def test_is_finished_for_running_statuses():
    """Test that non-finished statuses are not considered finished."""
    assert not RequestStatus.is_finished(RequestStatus.WAITING)
    assert not RequestStatus.is_finished(RequestStatus.RUNNING)
    assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)
    assert not RequestStatus.is_finished(RequestStatus.HEAD_DONE)


def test_head_done_ordering():
    """Test HEAD_DONE ordering: it should be between PREEMPTED and FINISHED_*.

    This ensures the is_finished() logic (status > PREEMPTED) does NOT
    include HEAD_DONE.
    """
    assert RequestStatus.PREEMPTED < RequestStatus.HEAD_DONE
    assert RequestStatus.HEAD_DONE < RequestStatus.FINISHED_STOPPED


def test_request_status_str():
    """Test string representation includes HEAD_DONE."""
    assert str(RequestStatus.HEAD_DONE) == "HEAD_DONE"
