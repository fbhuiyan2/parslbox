"""
Tests for node health / fault-tolerance escalation.

Two properties under test:
  (a) A single job failing — even a multi-node job spanning every node, even
      retried — never quarantines (or suspects) a node. Escalation counts
      DISTINCT failing jobs per node.
  (b) A node is only suspected/quarantined once several *different* jobs fail on
      it (the real signature of a bad node, vs. a bad app/input).

Unit tests exercise NodeHealthTracker directly; integration tests drive a real
ResourceManager for the reported multi-node-job scenario.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from parslbox.resource_manager import ResourceManager
from parslbox.resource_manager.node_failure_tracker import (
    NodeHealthTracker,
    NodeHealth,
    ErrorType,
)
from test_comprehensive_resource_manager import MockSystemConfig


NODE_ERR = "Segmentation fault"          # node-suspicious pattern
JOB_ERR = "Invalid input"                # job-specific pattern


# --------------------------------------------------------------------------- #
# NodeHealthTracker — distinct-job escalation
# --------------------------------------------------------------------------- #
class TestNodeHealthTracker:
    def test_single_job_never_escalates(self):
        t = NodeHealthTracker()
        out = t.record_failure(job_id=51, error_message=NODE_ERR)
        assert out.counted is True
        assert out.error_type == ErrorType.PERSISTENT
        assert out.distinct_failures == 1
        assert t.health_status == NodeHealth.HEALTHY

    def test_same_job_is_deduplicated(self):
        """The same job charged repeatedly (multi-node accounting / retry /
        restart) counts once and never escalates."""
        t = NodeHealthTracker()
        for _ in range(6):
            out = t.record_failure(job_id=51, error_message=NODE_ERR)
        assert t.failed_job_ids == {51}
        assert out.distinct_failures == 1
        assert out.counted is False   # already counted after the first
        assert t.health_status == NodeHealth.HEALTHY

    def test_two_distinct_jobs_suspected(self):
        t = NodeHealthTracker()             # max_consecutive_failures=4 → suspect at 2
        t.record_failure(job_id=1, error_message=NODE_ERR)
        assert t.health_status == NodeHealth.HEALTHY
        out = t.record_failure(job_id=2, error_message=NODE_ERR)
        assert out.distinct_failures == 2
        assert t.health_status == NodeHealth.SUSPECTED

    def test_four_distinct_jobs_quarantined(self):
        t = NodeHealthTracker()
        for jid in (1, 2, 3):
            t.record_failure(job_id=jid, error_message=NODE_ERR)
        assert t.health_status == NodeHealth.SUSPECTED
        out = t.record_failure(job_id=4, error_message=NODE_ERR)
        assert out.distinct_failures == 4
        assert t.health_status == NodeHealth.QUARANTINED
        assert t.can_accept_jobs() is False

    def test_job_specific_never_counts(self):
        t = NodeHealthTracker()
        for jid in range(10):
            out = t.record_failure(job_id=jid, error_message=JOB_ERR)
            assert out.counted is False
        assert t.failed_job_ids == set()
        assert t.health_status == NodeHealth.HEALTHY

    def test_no_error_message_is_job_specific(self):
        t = NodeHealthTracker()
        out = t.record_failure(job_id=1, error_message=None)
        assert out.counted is False
        assert out.error_type == ErrorType.JOB_SPECIFIC
        assert t.health_status == NodeHealth.HEALTHY

    def test_segfault_still_node_suspicious(self):
        """We did NOT reclassify segfault to job-specific — it still counts as a
        distinct failure; it just can't quarantine on its own."""
        t = NodeHealthTracker()
        out = t.record_failure(job_id=1, error_message="Bus error")
        assert out.error_type == ErrorType.PERSISTENT
        assert out.counted is True

    def test_success_clears_failure_evidence(self):
        t = NodeHealthTracker()
        t.record_failure(job_id=1, error_message=NODE_ERR)
        t.record_failure(job_id=2, error_message=NODE_ERR)
        assert t.health_status == NodeHealth.SUSPECTED
        t.record_success()
        assert t.health_status == NodeHealth.HEALTHY
        assert t.failed_job_ids == set()

    def test_outcome_reports_transition(self):
        t = NodeHealthTracker()
        t.record_failure(job_id=1, error_message=NODE_ERR)
        out = t.record_failure(job_id=2, error_message=NODE_ERR)
        assert out.old_status == NodeHealth.HEALTHY
        assert out.new_status == NodeHealth.SUSPECTED

    def test_reset_health(self):
        t = NodeHealthTracker()
        for jid in (1, 2, 3, 4):
            t.record_failure(job_id=jid, error_message=NODE_ERR)
        assert t.health_status == NodeHealth.QUARANTINED
        t.reset_health()
        assert t.health_status == NodeHealth.HEALTHY
        assert t.failed_job_ids == set()

    def test_summary_backward_compat_key(self):
        t = NodeHealthTracker()
        t.record_failure(job_id=1, error_message=NODE_ERR)
        summary = t.get_status_summary()
        assert summary['consecutive_failures'] == 1   # legacy key = distinct count
        assert summary['distinct_failed_jobs'] == 1


# --------------------------------------------------------------------------- #
# ResourceManager — the reported scenario
# --------------------------------------------------------------------------- #
class TestResourceManagerHealth:
    def test_multinode_single_job_failure_keeps_all_nodes_healthy(self):
        """A 4-node job that segfaults must NOT push any of its nodes toward
        quarantine — this is the reported '30 warnings for one failure' case."""
        rm = ResourceManager(MockSystemConfig(num_nodes=4, gpus_per_node=4),
                             job_tracker=None)
        job = {'job_id': 51, 'num_nodes': 4, 'ngpus': 4,
               'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        assert len(assignment.node_ids) == 4

        rm.record_job_failure(51, error_message=NODE_ERR)

        for node in rm.nodes:
            assert node.health_tracker.health_status == NodeHealth.HEALTHY
            assert node.health_tracker.can_accept_jobs() is True
            assert node.health_tracker.failed_job_ids == {51}

    def test_retried_job_never_quarantines_node(self):
        """Same job failing repeatedly on the same node (e.g. restart chain)
        stays at one distinct failure — node remains usable."""
        rm = ResourceManager(MockSystemConfig(num_nodes=1), job_tracker=None)
        for _ in range(5):
            job = {'job_id': 7, 'num_nodes': 1, 'ngpus': 0,
                   'node_occupancy': 1.0, 'ranks_per_node': 1}
            rm.assign_resources(job)
            rm.record_job_failure(7, error_message=NODE_ERR)
            rm.free_resources(7)
        node = rm.nodes[0]
        assert node.health_tracker.health_status == NodeHealth.HEALTHY
        assert node.health_tracker.failed_job_ids == {7}

    def test_distinct_jobs_on_same_node_quarantine(self):
        """Four *different* jobs dying on the same node quarantines it."""
        rm = ResourceManager(MockSystemConfig(num_nodes=1), job_tracker=None)
        for jid in (1, 2, 3, 4):
            job = {'job_id': jid, 'num_nodes': 1, 'ngpus': 0,
                   'node_occupancy': 1.0, 'ranks_per_node': 1}
            rm.assign_resources(job)
            rm.record_job_failure(jid, error_message=NODE_ERR)
            rm.free_resources(jid)
        node = rm.nodes[0]
        assert node.health_tracker.health_status == NodeHealth.QUARANTINED

    def test_job_specific_failures_never_quarantine(self):
        rm = ResourceManager(MockSystemConfig(num_nodes=1), job_tracker=None)
        for jid in (1, 2, 3, 4, 5):
            job = {'job_id': jid, 'num_nodes': 1, 'ngpus': 0,
                   'node_occupancy': 1.0, 'ranks_per_node': 1}
            rm.assign_resources(job)
            rm.record_job_failure(jid, error_message=JOB_ERR)
            rm.free_resources(jid)
        node = rm.nodes[0]
        assert node.health_tracker.health_status == NodeHealth.HEALTHY
