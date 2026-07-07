"""
Node Failure Tracker

Handles node health tracking and fault tolerance for the ParslBox resource manager.
Implements quarantine logic to prevent cascading failures from problematic nodes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import logging
import time
import re

logger = logging.getLogger(__name__)


class NodeHealth(Enum):
    """Node health status for fault tolerance."""
    HEALTHY = "healthy"
    SUSPECTED = "suspected"
    QUARANTINED = "quarantined"


class ErrorType(Enum):
    """Classification of job errors for quarantine decisions."""
    TRANSIENT = "transient"      # Temporary issues, allow immediate reuse
    PERSISTENT = "persistent"    # Node-level issues, quarantine node
    JOB_SPECIFIC = "job_specific"  # Input/config issues, allow immediate reuse


@dataclass
class FailureOutcome:
    """Result of recording one job failure on a node — returned to the caller
    (ResourceManager) so it can emit a single aggregated log line per job
    failure instead of one per node."""
    counted: bool                 # did this failure count against node health?
    error_type: ErrorType
    old_status: NodeHealth
    new_status: NodeHealth
    distinct_failures: int        # distinct failing jobs charged to this node


@dataclass
class NodeHealthTracker:
    """Tracks node health and failure patterns for fault tolerance.

    Escalation is driven by the number of **distinct jobs** that have failed on
    this node with a node-suspicious error (not by raw failure events). A single
    job — even one spanning every node in the allocation, and even if retried —
    contributes at most 1 to every node it touched, so it can never by itself
    quarantine a node. Nodes are only suspected/quarantined when several
    *different* jobs die on them, which is the real signature of a bad node
    rather than a bad app/input.
    """
    total_failures: int = 0
    last_failure_time: Optional[float] = None
    last_failure_error: Optional[str] = None
    quarantine_start_time: Optional[float] = None
    health_status: NodeHealth = NodeHealth.HEALTHY

    # Distinct job ids that have failed on this node with a node-suspicious
    # error since the last success/recovery. Its size is the escalation metric.
    failed_job_ids: set = field(default_factory=set)

    # Configuration (can be overridden by system config). `max_consecutive_failures`
    # now means "distinct failing jobs before quarantine" (name kept for config
    # compatibility).
    max_consecutive_failures: int = 4
    quarantine_duration: int = 600  # 10 minutes
    
    # Error patterns that indicate persistent node issues
    persistent_error_patterns: List[str] = field(default_factory=lambda: [
        r"unix exit code 127",  # Command not found
        r"command not found",
        r"No such file or directory.*executable",
        r"Permission denied.*executable",
        r"Permission denied.*vasp",  # VASP-specific permission issues
        r"Permission denied.*lammps",  # LAMMPS-specific permission issues
        r"cannot execute binary file",
        r"Exec format error",
        r"Text file busy",
        r"Input/output error",
        r"Device or resource busy",
        r"No space left on device",
        # MPI rank/node failure patterns
        r"rank \d+ died from signal",
        r"rank \d+ exit\w* with (?:signal|code)",
        r"mpirun.*detected.*terminated",
        r"mpirun.*has exited with a non-zero exit code",
        r"MPI_ABORT",
        r"ORTE_ERROR_LOG",
        r"srun.*error.*task.*exited with",
        r"srun.*error.*Node failure",
        r"PMI_FAIL",
        # System-level node failure patterns
        r"Segmentation fault",
        r"Bus error",
        r"Killed",
        r"Out of memory",
        r"OOM",
        r"Cannot allocate memory",
        r"CUDA.*error",
        r"GPU.*error",
        r"Xid.*error",
    ])
    
    def classify_error(self, error_message: str) -> ErrorType:
        """
        Classify an error to determine if it indicates a persistent node issue.
        
        Args:
            error_message: Error message from job failure
            
        Returns:
            ErrorType indicating how to handle the error
        """
        if not error_message:
            return ErrorType.JOB_SPECIFIC  # No error info = no evidence of node issue
        
        # Check for persistent error patterns
        for pattern in self.persistent_error_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                logger.debug(f"Classified error as PERSISTENT (matched pattern)")
                return ErrorType.PERSISTENT
        
        # Check for job-specific patterns (input file issues, etc.)
        job_specific_patterns = [
            r"No such file or directory.*input",
            r"cannot open.*input",
            r"Invalid input",
            r"Parse error",
            r"Configuration error"
        ]
        
        for pattern in job_specific_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                logger.debug(f"Classified error as JOB_SPECIFIC (matched pattern)")
                return ErrorType.JOB_SPECIFIC
        
        # Default to job-specific for unrecognized errors — don't blame the node
        # unless we see a known node-level error pattern
        logger.debug(f"Classified error as JOB_SPECIFIC (unrecognized): {error_message[:200]}")
        return ErrorType.JOB_SPECIFIC
    
    def record_failure(self, job_id=None, error_message: str = None) -> "FailureOutcome":
        """
        Record a job failure on this node.

        Escalation counts DISTINCT failing jobs, so a job passed more than once
        (multi-node accounting, retry, restart) is charged at most once. This
        method does no logging — it returns a `FailureOutcome` so the caller can
        emit a single aggregated line per job failure.

        Args:
            job_id: ID of the failed job (used to de-duplicate distinct failures).
            error_message: Error message from the failed job.

        Returns:
            FailureOutcome describing the classification and any status change.
        """
        current_time = time.time()
        error_type = self.classify_error(error_message)

        self.total_failures += 1
        self.last_failure_time = current_time
        self.last_failure_error = error_message[:200] if error_message else error_message

        old_status = self.health_status

        # Job-specific / no-info errors never count against the node.
        if error_type == ErrorType.JOB_SPECIFIC:
            return FailureOutcome(
                counted=False, error_type=error_type, old_status=old_status,
                new_status=self.health_status, distinct_failures=len(self.failed_job_ids),
            )

        # Node-suspicious error: charge this node once per distinct job.
        before = len(self.failed_job_ids)
        self.failed_job_ids.add(job_id)
        distinct = len(self.failed_job_ids)
        counted = distinct > before  # False if this job already failed here

        if distinct >= self.max_consecutive_failures:
            self.health_status = NodeHealth.QUARANTINED
            self.quarantine_start_time = current_time
        elif distinct >= max(2, self.max_consecutive_failures // 2):
            # Require at least 2 distinct jobs — one job must never suspect a node.
            if self.health_status == NodeHealth.HEALTHY:
                self.health_status = NodeHealth.SUSPECTED

        return FailureOutcome(
            counted=counted, error_type=error_type, old_status=old_status,
            new_status=self.health_status, distinct_failures=distinct,
        )

    def record_success(self) -> None:
        """Record a successful job completion — clears accumulated failure
        evidence (the node just proved it can run work)."""
        if self.health_status == NodeHealth.SUSPECTED:
            self.health_status = NodeHealth.HEALTHY
            self.failed_job_ids.clear()
            logger.info("Node health restored to HEALTHY after successful job")
        elif self.health_status == NodeHealth.HEALTHY:
            self.failed_job_ids.clear()
    
    def can_accept_jobs(self) -> bool:
        """Check if node can accept new jobs based on health status."""
        if self.health_status == NodeHealth.QUARANTINED:
            return self.is_quarantine_expired()
        return True
    
    def is_quarantine_expired(self) -> bool:
        """Check if quarantine period has expired."""
        if not self.quarantine_start_time:
            return True
        
        return (time.time() - self.quarantine_start_time) >= self.quarantine_duration
    
    def attempt_recovery(self) -> bool:
        """
        Attempt to recover a quarantined node.
        
        Returns:
            True if node was recovered, False otherwise
        """
        if self.health_status != NodeHealth.QUARANTINED:
            return False
        
        if self.is_quarantine_expired():
            self.health_status = NodeHealth.SUSPECTED
            # Keep the node on probation: retain half the distinct-failure
            # evidence so a still-bad node re-quarantines quickly.
            keep = len(self.failed_job_ids) // 2
            self.failed_job_ids = set(list(self.failed_job_ids)[:keep])
            self.quarantine_start_time = None
            logger.info(f"Node recovered from quarantine, status: {self.health_status.value}")
            return True

        return False

    def reset_health(self) -> None:
        """Clear all failure evidence and restore the node to HEALTHY."""
        self.failed_job_ids.clear()
        self.health_status = NodeHealth.HEALTHY
        self.quarantine_start_time = None

    def get_status_summary(self) -> Dict:
        """Get a summary of node health status."""
        return {
            'health_status': self.health_status.value,
            # Distinct failing jobs charged to this node (drives escalation). Key
            # name kept as 'consecutive_failures' for backward compatibility.
            'consecutive_failures': len(self.failed_job_ids),
            'distinct_failed_jobs': len(self.failed_job_ids),
            'total_failures': self.total_failures,
            'last_failure_time': self.last_failure_time,
            'last_failure_error': self.last_failure_error,
            'quarantine_start_time': self.quarantine_start_time,
            'can_accept_jobs': self.can_accept_jobs(),
            'quarantine_expired': self.is_quarantine_expired() if self.health_status == NodeHealth.QUARANTINED else None
        }


class NodeFailureTracker:
    """
    Manages failure tracking for all nodes in the system.
    
    Provides centralized node health management and quarantine coordination.
    """
    
    def __init__(self, max_consecutive_failures: int = 4, quarantine_duration: int = 300):
        """
        Initialize the failure tracker.
        
        Args:
            max_consecutive_failures: Number of failures before quarantine
            quarantine_duration: Quarantine duration in seconds
        """
        self.node_health: Dict[str, NodeHealthTracker] = {}
        self.max_consecutive_failures = max_consecutive_failures
        self.quarantine_duration = quarantine_duration
        
        logger.info(f"Initialized NodeFailureTracker with max_failures={max_consecutive_failures}, "
                   f"quarantine_duration={quarantine_duration}s")
    
    def get_or_create_tracker(self, node_id: str) -> NodeHealthTracker:
        """Get or create a health tracker for a node."""
        if node_id not in self.node_health:
            self.node_health[node_id] = NodeHealthTracker(
                max_consecutive_failures=self.max_consecutive_failures,
                quarantine_duration=self.quarantine_duration
            )
        return self.node_health[node_id]
    
    def record_job_failure(self, node_id: str, job_id: int, error_message: str = None) -> bool:
        """
        Record a job failure on a specific node.
        
        Args:
            node_id: ID of the node where job failed
            job_id: ID of the failed job
            error_message: Error message from the failure
            
        Returns:
            True if node should be quarantined, False otherwise
        """
        tracker = self.get_or_create_tracker(node_id)
        outcome = tracker.record_failure(job_id, error_message)
        should_quarantine = outcome.new_status == NodeHealth.QUARANTINED
        logger.debug(
            f"Centralized tracker: job {job_id} on node {node_id} → "
            f"{tracker.health_status.value} ({outcome.distinct_failures} distinct)"
        )
        return should_quarantine
    
    def record_job_success(self, node_id: str, job_id: int) -> None:
        """
        Record a successful job completion on a specific node.
        
        Args:
            node_id: ID of the node where job succeeded
            job_id: ID of the successful job
        """
        tracker = self.get_or_create_tracker(node_id)
        old_status = tracker.health_status
        tracker.record_success()
        
        if old_status != tracker.health_status:
            logger.info(f"Node {node_id} health improved from {old_status.value} to {tracker.health_status.value} "
                       f"after job {job_id} success")
    
    def can_node_accept_jobs(self, node_id: str) -> bool:
        """
        Check if a node can accept new jobs based on its health status.
        
        Args:
            node_id: ID of the node to check
            
        Returns:
            True if node can accept jobs, False if quarantined
        """
        if node_id not in self.node_health:
            return True  # New nodes are healthy by default
        
        tracker = self.node_health[node_id]
        can_accept = tracker.can_accept_jobs()
        
        # Attempt recovery if quarantine has expired
        if not can_accept and tracker.health_status == NodeHealth.QUARANTINED:
            if tracker.attempt_recovery():
                logger.info(f"Node {node_id} recovered from quarantine")
                can_accept = True
        
        return can_accept
    
    def get_healthy_nodes(self, node_ids: List[str]) -> List[str]:
        """
        Filter a list of nodes to return only healthy ones.
        
        Args:
            node_ids: List of node IDs to filter
            
        Returns:
            List of node IDs that can accept jobs
        """
        healthy_nodes = []
        for node_id in node_ids:
            if self.can_node_accept_jobs(node_id):
                healthy_nodes.append(node_id)
        
        return healthy_nodes
    
    def get_quarantined_nodes(self) -> List[str]:
        """Get list of currently quarantined node IDs."""
        quarantined = []
        for node_id, tracker in self.node_health.items():
            if tracker.health_status == NodeHealth.QUARANTINED and not tracker.is_quarantine_expired():
                quarantined.append(node_id)
        
        return quarantined
    
    def get_system_health_summary(self) -> Dict:
        """Get overall system health summary."""
        total_nodes = len(self.node_health)
        healthy_count = 0
        suspected_count = 0
        quarantined_count = 0
        
        for tracker in self.node_health.values():
            if tracker.health_status == NodeHealth.HEALTHY:
                healthy_count += 1
            elif tracker.health_status == NodeHealth.SUSPECTED:
                suspected_count += 1
            elif tracker.health_status == NodeHealth.QUARANTINED:
                if not tracker.is_quarantine_expired():
                    quarantined_count += 1
                else:
                    # Expired quarantine counts as suspected
                    suspected_count += 1
        
        return {
            'total_tracked_nodes': total_nodes,
            'healthy_nodes': healthy_count,
            'suspected_nodes': suspected_count,
            'quarantined_nodes': quarantined_count,
            'quarantined_node_ids': self.get_quarantined_nodes()
        }
    
    def get_node_health_details(self) -> Dict[str, Dict]:
        """Get detailed health information for all tracked nodes."""
        details = {}
        for node_id, tracker in self.node_health.items():
            details[node_id] = tracker.get_status_summary()
        
        return details
    
    def force_quarantine_node(self, node_id: str, reason: str = "Manual quarantine") -> None:
        """
        Manually quarantine a node.
        
        Args:
            node_id: ID of the node to quarantine
            reason: Reason for manual quarantine
        """
        tracker = self.get_or_create_tracker(node_id)
        tracker.health_status = NodeHealth.QUARANTINED
        tracker.quarantine_start_time = time.time()
        tracker.last_failure_error = reason
        
        logger.warning(f"Node {node_id} manually quarantined: {reason}")
    
    def force_recover_node(self, node_id: str) -> bool:
        """
        Manually recover a quarantined node.
        
        Args:
            node_id: ID of the node to recover
            
        Returns:
            True if node was recovered, False if not quarantined
        """
        if node_id not in self.node_health:
            return False
        
        tracker = self.node_health[node_id]
        if tracker.health_status == NodeHealth.QUARANTINED:
            tracker.reset_health()
            logger.info(f"Node {node_id} manually recovered from quarantine")
            return True

        return False
