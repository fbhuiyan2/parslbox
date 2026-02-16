# Devlog

**Current Version: 0.7.1**

## Purpose
This file tracks bug fixes and development issues encountered during ParslBox development. 
It serves as a quick reference for resolved problems and their solutions, separate from the formal CHANGELOG.md which documents version releases and feature changes.

---

## Bug Fixes

[2025-02-15]
- **Issue**: "Text file busy" error when rerunning jobs after updating status to ready - wrapper scripts couldn't be overwritten.
  - **Fix**: Added file deletion with try-except handling before writing new wrapper scripts and rankfiles in mpi_launcher_helpers.py.

[2025-02-15]
- **Issue**: GPU count not automatically updated when changing number of nodes via `pbx update -n` flag for GPU jobs.
  - **Fix**: Added auto-scaling logic in update.py to calculate and update GPU count based on GPUs-per-node ratio. Multi-node GPU jobs auto-scale; single-node→multi-node requires explicit -g flag with error message showing formula (total_gpus = nnodes * gpus_per_node).

[2025-02-15]
- **Issue**: Jobs left in "Running" state when batch job runs out of walltime on Aurora - SIGTERM handler not properly updating job statuses.
  - **Fix**: Enhanced signal handler in run_cmd_helpers.py with: (1) 10-second timeout protection via signal.alarm() to prevent SIGKILL, (2) detailed timestamped logging at each shutdown step, (3) separate try-except blocks for status buffer flush (critical) vs Parsl cleanup (secondary), ensuring status updates always complete even if Parsl cleanup fails. Added SIGALRM handler in run.py to force exit if shutdown exceeds timeout.
