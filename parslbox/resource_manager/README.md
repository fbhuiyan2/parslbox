# ParslBox Resource Manager

A comprehensive resource management system for ParslBox that handles allocation and tracking of nodes, CPUs, and GPUs across different HPC systems.

## Overview

The ParslBox Resource Manager provides intelligent resource allocation for jobs running on HPC systems like Polaris and Sophia. It supports:

- **Single-node GPU jobs** with automatic GPU assignment and sharing
- **Single-node CPU jobs** with fractional node occupancy
- **Multi-node MPI jobs** with exclusive node allocation
- **Resource backlog and queuing** when resources are unavailable
- **Automatic resource cleanup** when jobs complete

## Architecture

### Core Components

- **`models.py`** - Data models for nodes, job specs, and assignments
- **`manager.py`** - Main resource manager class
- **`exceptions.py`** - Custom exceptions for resource management
- **`utils.py`** - Utility functions for MPI commands and resource formatting

### Key Classes

#### `NodeResource`
Represents a compute node with:
- GPU tracking (individual GPU IDs)
- CPU occupancy (fractional usage)
- Job assignments

#### `JobResourceSpec`
Defines job resource requirements:
- Number of nodes
- GPUs per node
- Node occupancy (for CPU-only jobs)

#### `NodeAssignment`
Result of resource allocation:
- Assigned nodes and hostnames
- GPU assignments per node
- Environment variables for job execution

#### `ParslboxResourceManager`
Main manager class that:
- Initializes nodes from system configuration
- Assigns resources based on job requirements
- Handles resource cleanup and backlog scheduling

## Usage Examples

### Basic Resource Manager Setup

```python
from parslbox.configs.polaris import PolarisConfig
from parslbox.resource_manager import JobResourceSpec

# Create system config and resource manager
config = PolarisConfig()
resource_manager = config.create_resource_manager()
```

### Single-Node GPU Job

```python
# Job requiring 2 GPUs on 1 node
spec = JobResourceSpec(job_id=1, num_nodes=1, gpus_per_node=2)
assignment = resource_manager.assign_resources(spec)

print(f"Assigned to: {assignment.hostnames[0]}")
print(f"GPUs: {assignment.gpu_assignments[0]}")
print(f"Environment: {assignment.get_env_vars()}")
# Output: {'CUDA_VISIBLE_DEVICES': '0,1', 'ZE_AFFINITY_MASK': '0,1'}
```

### Single-Node CPU Job

```python
# Job using 25% of a node's CPU capacity
spec = JobResourceSpec(job_id=2, num_nodes=1, gpus_per_node=0, node_occupancy=0.25)
assignment = resource_manager.assign_resources(spec)
```

### Multi-Node Job

```python
# Job requiring 4 nodes
spec = JobResourceSpec(job_id=3, num_nodes=4, gpus_per_node=0)
assignment = resource_manager.assign_resources(spec)

print(f"Hostlist: {assignment.get_mpi_hostlist()}")
# Output: "node-01,node-02,node-03,node-04"
```

### MPI Command Generation

```python
from parslbox.resource_manager.utils import build_job_command

# Generate complete job command
command = build_job_command(
    assignment=assignment,
    app_config={"ranks_per_node": 2},
    executable="lmp",
    args="-k on g 2 -sf kk -in input.lammps",
    mpi_opts="-x OMP_NUM_THREADS=1"
)

print(command)
# Output: "export CUDA_VISIBLE_DEVICES=0,1 && mpirun -n 2 -host node-01 -x OMP_NUM_THREADS=1 lmp -k on g 2 -sf kk -in input.lammps"
```

### Resource Status Monitoring

```python
status = resource_manager.get_resource_status()
print(f"Active jobs: {status['active_jobs']}")
print(f"Available GPUs: {status['available_gpus']}")
print(f"Free nodes: {status['free_nodes']}")

# Get detailed summary
from parslbox.resource_manager.utils import format_resource_summary
summary = format_resource_summary(status)
print(summary)
```

### Resource Cleanup

```python
# When job completes
resource_manager.free_resources(job_id=1)
# Automatically tries to schedule backlogged jobs
```

## Resource Allocation Logic

### Single-Node Jobs

**GPU Jobs:**
- Find node with sufficient available GPUs
- Assign specific GPU IDs (e.g., [0,1] for 2 GPUs)
- Multiple jobs can share a node if GPU count allows
- Set `CUDA_VISIBLE_DEVICES` for GPU isolation

**CPU-Only Jobs:**
- Use `node_occupancy` to specify fractional node usage
- Multiple jobs can share a node until occupancy reaches 1.0
- Example: 4 jobs with 0.25 occupancy each can share one node

### Multi-Node Jobs

- Require completely free nodes (no sharing)
- Take exclusive control of assigned nodes
- Generate hostlist for MPI execution

### Resource Sharing Examples

**Polaris Node (4 GPUs):**
```
Job 1: 2 GPUs → Gets GPUs [0,1]
Job 2: 1 GPU  → Gets GPU [2]  
Job 3: 1 GPU  → Gets GPU [3]
Job 4: 1 GPU  → Waits (no GPUs available)
```

**CPU Sharing:**
```
Job 1: 0.5 occupancy → Node occupancy = 0.5
Job 2: 0.3 occupancy → Node occupancy = 0.8
Job 3: 0.2 occupancy → Node occupancy = 1.0 (full)
Job 4: 0.1 occupancy → Waits (insufficient capacity)
```

## Integration with ParslBox

The resource manager integrates seamlessly with ParslBox's existing components:

1. **System Configs** - Each system config can create a resource manager
2. **Job Database** - Resource specs can be stored alongside job metadata
3. **App Execution** - Apps use resource assignments to generate proper commands
4. **Scheduler Integration** - Automatically detects nodes from PBS/SLURM

## Testing

Run the test suite to verify functionality:

```bash
cd tests
python3 test_resource_manager.py
```

The test suite covers:
- Basic resource manager initialization
- Single-node GPU job sharing
- CPU-only job occupancy
- Multi-node job allocation
- MPI command generation
- Resource cleanup and backlog scheduling
- Error handling for insufficient resources

## Future Enhancements

- **Per-job resource tracking** - More sophisticated resource freeing
- **Dynamic resource rebalancing** - Optimize resource utilization
- **Resource usage monitoring** - Track actual vs. allocated resources
- **Priority-based scheduling** - Advanced job prioritization
- **Resource reservations** - Pre-allocate resources for future jobs
