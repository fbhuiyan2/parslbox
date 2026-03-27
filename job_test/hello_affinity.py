import os
import socket
import time
from parslbox.apps.utils import report_status

time.sleep(5)  # Simulate some work being done

hostname = socket.gethostname()

# Detect GPUs
gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
if gpu_id is None:
    gpu_id = os.environ.get("ZE_AFFINITY_MASK", "No GPUs assigned")

# Detect CPU affinity (Linux provides sched_getaffinity)
try:
    cpu_affinity = os.sched_getaffinity(0)  # CPUs this process is allowed to run on
    ncpu = len(cpu_affinity)
    print(f"Detected CPU affinity using sched_getaffinity: {cpu_affinity}")
except AttributeError:
    # Fallback if sched_getaffinity is not available (e.g., macOS)
    cpu_affinity = set(range(os.cpu_count()))
    ncpu = len(cpu_affinity)
    print(f"Fallback CPU affinity using os.cpu_count: {cpu_affinity}")

cpu_ids = sorted(cpu_affinity)

print(f"\nHello from host {hostname}:")
print(f"  GPU ID(s): {gpu_id}")
print(f"  CPU affinity: {ncpu} CPUs available to this process")
print(f"  CPU IDs: {cpu_ids}\n")

report_status("done")
