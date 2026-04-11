"""
Spawner job script for testing dynamic job discovery.

This script:
1. Prints host/GPU/CPU affinity info (same as hello_affinity.py)
2. Dynamically adds 5 hello_affinity child jobs to the DB via ParslBox API

The child jobs will only be picked up by `pbx run` if --dynamic mode is enabled.
Without --dynamic, the run session exits after this spawner completes and the
children are never executed.

Usage:
    pbx add --app python --input spawner.py --config polaris --envfile spawner_env.sh
"""

import os
import sys
import socket
import time
from pathlib import Path

from parslbox.apps.utils import report_status

# --- Step 1: Print affinity info (same as hello_affinity.py) ---

time.sleep(3)  # Simulate some work

hostname = socket.gethostname()

gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
if gpu_id is None:
    gpu_id = os.environ.get("ZE_AFFINITY_MASK", "No GPUs assigned")

try:
    cpu_affinity = os.sched_getaffinity(0)
    ncpu = len(cpu_affinity)
except AttributeError:
    cpu_affinity = set(range(os.cpu_count()))
    ncpu = len(cpu_affinity)

cpu_ids = sorted(cpu_affinity)

print(f"\n[Spawner] Hello from host {hostname}:")
print(f"  GPU ID(s): {gpu_id}")
print(f"  CPU affinity: {ncpu} CPUs available to this process")
print(f"  CPU IDs: {cpu_ids}")

# --- Step 2: Dynamically add 5 hello_affinity child jobs ---

# The hello_affinity.py and env file should be in the same directory as this script
script_dir = Path(__file__).parent
hello_script = "hello_affinity.py"
env_file = "spawner_env.sh"

# Read config name from PBX environment (set by the resource launcher)
config_name = os.environ.get("PBX_CONFIG_NAME")
if not config_name:
    print("[Spawner] WARNING: PBX_CONFIG_NAME not set, cannot spawn child jobs")
    report_status("done")
    sys.exit(0)

try:
    from parslbox.api import ParslBox
    pbx = ParslBox()

    num_children = 5
    print(f"\n[Spawner] Adding {num_children} hello_affinity child jobs...")

    for i in range(1, num_children + 1):
        child_dir = Path.cwd() / f"child_{i}"
        child_dir.mkdir(exist_ok=True)

        # Copy hello_affinity.py to child dir
        import shutil
        shutil.copy2(script_dir / hello_script, child_dir / hello_script)

        # Copy env file if it exists
        env_file_path = script_dir / env_file
        if env_file_path.exists():
            shutil.copy2(env_file_path, child_dir / env_file)

        # Add job via API
        job_ids, failed, msg_log = pbx.add_jobs(
            paths=[str(child_dir)],
            app="python",
            config=config_name,
            input_file=hello_script,
            env_file=str(child_dir / env_file) if env_file_path.exists() else None,
            tag="dynamic_test",
            ngpus=1,
        )

        if job_ids:
            print(f"  Added child_{i}: Job ID {job_ids[0]}")
        elif failed:
            for path, error in failed:
                print(f"  Failed child_{i}: {error}")

    print(f"[Spawner] Done adding {num_children} child jobs.")

except Exception as e:
    print(f"[Spawner] ERROR adding child jobs: {e}")
    import traceback
    traceback.print_exc()

report_status("done")
