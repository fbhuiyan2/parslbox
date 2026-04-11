#!/usr/bin/env python3
"""
Test Orchestrator for Dynamic Job Discovery

Creates spawner jobs that each dynamically add hello_affinity child jobs
when they run. Tests the --dynamic flag in `pbx run`.

Expects spawner.py, hello_affinity.py, and spawner_env.sh in the same
directory as this script.

Usage:
    python test_dynamic_orchestrator.py <config_name> (--ngpus N | --nocc N) [options]

Arguments:
    config_name              System configuration name (e.g., polaris, sophia)

Required (mutually exclusive):
    --ngpus N                Number of GPUs per spawner job
    --nocc N                 Node occupancy fraction per spawner job (0.0-1.0)

Optional:
    --njobs N                Number of spawner jobs to create (default: 2)
    --tag TAG                Tag for all jobs (default: dynamic_test)

Examples:
    python test_dynamic_orchestrator.py polaris --njobs 2 --ngpus 1
    python test_dynamic_orchestrator.py sophia --njobs 3 --nocc 0.5
    python test_dynamic_orchestrator.py polaris --njobs 1 --ngpus 1 --tag my_test
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from parslbox.api import ParslBox


def main():
    parser = argparse.ArgumentParser(
        description="Create spawner jobs for testing dynamic job discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("config_name", help="System configuration name (e.g., polaris)")

    resource_group = parser.add_mutually_exclusive_group(required=True)
    resource_group.add_argument("--ngpus", type=int, metavar="N", help="GPUs per spawner job")
    resource_group.add_argument("--nocc", type=float, metavar="N", help="Node occupancy per spawner job (0.0-1.0)")

    parser.add_argument("--njobs", type=int, default=2, metavar="N", help="Number of spawner jobs (default: 2)")
    parser.add_argument("--tag", default="dynamic_test", metavar="TAG", help="Tag for all jobs (default: dynamic_test)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    required_files = ["spawner.py", "hello_affinity.py", "spawner_env.sh"]
    for f in required_files:
        if not (script_dir / f).exists():
            print(f"ERROR: Required file '{f}' not found in {script_dir}")
            sys.exit(1)

    pbx = ParslBox()
    test_dir = Path.cwd() / "tests_dynamic"
    test_dir.mkdir(exist_ok=True)

    print(f"Creating {args.njobs} spawner jobs for {args.config_name}...")
    print(f"  Each spawner will dynamically add 5 hello_affinity child jobs at runtime.")
    print(f"  Tag: {args.tag}")
    print()

    created_ids = []
    for i in range(1, args.njobs + 1):
        job_dir = test_dir / f"spawner_{i}"
        job_dir.mkdir(exist_ok=True)

        # Copy all required files into the job directory
        for f in required_files:
            shutil.copy2(script_dir / f, job_dir / f)

        # Build resource kwargs
        resource_kwargs = {}
        if args.ngpus:
            resource_kwargs["ngpus"] = args.ngpus
        else:
            resource_kwargs["node_occupancy"] = args.nocc

        env_file_path = str((job_dir / "spawner_env.sh").resolve())

        job_ids, failed, msg_log = pbx.add_jobs(
            paths=[str(job_dir)],
            app="python",
            config=args.config_name,
            input_file="spawner.py",
            env_file=env_file_path,
            tag=args.tag,
            **resource_kwargs,
        )

        for warning in msg_log["warnings"]:
            print(f"  WARNING: {warning}")

        if job_ids:
            created_ids.append(job_ids[0])
            print(f"  Added spawner_{i}: Job ID {job_ids[0]}")
        elif failed:
            for path, error in failed:
                print(f"  Failed spawner_{i}: {error}")

    print(f"\nCreated {len(created_ids)} spawner jobs.")
    print(f"Test directory: {test_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Submit with: pbx qsub --dynamic ...  (or pbx sbatch --dynamic ...)")
    print("  2. Each spawner will add 5 hello_affinity jobs when it runs")
    print("  3. With --dynamic, the run session picks up the children automatically")
    print("  4. With --static, the children remain in 'Ready' status after the run")


if __name__ == "__main__":
    main()
