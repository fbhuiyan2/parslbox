"""
LAMMPS-Kokkos with restart-chain support.

Layered on top of LammpsKokkosApp. Adds:
  - restart(): generates per-link input from `in.restart_template.lammps`,
    rolls the prior `log.lammps` aside, picks the latest `restart.*` checkpoint.
  - postprocess(): concatenates `dump_link-N.lammpsdump` files (in numeric order)
    into the main `dump.lammpsdump`.

User responsibilities (per job dir):
  - in.lammps                       : link 0 input
  - in.restart_template.lammps      : template containing
      variable sim_total_steps equal <int>
      variable link equal <<PBX_LINK>>
      variable remaining_steps_2_run equal <<PBX_REMAINING_STEPS>>
      variable restart_file string <<PBX_RESTART_FILE>>
    plus the usual read_restart/dump/run commands using those variables, e.g.
      read_restart ${restart_file}
      dump 1 all custom N dump_link-${link}.lammpsdump ...
      run ${remaining_steps_2_run}

pbx_config.yaml registration:
  custom_apps:
    lammps-kk-restart:
      module: "parslbox/apps/custom_apps/lammps_kk_restart.py"
      class: "LammpsKkRestart"
  lammps-kk-restart:
    <system>:
      executable_path: "..."
      environment_setup: |
        ...
"""

import logging
import re
import shutil
from pathlib import Path

from parslbox.apps.lammps_kk import LammpsKokkosApp


_LINK_RE = re.compile(r"in\.restart_link-(\d+)\.lammps$")
_RESTART_STEP_RE = re.compile(r"\.(\d+)$")
_SIM_TOTAL_RE = re.compile(
    r"^\s*variable\s+sim_total_steps\s+equal\s+(\d+)", re.MULTILINE
)
_DUMP_LINK_RE = re.compile(r"dump_link-(\d+)\.lammpsdump$")


class LammpsKkRestart(LammpsKokkosApp):
    """LAMMPS-Kokkos with chain-restart support via in.restart_template.lammps."""

    INPUT_REQUIRED = True
    DFLT_INPUT = "in.lammps"
    RUN_HOOKS_ON_COMPUTE = True

    def min_remaining_walltime(self, job_dict: dict) -> int:
        # Fresh LAMMPS dispatches need a meaningful runtime window to write
        # at least one restart checkpoint and make useful progress; skip if
        # the batch has under an hour left. Restart-continuations are exempt
        # in the gating helper (they have checkpoint state already).
        return 3600

    def restart(self, job_dict: dict) -> dict | None:
        logger = logging.getLogger(__name__)
        job_id = job_dict["job_id"]
        job_path = Path(job_dict["path"])

        template = job_path / "in.restart_template.lammps"
        if not template.is_file():
            raise NotImplementedError(
                f"job {job_id}: in.restart_template.lammps not found in {job_path}"
            )

        existing = [
            (int(m.group(1)), p)
            for p in job_path.glob("in.restart_link-*.lammps")
            for m in [_LINK_RE.search(p.name)] if m
        ]
        new_link = (max(n for n, _ in existing) + 1) if existing else 1
        prev_link = new_link - 1

        log = job_path / "log.lammps"
        if log.is_file():
            target = job_path / f"log_link-{prev_link}.lammps"
            log.rename(target)
            logger.info(f"job {job_id}: renamed log.lammps -> {target.name}")
        else:
            logger.info(f"job {job_id}: no log.lammps to rename (link {prev_link})")

        restarts = [
            (int(m.group(1)), p)
            for p in job_path.glob("restart.*")
            for m in [_RESTART_STEP_RE.search(p.name)] if m
        ]
        if not restarts:
            raise NotImplementedError(
                f"job {job_id}: no restart.<step> files found in {job_path}"
            )
        latest_step, latest_file = max(restarts, key=lambda t: t[0])

        template_text = template.read_text()
        m = _SIM_TOTAL_RE.search(template_text)
        if not m:
            raise NotImplementedError(
                f"job {job_id}: 'variable sim_total_steps equal <int>' "
                f"not found in {template.name}"
            )
        sim_total_steps = int(m.group(1))

        remaining = sim_total_steps - latest_step
        if remaining < 0:
            remaining = 0
        logger.info(
            f"job {job_id}: link {new_link}, latest_step={latest_step}, "
            f"sim_total_steps={sim_total_steps}, remaining={remaining}"
        )

        content = (
            template_text
            .replace("<<PBX_LINK>>", str(new_link))
            .replace("<<PBX_REMAINING_STEPS>>", str(remaining))
            .replace("<<PBX_RESTART_FILE>>", latest_file.name)
        )
        new_input = job_path / f"in.restart_link-{new_link}.lammps"
        new_input.write_text(content)
        logger.info(f"job {job_id}: wrote {new_input.name}")

        return {"in_file": new_input.name}

    def postprocess(self, job_id: int, job_path: Path, db_path: Path) -> str:
        logger = logging.getLogger(__name__)
        logger.info(f"job {job_id}: post-processing started")

        main_dump = job_path / "dump.lammpsdump"
        link_dumps = sorted(
            (
                (int(m.group(1)), p)
                for p in job_path.glob("dump_link-*.lammpsdump")
                for m in [_DUMP_LINK_RE.search(p.name)] if m
            ),
            key=lambda t: t[0],
        )

        if not link_dumps:
            logger.info(f"job {job_id}: no dump_link-*.lammpsdump files to merge")
            return "Done"

        if not main_dump.is_file():
            logger.warning(
                f"job {job_id}: {main_dump.name} missing; creating it from link dumps"
            )
            main_dump.touch()

        with open(main_dump, "ab") as dst:
            for n, src in link_dumps:
                with open(src, "rb") as f:
                    shutil.copyfileobj(f, dst)
                logger.info(f"job {job_id}: appended {src.name} (link {n})")

        logger.info(f"job {job_id}: post-processing completed")
        return "Done"
