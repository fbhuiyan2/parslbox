"""
Compute-node hook runner.

Invoked as a subprocess on the assigned compute node when an app opts into
`RUN_HOOKS_ON_COMPUTE = True`. Re-instantiates the app via the registry,
calls the requested hook method, and reports the return value via a file.

Invocation:
    python -m parslbox.apps._hook_runner <app_name> <method_name> <args_json_path>

Return-value channel: writes str(result or "") to <job_path>/PBX_HOOK_RETURN on
success. The empty-string sentinel maps to None on the head side (mirrors the
PBX_JOB_STATUS_REPORT idiom in parslbox/apps/utils.py + parslbox/apps/python.py).
"""

import json
import logging
import sys
import traceback
from pathlib import Path

from parslbox.apps.app_registry import get_app_instance


ALLOWED_METHODS = {"preprocess", "postprocess"}
PBX_HOOK_RETURN_FILE = "PBX_HOOK_RETURN"
# Args that should be rehydrated from str to Path before passing to the method.
PATH_ARGS = {"job_path", "db_path"}


def _rehydrate_paths(kwargs: dict) -> dict:
    for key in PATH_ARGS:
        if key in kwargs and isinstance(kwargs[key], str):
            kwargs[key] = Path(kwargs[key])
    return kwargs


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        print(
            f"Usage: python -m parslbox.apps._hook_runner <app_name> <method_name> <args_json_path>",
            file=sys.stderr,
        )
        return 2

    app_name, method_name, args_json_path = argv
    args_path = Path(args_json_path)

    if method_name not in ALLOWED_METHODS:
        print(
            f"Refusing to dispatch method '{method_name}'. "
            f"Allowed: {sorted(ALLOWED_METHODS)}",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    try:
        kwargs = json.loads(args_path.read_text())
        kwargs = _rehydrate_paths(kwargs)
        # job_path is required for the return-value file location.
        job_path = kwargs.get("job_path")
        if not isinstance(job_path, Path):
            print(
                f"Missing or invalid 'job_path' in args JSON; cannot place {PBX_HOOK_RETURN_FILE}",
                file=sys.stderr,
            )
            return 2

        logger.info(f"hook_runner: app={app_name} method={method_name}")
        app = get_app_instance(app_name)
        method = getattr(app, method_name)
        result = method(**kwargs)

        return_file = job_path / PBX_HOOK_RETURN_FILE
        return_file.write_text(str(result) if result is not None else "")
        logger.info(f"hook_runner: wrote {return_file}")
        return 0

    except Exception:
        traceback.print_exc()
        return 1

    finally:
        # Always clear the args file so a retry can't read stale state.
        args_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
