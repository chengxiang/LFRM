"""Run official EvalPlus inside the external sandbox with a writable cache copy."""

import os
import shutil
import sys
from pathlib import Path


def main():
    if os.environ.get("OCI_UNIT_TEST_SANDBOX") != "1":
        raise RuntimeError("EvalPlus worker must be launched by the sandbox adapter")
    destination = Path.home() / ".cache" / "evalplus"
    # Dataset inputs remain read-only on the host. EvalPlus can add its own
    # ground-truth/timing cache files inside this disposable sandbox.
    shutil.copytree("/input-cache", destination, dirs_exist_ok=True)
    # Import by its canonical name so ProcessPoolExecutor can pickle the
    # official check_correctness function for each worker.
    from evalplus.evaluate import main as evaluate_main

    sys.argv[0] = "evalplus.evaluate"
    evaluate_main()


if __name__ == "__main__":
    main()
