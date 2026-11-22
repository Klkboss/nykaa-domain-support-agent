"""v2 lint + type-check gate, the same commands locally and in CI.

BLOCKING: ruff and strict mypy on the new v2 code (V2_PATHS / V2_MODULES).
REPORT-ONLY: ruff on the whole repository (the approved capstone code is never restyled); its count is printed.
Usage: .venv\\Scripts\\python.exe scripts\\check_v2.py   (exit code 1 when a blocking check fails)
"""

from __future__ import annotations

import subprocess
import sys

V2_PATHS = ["app/llm", "app/db", "migrations", "tests_v2", "tests_live", "scripts/check_v2.py",
            "scripts/measure_providers.py", "scripts/seed_db.py", "scripts/smoke_api.py", "app/api/rate_limit.py"]
V2_MODULES = ["app/llm", "app/db", "app/api/rate_limit.py"]


def run(*args: str) -> int:
    print("$", " ".join(args), flush=True)
    return subprocess.call([sys.executable, "-m", *args])


def main() -> int:
    failed = run("ruff", "check", *V2_PATHS) != 0
    failed |= run("mypy", *V2_MODULES) != 0
    print("\n--- report only: whole repository (not blocking) ---", flush=True)
    run("ruff", "check", ".", "--exit-zero", "--statistics")
    print("\nBLOCKING CHECKS:", "FAILED" if failed else "PASSED")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
