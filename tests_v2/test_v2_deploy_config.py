"""v2 Stage D: the Docker / CI configuration keeps its promises (static checks, no Docker needed)."""

from __future__ import annotations

import re

import yaml

from app.core.config import ROOT


def _lock(name: str) -> dict[str, str]:
    pins = (line.split("==", 1) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if "==" in line)
    return {pkg.strip().lower().replace("_", "-"): ver.split("+")[0].strip() for pkg, ver in pins}


def test_every_github_action_is_pinned_to_a_full_commit_sha():
    uses = re.findall(r"uses:\s*(\S+)", (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses), uses


def test_ci_needs_no_secrets_and_runs_the_suite_lint_and_postgres_tests():
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "secrets." not in text and 'MOCK_LLM: "true"' in text
    for command in ("scripts/check_v2.py", "python -m pytest -q\n", "REQUIRE_TEST_DATABASE", "scripts/smoke_api.py"):
        assert command in text


def test_compose_has_no_literal_secrets_and_exposes_only_the_api_on_loopback():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert services["postgres"]["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?")
    assert "ports" not in services["postgres"]
    assert services["api"]["ports"] == ["127.0.0.1:8000:8000"]
    assert "${POSTGRES_PASSWORD" in services["api"]["environment"]["DATABASE_URL"]


def test_the_image_runs_as_non_root_in_mock_mode_and_never_copies_env_files():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime = dockerfile.split("AS runtime", 1)[1]
    assert "USER app" in runtime and "MOCK_LLM=true" in runtime
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ignore[ignore.index("*")] == "*" and not any(line.startswith("!.env") for line in ignore)


def test_the_linux_lock_pins_the_same_versions_as_the_windows_lock():
    windows, linux = _lock("requirements.lock.txt"), _lock("requirements.lock.linux.txt")
    shared = windows.keys() & linux.keys()
    assert {"crewai", "chromadb", "torch", "sqlalchemy", "psycopg"} <= shared
    assert {k: windows[k] for k in shared} == {k: linux[k] for k in shared}
    assert "pywin32" not in linux and "+cpu" in (ROOT / "requirements.lock.linux.txt").read_text(encoding="utf-8")
