"""Behavioral tests for the Docker argument construction in run.sh."""

from __future__ import annotations

import os
import pty
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    run_sh = tmp_path / "run.sh"
    run_sh.write_text((ROOT / "run.sh").read_text(encoding="utf-8"), encoding="utf-8")
    run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker-args"
    docker = bin_dir / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$DOCKER_ARGS_LOG"\n',
        encoding="utf-8",
    )
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["DOCKER_ARGS_LOG"] = str(log)
    return run_sh, log, env


def test_non_tty_invocation_does_not_request_tty(
    runner: tuple[Path, Path, dict[str, str]],
) -> None:
    run_sh, log, env = runner
    result = subprocess.run(
        [str(run_sh), "true"],
        cwd=run_sh.parent,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = log.read_text(encoding="utf-8").splitlines()
    assert "-i" in args
    assert "-t" not in args
    assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert args[args.index("-e") + 1] == "HOME=/tmp"


def test_terminal_invocation_requests_tty(
    runner: tuple[Path, Path, dict[str, str]],
) -> None:
    run_sh, log, env = runner
    master_fd, slave_fd = pty.openpty()
    try:
        result = subprocess.run(
            [str(run_sh), "true"],
            cwd=run_sh.parent,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            check=False,
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)

    assert result.returncode == 0
    assert "-t" in log.read_text(encoding="utf-8").splitlines()
