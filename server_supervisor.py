"""Prototype server supervisor.

Keeps the running Codespaces server synchronized with the repository. It checks
for new commits, performs a fast-forward-only pull when available, and restarts
the chat server so code changes become live automatically.

It never force-resets the checkout and never overwrites uncommitted work.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECK_INTERVAL = max(2, int(os.environ.get("PROTOTYPE_UPDATE_INTERVAL", "2")))


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def checkout_is_clean() -> bool:
    return not run_git("status", "--porcelain").stdout.strip()


def update_available() -> bool:
    fetch = run_git("fetch", "origin", "main", "--quiet")
    if fetch.returncode != 0:
        print(f"[Prototype] update check failed: {fetch.stderr.strip()}")
        return False
    local = run_git("rev-parse", "HEAD").stdout.strip()
    remote = run_git("rev-parse", "origin/main").stdout.strip()
    return bool(local and remote and local != remote)


def update_checkout() -> bool:
    if not checkout_is_clean():
        print("[Prototype] update found, but checkout has local changes; skipping.")
        return False

    result = run_git("pull", "--ff-only", "origin", "main")
    if result.returncode != 0:
        print(f"[Prototype] automatic update skipped: {result.stderr.strip()}")
        return False

    print("[Prototype] repository updated.")
    return True


def start_server() -> subprocess.Popen[str]:
    print("[Prototype] starting chat server...")
    return subprocess.Popen(
        [sys.executable, str(ROOT / "chat_server.py")],
        cwd=ROOT,
        text=True,
    )


def main() -> None:
    child = start_server()

    def stop(*_args: object) -> None:
        if child.poll() is None:
            child.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while True:
        time.sleep(CHECK_INTERVAL)

        if child.poll() is not None:
            print("[Prototype] chat server stopped; restarting.")
            child = start_server()
            continue

        if update_available() and update_checkout():
            print("[Prototype] restarting chat server with the new code...")
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            child = start_server()


if __name__ == "__main__":
    main()
