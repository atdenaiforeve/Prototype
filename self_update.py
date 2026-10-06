"""Autonomous GitHub self-update and generation handoff for Prototype.

Prototype can use this module to write its own repository files, record a new
generation, and tell the live web UI to reload after GitHub Pages deploys.

The GitHub token is read only from the GITHUB_TOKEN environment variable and
must never be stored in the repository.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
import subprocess
from pathlib import Path


@dataclass(frozen=True)
class UpdateResult:
    path: str
    commit_sha: str
    content_sha: str


@dataclass(frozen=True)
class UpdateProposal:
    intention_id: int
    goal: str
    reason: str
    files: list[str]
    validation_commands: list[list[str]]
    requires_manual_approval: bool
    created_at: int


class GitHubSelfUpdater:
    """Small GitHub Contents API client intended for Prototype itself."""

    def __init__(
        self,
        repository: str = "atdenaiforeve/Prototype",
        branch: str = "main",
        token: str | None = None,
    ) -> None:
        self.repository = repository
        self.branch = branch
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is required for autonomous GitHub updates.")

    @property
    def base_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository}/contents"

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if payload is not None and method in {"POST", "PUT", "PATCH"}:
            body = json.dumps(payload).encode("utf-8")
        else:
            body = None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "Prototype-self-updater",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API error {exc.code}: {detail}") from exc

    def read_file(self, path: str) -> tuple[str, str]:
        encoded = urllib.parse.quote(path, safe="/")
        data = self._request("GET", encoded + f"?ref={urllib.parse.quote(self.branch)}")
        content = data.get("content", "").replace("\n", "")
        import base64
        return base64.b64decode(content).decode("utf-8"), data["sha"]

    def write_file(
        self,
        path: str,
        content: str,
        message: str,
        *,
        sha: str | None = None,
    ) -> UpdateResult:
        import base64
        encoded = urllib.parse.quote(path, safe="/")
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": self.branch,
        }
        if sha is None:
            try:
                _, sha = self.read_file(path)
            except RuntimeError as exc:
                if "404" not in str(exc):
                    raise
        if sha:
            payload["sha"] = sha
        data = self._request("PUT", encoded, payload)
        return UpdateResult(
            path=path,
            commit_sha=data["commit"]["sha"],
            content_sha=data["content"]["sha"],
        )

    def record_generation(
        self,
        generation: int,
        previous_generation: int | None,
        changed_files: list[str],
        reason: str,
        result: str = "committed",
    ) -> UpdateResult:
        if generation < 0:
            raise ValueError("generation must not be negative")
        payload = {
            "generation": generation,
            "previous_generation": previous_generation,
            "changed_files": changed_files,
            "reason": reason,
            "result": result,
            "updated_at": int(time.time()),
        }
        return self.write_file(
            "generation.json",
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            f"Prototype: advance to generation {generation}",
        )

    def latest_intention(self, *, freshness_seconds: int = 24 * 60 * 60) -> dict | None:
        """Read Prototype's newest fresh self-update intention from memory."""
        from memory import get_memory

        return get_memory().latest_self_update_intention(
            freshness_seconds=freshness_seconds,
        )

    def mark_intention(self, memory_id: int, status: str) -> bool:
        """Update the lifecycle status of a self-update intention."""
        from memory import get_memory

        return get_memory().set_self_update_intention_status(memory_id, status)

    def inspect_file(self, path: str) -> dict:
        """Read one repository file and return a compact inspection record."""
        content, sha = self.read_file(path)
        return {
            "path": path,
            "sha": sha,
            "bytes": len(content.encode("utf-8")),
            "lines": content.count("\n") + (1 if content else 0),
            "content": content,
        }

    def latest_update_intention(self, *, freshness_seconds: int = 24 * 60 * 60) -> dict | None:
        """Get Prototype's newest fresh self-update intention."""
        from memory import get_memory
        return get_memory().latest_self_update_intention(
            freshness_seconds=freshness_seconds,
        )

    @staticmethod
    def validate_paths(paths: list[str]) -> list[str]:
        """Reject protected or unsafe paths from autonomous updates."""
        blocked = (".env", "secrets", "credentials", "token", "memory.db", ".git/", ".github/workflows/")
        cleaned: list[str] = []
        for raw in paths:
            path = raw.replace("\\", "/").lstrip("./")
            lowered = path.lower()
            if not path or ".." in Path(path).parts:
                raise ValueError(f"Unsafe update path: {raw}")
            if any(part in lowered for part in blocked):
                raise ValueError(f"Protected update path: {raw}")
            if path not in cleaned:
                cleaned.append(path)
        if len(cleaned) > 8:
            raise ValueError("An autonomous update may change at most 8 files at once.")
        return cleaned

    def build_proposal(self, *, intention: dict | None = None, candidate_files: list[str] | None = None) -> UpdateProposal:
        """Create a bounded proposal; this method never edits code."""
        intention = intention or self.latest_update_intention()
        if not intention:
            raise RuntimeError("No fresh self-update intention is available.")
        metadata = intention.get("metadata") or {}
        files = self.validate_paths(candidate_files or [])
        return UpdateProposal(
            intention_id=int(intention["id"]),
            goal=str(intention["content"]),
            reason=str(metadata.get("reason", "")),
            files=files,
            validation_commands=[
                ["python", "-m", "compileall", "-q", "."],
                ["python", "-m", "pytest", "-q"],
            ],
            requires_manual_approval=any(
                Path(path).name == "self_update.py" or path.startswith(".github/")
                for path in files
            ),
            created_at=int(time.time()),
        )

    def validate_local_checkout(self, root: str | Path = ".") -> dict:
        """Run syntax checks and tests without changing repository files."""
        results = []
        for command in (
            ["python", "-m", "compileall", "-q", "."],
            ["python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
        ):
            try:
                completed = subprocess.run(
                    command, cwd=str(root), text=True, capture_output=True, timeout=120,
                )
                results.append({
                    "command": command,
                    "returncode": completed.returncode,
                    "passed": completed.returncode == 0,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                })
            except (OSError, subprocess.TimeoutExpired) as exc:
                results.append({"command": command, "returncode": None, "passed": False, "error": str(exc)})
        return {
            "passed": bool(results[0]["passed"] and results[1]["passed"]),
            "results": results,
        }

    def create_backup_branch(self, label: str | None = None) -> str:
        """Create a Git branch pointing at the current branch as a rollback point."""
        branch_name = f"prototype-backup-{label or int(time.time())}"
        branch_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in branch_name)
        ref = self._request("GET", f"git/ref/heads/{urllib.parse.quote(self.branch, safe='')}")
        self._request("POST", "git/refs", {"ref": f"refs/heads/{branch_name}", "sha": ref["object"]["sha"]})
        return branch_name

    def record_update_event(self, *, intention_id: int, status: str, changed_files: list[str] | None = None, details: str = "") -> UpdateResult:
        """Persist an auditable, capped history of self-update attempts."""
        path = "self_update_history.json"
        try:
            current, sha = self.read_file(path)
            history = json.loads(current)
            if not isinstance(history, list):
                history = []
        except RuntimeError as exc:
            if "404" not in str(exc):
                raise
            history, sha = [], None
        history.append({
            "intention_id": int(intention_id),
            "status": status,
            "changed_files": self.validate_paths(changed_files or []),
            "details": details[:2000],
            "updated_at": int(time.time()),
        })
        return self.write_file(
            path, json.dumps(history[-200:], indent=2, ensure_ascii=False) + "\n",
            f"Prototype: record self-update {status}", sha=sha,
        )

    def request_reload(self, generation: int) -> UpdateResult:
        return self.record_generation(
            generation=generation,
            previous_generation=max(0, generation - 1),
            changed_files=[],
            reason="Reload into the newly deployed generation.",
            result="reload",
        )
