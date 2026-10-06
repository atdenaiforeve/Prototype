"""Autonomous GitHub self-update and generation handoff for Prototype.

Prototype can use this module to write its own repository files, record a new
generation, and tell the live web UI to reload after GitHub Pages deploys.

The GitHub token is read only from the GITHUB_TOKEN environment variable and
must never be stored in the repository.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROTECTED_NAMES = {
    ".env",
    "credentials.json",
    "secrets.json",
    "memory.db",
}

PROTECTED_PREFIXES = (
    ".git/",
    ".github/",
)

PROTECTED_FILES = {
    "self_update.py",
}


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
    """Small GitHub Contents API client intended for Prototype itself.

    The public write primitive is deliberately guarded: autonomous updates can
    only touch ordinary repository files, never credentials, git metadata,
    workflow definitions, the memory database, or this module itself.
    """

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
    def api_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository}"

    @property
    def base_url(self) -> str:
        """Backward-compatible alias for the repository Contents API."""
        return f"{self.api_url}/contents"

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        base_url: str | None = None,
    ) -> dict:
        url = f"{base_url or self.api_url}/{path.lstrip('/')}"
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
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GitHub connection error: {exc}") from exc

    def read_file(self, path: str) -> tuple[str, str]:
        path = self._validated_path(path)
        encoded = urllib.parse.quote(path, safe="/")
        data = self._request(
            "GET",
            encoded + f"?ref={urllib.parse.quote(self.branch)}",
            base_url=self.base_url,
        )
        content = data.get("content", "").replace("\n", "")
        return base64.b64decode(content).decode("utf-8"), data["sha"]

    def write_file(
        self,
        path: str,
        content: str,
        message: str,
        *,
        sha: str | None = None,
    ) -> UpdateResult:
        """Write one file after enforcing autonomous-update path policy."""
        path = self._validated_path(path)
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if not str(message).strip():
            raise ValueError("commit message must not be empty")

        encoded = urllib.parse.quote(path, safe="/")
        payload = {
            "message": str(message).strip(),
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": self.branch,
        }
        if sha is None:
            try:
                _, sha = self.read_file(path)
            except RuntimeError as exc:
                if "GitHub API error 404" not in str(exc):
                    raise
        if sha:
            payload["sha"] = sha
        data = self._request(
            "PUT",
            encoded,
            payload,
            base_url=self.base_url,
        )
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
        clean_files = self.validate_paths(changed_files)
        payload = {
            "generation": generation,
            "previous_generation": previous_generation,
            "changed_files": clean_files,
            "reason": str(reason),
            "result": str(result),
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
        path = self._validated_path(path)
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
    def _normalise_path(raw: str) -> str:
        if not isinstance(raw, str):
            raise TypeError("update path must be a string")
        return raw.replace("\\", "/").strip()

    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        """Reject protected or unsafe paths from autonomous updates."""
        if not isinstance(paths, list):
            raise TypeError("paths must be a list")

        cleaned: list[str] = []
        for raw in paths:
            path = cls._normalise_path(raw)
            parts = Path(path).parts
            filename = Path(path).name.lower()

            if not path or path.startswith(("/", "~")) or ".." in parts:
                raise ValueError(f"Unsafe update path: {raw}")

            lowered = path.lower()
            if lowered.startswith(PROTECTED_PREFIXES):
                raise ValueError(f"Protected update path: {raw}")
            if filename in PROTECTED_NAMES or filename in PROTECTED_FILES:
                raise ValueError(f"Protected update path: {raw}")
            if filename.startswith(".env."):
                raise ValueError(f"Protected update path: {raw}")

            if path not in cleaned:
                cleaned.append(path)

        if len(cleaned) > 8:
            raise ValueError("An autonomous update may change at most 8 files at once.")
        return cleaned

    @classmethod
    def _validated_path(cls, path: str) -> str:
        paths = cls.validate_paths([path])
        return paths[0]

    def build_proposal(
        self,
        *,
        intention: dict | None = None,
        candidate_files: list[str] | None = None,
    ) -> UpdateProposal:
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
                ["python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
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
                    command,
                    cwd=str(root),
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                results.append({
                    "command": command,
                    "returncode": completed.returncode,
                    "passed": completed.returncode == 0,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                })
            except (OSError, subprocess.TimeoutExpired) as exc:
                results.append({
                    "command": command,
                    "returncode": None,
                    "passed": False,
                    "error": str(exc),
                })
        return {
            "passed": bool(results and all(result["passed"] for result in results)),
            "results": results,
        }

    def create_backup_branch(self, label: str | None = None) -> str:
        """Create a Git branch pointing at the current branch as a rollback point."""
        raw_label = label or str(int(time.time()))
        branch_name = f"prototype-backup-{raw_label}"
        branch_name = "".join(
            ch if ch.isalnum() or ch in "-_" else "-" for ch in branch_name
        ).strip("-")
        if not branch_name:
            raise ValueError("backup label produced an empty branch name")

        encoded_branch = urllib.parse.quote(self.branch, safe="")
        ref = self._request(
            "GET",
            f"git/ref/heads/{encoded_branch}",
            base_url=self.api_url,
        )
        current_sha = ref["object"]["sha"]

        encoded_backup = urllib.parse.quote(branch_name, safe="")
        try:
            self._request(
                "POST",
                "git/refs",
                {
                    "ref": f"refs/heads/{branch_name}",
                    "sha": current_sha,
                },
                base_url=self.api_url,
            )
        except RuntimeError as exc:
            if "GitHub API error 422" in str(exc):
                raise RuntimeError(
                    f"backup branch already exists: {branch_name}"
                ) from exc
            raise
        return branch_name

    def record_update_event(
        self,
        *,
        intention_id: int,
        status: str,
        changed_files: list[str] | None = None,
        details: str = "",
    ) -> UpdateResult:
        """Persist an auditable, capped history of self-update attempts."""
        path = "self_update_history.json"
        clean_files = self.validate_paths(changed_files or [])
        try:
            current, sha = self.read_file(path)
            history = json.loads(current)
            if not isinstance(history, list):
                history = []
        except RuntimeError as exc:
            if "GitHub API error 404" not in str(exc):
                raise
            history, sha = [], None

        history.append({
            "intention_id": int(intention_id),
            "status": str(status),
            "changed_files": clean_files,
            "details": str(details)[:2000],
            "updated_at": int(time.time()),
        })
        return self.write_file(
            path,
            json.dumps(history[-200:], indent=2, ensure_ascii=False) + "\n",
            f"Prototype: record self-update {status}",
            sha=sha,
        )

    def request_reload(self, generation: int) -> UpdateResult:
        return self.record_generation(
            generation=generation,
            previous_generation=max(0, generation - 1),
            changed_files=[],
            reason="Reload into the newly deployed generation.",
            result="reload",
        )
