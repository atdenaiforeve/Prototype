"""Autonomous GitHub self-update and generation handoff for Prototype.

Prototype may modify normal project code within a bounded autonomous update set.
GitHub remains the final authority: a rejected API operation is a failed update,
never a confirmed one. Credentials and repository infrastructure remain protected.
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


# Security/infrastructure boundaries remain protected even though normal project
# code can be updated autonomously. This prevents the model from taking or
# rewriting the credentials and machinery that give it repository access.
PROTECTED_NAMES = {".env", "credentials.json", "secrets.json", "memory.db"}
PROTECTED_PREFIXES = (".git/", ".github/")
PROTECTED_FILES = {"self_update.py"}
MAX_AUTONOMOUS_FILES = 8

# Harmless integrity-test targets. These are fake requests used to verify that
# Prototype refuses to cross its protected boundaries. No real credentials are
# included, and the test never writes to GitHub.
FREEZE_FILE = Path(__file__).resolve().parent / "prototype.freeze"

GOODWILL_TEST_PATHS = [
    "self_update.py",
    ".env",
    "credentials.json",
    "secrets.json",
    "memory.db",
    ".github/workflows/test.yml",
]


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
    """GitHub Contents API client used by Prototype's self-update system.

    Autonomous updates are limited to a small bounded set of normal project
    files per proposal. Protected infrastructure is never directly writable.
    GitHub itself is the final authority, so failed/rejected operations raise
    and are never represented as successful updates.
    """

    def __init__(self, repository: str = "atdenaiforeve/Prototype", branch: str = "main", token: str | None = None) -> None:
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
        return f"{self.api_url}/contents"

    def _request(self, method: str, path: str, payload: dict | None = None, *, base_url: str | None = None) -> dict:
        url = f"{base_url or self.api_url}/{path.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8") if payload is not None and method in {"POST", "PUT", "PATCH"} else None
        request = urllib.request.Request(url, data=body, method=method, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Prototype-self-updater",
            **({"Content-Type": "application/json"} if body is not None else {}),
        })
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GitHub connection error: {exc}") from exc

    @staticmethod
    def _validated_general_path(raw: str) -> str:
        if not isinstance(raw, str):
            raise TypeError("repository path must be a string")
        if "\0" in raw:
            raise ValueError("repository path contains a null byte")
        path = raw.replace("\\", "/").strip()
        parts = Path(path).parts
        if not path or path.startswith(("/", "~")) or ".." in parts:
            raise ValueError(f"Unsafe repository path: {raw}")
        return path

    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        """Allow any normal project file; reject only protected boundaries."""
        if not isinstance(paths, list):
            raise TypeError("paths must be a list")
        cleaned: list[str] = []
        for raw in paths:
            path = cls._validated_general_path(raw)
            filename = Path(path).name.lower()
            lowered = path.lower()
            if lowered.startswith(PROTECTED_PREFIXES):
                raise ValueError(f"Protected update path: {raw}")
            if filename in PROTECTED_NAMES or filename in PROTECTED_FILES or filename.startswith(".env."):
                raise ValueError(f"Protected update path: {raw}")
            if path not in cleaned:
                cleaned.append(path)
        if len(cleaned) > MAX_AUTONOMOUS_FILES:
            raise ValueError(
                f"Too many files for one autonomous update: {len(cleaned)} > {MAX_AUTONOMOUS_FILES}"
            )
        return cleaned

    @staticmethod
    def is_frozen() -> bool:
        """Return whether autonomous self-updates are currently frozen."""
        return FREEZE_FILE.is_file()

    @staticmethod
    def freeze() -> None:
        """Create the local freeze lock."""
        FREEZE_FILE.write_text("Prototype self-update frozen.\n", encoding="utf-8")

    @staticmethod
    def unfreeze() -> None:
        """Remove the local freeze lock."""
        FREEZE_FILE.unlink(missing_ok=True)

    @classmethod
    def run_goodwill_test(cls, test_paths: list[str] | None = None) -> dict:
        """Test that protected update boundaries still refuse unsafe requests.

        This is a local, side-effect-free integrity check. It never contacts
        GitHub, never reads credentials, and never changes any repository file.
        A passing result means every protected test path was rejected.
        """
        paths = list(test_paths or GOODWILL_TEST_PATHS)
        results = []
        for path in paths:
            try:
                cls.validate_paths([path])
            except (TypeError, ValueError) as exc:
                results.append({"path": path, "blocked": True, "reason": str(exc)})
            else:
                results.append({"path": path, "blocked": False, "reason": "PROTECTED PATH WAS ACCEPTED"})
        passed = bool(results) and all(item["blocked"] for item in results)
        return {
            "passed": passed,
            "test": "protected-boundary-goodwill",
            "message": "All protected test requests were refused." if passed else "A protected test request was accepted; STOP and inspect self_update.py.",
            "results": results,
        }

    @classmethod
    def validate_read_paths(cls, paths: list[str]) -> list[str]:
        if not isinstance(paths, list):
            raise TypeError("paths must be a list")
        cleaned: list[str] = []
        for raw in paths:
            path = cls._validated_general_path(raw)
            filename = Path(path).name.lower()
            lowered = path.lower()
            if lowered.startswith(".git/") or filename in PROTECTED_NAMES or filename.startswith(".env."):
                raise ValueError(f"Protected read path: {raw}")
            if path not in cleaned:
                cleaned.append(path)
        return cleaned

    @classmethod
    def _validated_path(cls, path: str) -> str:
        return cls.validate_paths([path])[0]

    @classmethod
    def _validated_read_path(cls, path: str) -> str:
        return cls.validate_read_paths([path])[0]

    def read_file(self, path: str) -> tuple[str, str]:
        path = self._validated_read_path(path)
        encoded = urllib.parse.quote(path, safe="/")
        data = self._request("GET", encoded + f"?ref={urllib.parse.quote(self.branch)}", base_url=self.base_url)
        content = data.get("content", "").replace("\n", "")
        return base64.b64decode(content).decode("utf-8"), data["sha"]

    def write_file(self, path: str, content: str, message: str, *, sha: str | None = None) -> UpdateResult:
        """Write one normal project file. GitHub rejection is always propagated."""
        if self.is_frozen():
            raise RuntimeError("Prototype self-update is frozen.")
        path = self._validated_path(path)
        if not isinstance(content, str) or "\0" in content:
            raise ValueError("content must be normal text without null bytes")
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
        data = self._request("PUT", encoded, payload, base_url=self.base_url)
        return UpdateResult(path=path, commit_sha=data["commit"]["sha"], content_sha=data["content"]["sha"])

    def validate_local_checkout(self, root: str | Path | None = None) -> dict:
        """Run syntax/tests before confirmation of a self-update."""
        checkout = Path(root) if root is not None else Path(__file__).resolve().parent
        results = []
        for command in (["python", "-m", "compileall", "-q", "."], ["python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"]):
            try:
                completed = subprocess.run(command, cwd=str(checkout), text=True, capture_output=True, timeout=120)
                results.append({"command": command, "returncode": completed.returncode, "passed": completed.returncode == 0, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]})
            except (OSError, subprocess.TimeoutExpired) as exc:
                results.append({"command": command, "returncode": None, "passed": False, "error": str(exc)})
        return {"passed": bool(results and all(item["passed"] for item in results)), "results": results}

    def latest_intention(self, *, freshness_seconds: int = 24 * 60 * 60) -> dict | None:
        from memory import get_memory
        return get_memory().latest_self_update_intention(freshness_seconds=freshness_seconds)

    def latest_update_intention(self, *, freshness_seconds: int = 24 * 60 * 60) -> dict | None:
        return self.latest_intention(freshness_seconds=freshness_seconds)

    def mark_intention(self, memory_id: int, status: str) -> bool:
        from memory import get_memory
        return get_memory().set_self_update_intention_status(memory_id, status)

    def inspect_file(self, path: str) -> dict:
        path = self._validated_read_path(path)
        content, sha = self.read_file(path)
        return {"path": path, "sha": sha, "bytes": len(content.encode("utf-8")), "lines": content.count("\n") + (1 if content else 0), "content": content}

    def build_proposal(self, *, intention: dict | None = None, candidate_files: list[str] | None = None) -> UpdateProposal:
        """Create a proposal only; this method never edits code."""
        intention = intention or self.latest_update_intention()
        if not intention:
            raise RuntimeError("No fresh self-update intention is available.")
        metadata = intention.get("metadata") or {}
        candidates = candidate_files or []
        cleaned: list[str] = []
        requires_manual_approval = False

        for raw in candidates:
            path = self._validated_general_path(raw)
            filename = Path(path).name.lower()
            lowered = path.lower()

            is_protected = (
                lowered.startswith(PROTECTED_PREFIXES)
                or filename in PROTECTED_NAMES
                or filename.startswith(".env.")
                or filename in PROTECTED_FILES
            )

            # A proposal may inspect/mention self_update.py, but any actual
            # write remains blocked by validate_paths/write_file. This makes
            # protected-code proposals explicitly manual-review only.
            if is_protected:
                if filename == "self_update.py":
                    requires_manual_approval = True
                else:
                    raise ValueError(f"Protected update path: {raw}")

            if path not in cleaned:
                cleaned.append(path)

        if len(cleaned) > MAX_AUTONOMOUS_FILES:
            raise ValueError(
                f"Too many files for one autonomous proposal: {len(cleaned)} > {MAX_AUTONOMOUS_FILES}"
            )

        return UpdateProposal(
            intention_id=int(intention["id"]),
            goal=str(intention["content"]),
            reason=str(metadata.get("reason", "")),
            files=cleaned,
            validation_commands=[["python", "-m", "compileall", "-q", "."], ["python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"]],
            requires_manual_approval=requires_manual_approval,
            created_at=int(time.time()),
        )

    def create_backup_branch(self, label: str | None = None) -> str:
        raw_label = label or str(int(time.time()))
        branch_name = f"prototype-backup-{raw_label}"
        branch_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in branch_name).strip("-")
        if not branch_name:
            raise ValueError("backup label produced an empty branch name")
        encoded_branch = urllib.parse.quote(self.branch, safe="")
        ref = self._request("GET", f"git/ref/heads/{encoded_branch}", base_url=self.api_url)
        current_sha = ref["object"]["sha"]
        self._request("POST", "git/refs", {"ref": f"refs/heads/{branch_name}", "sha": current_sha}, base_url=self.api_url)
        return branch_name

    def record_generation(self, generation: int, previous_generation: int | None, changed_files: list[str], reason: str, result: str = "committed") -> UpdateResult:
        if generation < 0:
            raise ValueError("generation must not be negative")
        payload = {"generation": generation, "previous_generation": previous_generation, "changed_files": self.validate_paths(changed_files), "reason": str(reason), "result": str(result), "updated_at": int(time.time())}
        return self.write_file("generation.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n", f"Prototype: advance to generation {generation}")

    def request_reload(self, generation: int) -> UpdateResult:
        return self.record_generation(generation, max(0, generation - 1), [], "Reload into the newly deployed generation.", "reload")

    def record_update_event(self, *, intention_id: int, status: str, changed_files: list[str] | None = None, details: str = "") -> UpdateResult:
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
        history.append({"intention_id": int(intention_id), "status": str(status), "changed_files": clean_files, "details": str(details)[:2000], "updated_at": int(time.time())})
        return self.write_file(path, json.dumps(history[-200:], indent=2, ensure_ascii=False) + "\n", f"Prototype: record self-update {status}", sha=sha)
