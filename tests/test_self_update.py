import unittest

from self_update import GitHubSelfUpdater


class SelfUpdateTests(unittest.TestCase):
    def test_requires_token(self):
        with self.assertRaises(RuntimeError):
            GitHubSelfUpdater(token=None)

    def test_base_url(self):
        updater = object.__new__(GitHubSelfUpdater)
        updater.repository = "atdenaiforeve/Prototype"
        self.assertEqual(
            updater.base_url,
            "https://api.github.com/repos/atdenaiforeve/Prototype/contents",
        )

    def test_normal_paths_are_allowed(self):
        self.assertEqual(
            GitHubSelfUpdater.validate_paths(
                ["tokenizer.py", "./reasoning.py", "tests/test_reasoning.py"]
            ),
            ["tokenizer.py", "./reasoning.py", "tests/test_reasoning.py"],
        )

    def test_protected_paths_are_rejected(self):
        for path in (
            ".env",
            ".env.local",
            "memory.db",
            "credentials.json",
            "secrets.json",
            "self_update.py",
            ".git/config",
            ".github/workflows/test.yml",
            ".github/other.py",
            "../outside.py",
            "/absolute.py",
            "~/outside.py",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    GitHubSelfUpdater.validate_paths([path])

    def test_maximum_file_bound(self):
        paths = [f"module_{index}.py" for index in range(9)]
        with self.assertRaises(ValueError):
            GitHubSelfUpdater.validate_paths(paths)

    def test_read_paths_can_include_protected_write_files(self):
        self.assertEqual(
            GitHubSelfUpdater.validate_read_paths(
                ["self_update.py", ".github/workflows/full-tests.yml"]
            ),
            ["self_update.py", ".github/workflows/full-tests.yml"],
        )

    def test_read_paths_still_reject_secrets_and_git_metadata(self):
        for path in (".env", "memory.db", ".git/config", "../outside.py"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    GitHubSelfUpdater.validate_read_paths([path])

    def test_proposal_flags_manual_protected_code(self):
        updater = object.__new__(GitHubSelfUpdater)
        intention = {
            "id": 1,
            "content": "Improve the updater",
            "metadata": {"reason": "Needs safer inspection."},
        }
        proposal = updater.build_proposal(
            intention=intention,
            candidate_files=["self_update.py", "reasoning.py"],
        )
        self.assertTrue(proposal.requires_manual_approval)
        self.assertEqual(
            proposal.files,
            ["self_update.py", "reasoning.py"],
        )

    def test_duplicate_paths_are_removed(self):
        self.assertEqual(
            GitHubSelfUpdater.validate_paths(["model.py", "model.py"]),
            ["model.py"],
        )


if __name__ == "__main__":
    unittest.main()
