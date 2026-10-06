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


if __name__ == "__main__":
    unittest.main()
