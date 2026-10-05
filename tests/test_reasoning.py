import unittest

from reasoning import ReasoningWorkspace


class ReasoningWorkspaceTests(unittest.TestCase):
    def test_compares_and_chooses_hypothesis(self):
        workspace = ReasoningWorkspace(max_hypotheses=2)
        first = workspace.add_hypothesis("first idea", 0.3)
        second = workspace.add_hypothesis("second idea", 0.8)
        workspace.add_evidence(second, "strong evidence")
        chosen = workspace.choose()

        self.assertIs(chosen, second)
        self.assertEqual(second.status, "chosen")
        self.assertEqual(first.status, "rejected")
        self.assertEqual(workspace.decisions, ["second idea"])

    def test_workspace_is_bounded(self):
        workspace = ReasoningWorkspace(max_hypotheses=2)
        workspace.add_hypothesis("one")
        workspace.add_hypothesis("two")
        workspace.add_hypothesis("three")
        self.assertEqual([h.text for h in workspace.hypotheses], ["two", "three"])


if __name__ == "__main__":
    unittest.main()
