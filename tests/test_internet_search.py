"""Tests for Prototype's on-demand web-search decision and formatting."""
import unittest

from internet_search import format_search_results, should_search


class InternetSearchTests(unittest.TestCase):
    def test_explicit_search_request_triggers_search(self):
        self.assertTrue(should_search("Please search the internet for this topic"))

    def test_freshness_request_triggers_search(self):
        self.assertTrue(should_search("What is the latest version today?"))

    def test_question_triggers_search_fallback(self):
        self.assertTrue(should_search("What is 2 + 2?"))

    def test_statement_does_not_force_search(self):
        self.assertFalse(should_search("I like this game."))

    def test_results_are_formatted_as_untrusted_evidence(self):
        text = format_search_results([{
            "title": "Example",
            "url": "https://example.com/page",
            "snippet": "A short excerpt",
        }])
        self.assertIn("https://example.com/page", text)
        self.assertIn("unverified evidence", text)


if __name__ == "__main__":
    unittest.main()
