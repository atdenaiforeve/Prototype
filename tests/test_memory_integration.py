"""Tests for Prototype's memory-aware reasoning integration."""

import tempfile
import unittest
from pathlib import Path

from learning import LearningLoop
from memory import MemoryEngine
from memory_context import build_prompt, format_memory_context
from reasoning import ReasoningWorkspace
from reasoning_engine import ReasoningResult


class MemoryContextTests(unittest.TestCase):
    def test_relevant_memory_is_formatted_and_unrelated_is_not(self):
        memories = [
            {"content": "Prototype learned that cats are animals.", "memory_type": "fact", "confidence": 0.9},
            {"content": "The weather lesson discussed clouds.", "memory_type": "training", "confidence": 0.8},
        ]
        context = format_memory_context([memories[0]], max_chars=300)
        self.assertIn("cats are animals", context)
        self.assertNotIn("weather", context)

    def test_context_is_bounded(self):
        memories = [
            {"content": "x " * 500, "memory_type": "note", "confidence": 1.0},
        ]
        context = format_memory_context(memories, max_chars=100, max_memory_chars=40)
        self.assertLessEqual(len(context), 100)

    def test_prompt_keeps_memory_separate_from_current_input(self):
        prompt = build_prompt("What are cats?", "[MEMORY CONTEXT]\n- Cats are animals.")
        self.assertTrue(prompt.startswith("[MEMORY CONTEXT]"))
        self.assertIn("[CURRENT INPUT]\nWhat are cats?", prompt)


class LearningLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = MemoryEngine(Path(self.temp.name) / "memory.db")
        self.loop = LearningLoop(self.memory)

    def tearDown(self):
        self.temp.cleanup()

    def test_retrieve_returns_relevant_experience(self):
        self.memory.remember("Prototype learned cats are animals.", memory_type="experience", confidence=0.9)
        self.memory.remember("Prototype learned about spacecraft.", memory_type="experience", confidence=0.9)
        results = self.loop.retrieve("cats animals", limit=3)
        self.assertEqual(len(results), 1)
        self.assertIn("cats", results[0]["content"])

    def test_empty_query_returns_no_memories(self):
        self.memory.remember("Cats are animals.", memory_type="fact", confidence=0.9)
        self.assertEqual(self.memory.recall("", limit=5), [])

    def test_deduplication_respects_memory_type(self):
        first = self.memory.remember("Same content.", memory_type="fact", confidence=0.9)
        same_type = self.memory.remember("Same content.", memory_type="fact", confidence=0.9)
        different_type = self.memory.remember("Same content.", memory_type="note", confidence=0.9)

        self.assertEqual(first, same_type)
        self.assertNotEqual(first, different_type)

    def test_self_update_freshness_uses_requested_window(self):
        intention_id = self.memory.remember_self_update_intention(
            "Old update intention",
            freshness_seconds=3600,
        )
        with self.memory._connect() as con:
            con.execute(
                "UPDATE memories SET created_at = created_at - 100 WHERE id = ?",
                (intention_id,),
            )

        self.assertIsNone(
            self.memory.latest_self_update_intention(freshness_seconds=10)
        )

    def test_feedback_creates_correction(self):
        workspace = ReasoningWorkspace()
        hypothesis = workspace.add_hypothesis("Cats are animals.", 0.8)
        result = ReasoningResult("Cats are animals.", hypothesis, workspace.snapshot())
        record = self.loop.record_result("What are cats?", result)
        updated = self.loop.give_feedback(record, good=False, feedback="Mention that cats are mammals too.")
        self.assertLess(updated.score, record.score)
        corrections = self.memory.recall("cats mammals correction", limit=5, memory_type="correction")
        self.assertTrue(corrections)


if __name__ == "__main__":
    unittest.main()
