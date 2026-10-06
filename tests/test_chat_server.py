"""Tests for the Prototype chat wrapper without loading a neural checkpoint."""

import threading
import unittest

from chat_server import PrototypeChat


class PrototypeChatTests(unittest.TestCase):
    def test_chat_class_can_be_imported(self):
        self.assertTrue(callable(PrototypeChat))

    def test_empty_message_is_rejected(self):
        chat = object.__new__(PrototypeChat)
        chat.conversations = {}
        chat.lock = threading.Lock()

        with self.assertRaises(ValueError) as context:
            chat.chat("   ")

        self.assertIn("empty", str(context.exception))

    def test_non_string_message_is_rejected(self):
        chat = object.__new__(PrototypeChat)
        chat.conversations = {}
        chat.lock = threading.Lock()

        with self.assertRaises(ValueError):
            chat.chat(None)


if __name__ == "__main__":
    unittest.main()
