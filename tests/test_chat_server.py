"""Tests for the Prototype chat wrapper without loading a neural checkpoint."""

from chat_server import PrototypeChat


def test_chat_class_can_be_imported():
    assert callable(PrototypeChat)


def test_empty_message_is_rejected():
    chat = object.__new__(PrototypeChat)
    chat.conversations = {}
    chat.lock = __import__("threading").Lock()

    try:
        chat.chat("   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty messages must be rejected")
