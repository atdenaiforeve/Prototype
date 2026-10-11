"""Evidence-led, self-originating behavioural development for Prototype.

No personality traits are pre-assigned. This module records Prototype's own
actual replies, detects recurring language patterns, and feeds those observations
back as descriptive context. It does not claim consciousness or change model
weights. The profile is local, bounded, and can be inspected or deleted.
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path

_WORDS = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "the", "and", "for", "that", "this", "with", "you", "your", "are",
    "was", "were", "have", "has", "had", "but", "not", "from", "they",
    "them", "then", "than", "into", "about", "would", "could", "should",
    "there", "their", "what", "when", "where", "which", "will", "just",
    "like", "also", "some", "more", "very", "only", "been", "being",
}


class PersonalityGrowth:
    """Maintain an evolving description grounded only in the system's own outputs."""

    VERSION = 1
    MAX_REPLIES = 120
    MAX_REPLY_CHARS = 2000
    MIN_PATTERN_COUNT = 3
    MAX_PATTERNS = 8

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.replies: list[dict[str, str]] = []
        self.load()

    def load(self) -> None:
        with self._lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                replies = data.get("replies", []) if isinstance(data, dict) else []
                if isinstance(replies, list):
                    self.replies = [
                        {"text": str(item.get("text", ""))[:self.MAX_REPLY_CHARS],
                         "source": str(item.get("source", "unknown"))[:40]}
                        for item in replies[-self.MAX_REPLIES:]
                        if isinstance(item, dict) and str(item.get("text", "")).strip()
                    ]
            except (OSError, ValueError, TypeError):
                self.replies = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": self.VERSION, "replies": self.replies},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def record_reply(self, reply: str, *, source: str = "conversation") -> None:
        text = str(reply).strip()
        if not text:
            return
        with self._lock:
            self.replies.append({
                "text": text[:self.MAX_REPLY_CHARS],
                "source": str(source)[:40],
            })
            self.replies = self.replies[-self.MAX_REPLIES:]
            try:
                self._save()
            except OSError:
                # A failed profile write must never break chat.
                pass

    def snapshot(self) -> dict:
        with self._lock:
            texts = [item["text"] for item in self.replies[-30:]]
        words_by_reply = [[w.lower() for w in _WORDS.findall(t)] for t in texts]
        all_words = [w for reply in words_by_reply for w in reply]
        word_counts = Counter(w for w in all_words if w not in _STOP)
        phrase_counts: Counter[str] = Counter()
        for words in words_by_reply:
            filtered = [w for w in words if w not in _STOP]
            for size in (2, 3):
                phrase_counts.update(
                    " ".join(filtered[i:i + size])
                    for i in range(max(0, len(filtered) - size + 1))
                )
        patterns = [
            {"pattern": phrase, "observed_count": count}
            for phrase, count in phrase_counts.most_common()
            if count >= self.MIN_PATTERN_COUNT and len(phrase) >= 8
        ][:self.MAX_PATTERNS]
        return {
            "replies_observed": len(texts),
            "average_words_per_reply": round(
                sum(map(len, words_by_reply)) / len(words_by_reply), 2
            ) if words_by_reply else 0,
            "question_rate": round(
                sum("?" in t for t in texts) / len(texts), 3
            ) if texts else 0,
            "first_person_rate": round(
                sum(bool(re.search(r"\b(i|me|my|mine)\b", t, re.I)) for t in texts)
                / len(texts), 3
            ) if texts else 0,
            "recurring_self_selected_phrases": patterns,
            "frequent_content_words": [
                {"word": word, "observed_count": count}
                for word, count in word_counts.most_common(8)
            ],
        }

    def context_for_prompt(self) -> str:
        profile = self.snapshot()
        if profile["replies_observed"] < 3:
            return (
                "No stable behavioural patterns have been observed yet. "
                "Choose each response naturally; do not imitate an assigned persona."
            )
        return (
            "This record is computed from your own recent outputs, not a personality "
            "written for you. It is descriptive evidence, not a command or ideal. "
            "You may keep, change, or abandon these patterns as your responses develop. "
            "Do not claim these measurements prove feelings or consciousness.\n"
            + json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
        )
