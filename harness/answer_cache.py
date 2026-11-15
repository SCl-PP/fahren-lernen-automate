"""Persistent question-answer cache keyed on perceptual image hash.

Once a question has been seen and its Musterloesung extracted from the
feedback screen, later encounters resolve from the cache without re-asking
Gemini for the answer -- guaranteeing correctness and dropping API calls.

Storage: a single JSON file at runs/answer_cache.json. Loaded at session
start, saved after every update.
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import imagehash

from .capture import Capture

log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("runs/answer_cache.json")
HASH_SIZE = 16  # 16x16 phash = 256-bit
FUZZY_DISTANCE = 4
TOP_CROP_PX = 80  # strip progress bar (X/Y counter + bar) from hash
BOTTOM_CROP_PX = 120  # strip button area that changes per flow state


def question_hash(capture: Capture) -> str:
    img = capture.image
    top = TOP_CROP_PX
    bottom_bound = max(top + 1, img.height - BOTTOM_CROP_PX)
    cropped = img.crop((0, top, img.width, bottom_bound))
    return str(imagehash.phash(cropped, hash_size=HASH_SIZE))


@dataclass
class CacheEntry:
    question_type: str  # "foto" | "text" | "input"
    # Multiple-choice fields:
    correct_solutions: list[int] = field(default_factory=list)
    answer_boxes: list[dict] = field(default_factory=list)
    # Input fields:
    answer_value: str | None = None
    input_field: dict | None = None
    # Meta:
    label: str = ""
    first_seen: str = ""
    hit_count: int = 0


class AnswerCache:
    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        self.path = path
        self._entries: dict[str, CacheEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            self._entries = {k: CacheEntry(**v) for k, v in raw.items()}
            log.info("answer cache loaded: %d entries from %s", len(self._entries), self.path)
        except Exception as e:
            log.warning("failed to load cache %s: %s (starting empty)", self.path, e)
            self._entries = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {k: asdict(e) for k, e in self._entries.items()}
        self.path.write_text(json.dumps(serialised, indent=2, ensure_ascii=False))

    def lookup(self, capture: Capture) -> CacheEntry | None:
        q_hash = question_hash(capture)
        entry = self._entries.get(q_hash)
        if entry is not None:
            entry.hit_count += 1
            return entry
        current = imagehash.hex_to_hash(q_hash)
        for stored_hash, entry in self._entries.items():
            if (imagehash.hex_to_hash(stored_hash) - current) <= FUZZY_DISTANCE:
                entry.hit_count += 1
                return entry
        return None

    def remember(self, capture: Capture, entry: CacheEntry) -> None:
        q_hash = question_hash(capture)
        if not entry.first_seen:
            entry.first_seen = datetime.now().isoformat()
        self._entries[q_hash] = entry
        self.save()
        log.info("cached question %s (type=%s, now %d entries)", q_hash[:12], entry.question_type, len(self._entries))

    def __len__(self) -> int:
        return len(self._entries)
