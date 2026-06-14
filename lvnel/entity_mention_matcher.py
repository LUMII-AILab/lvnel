"""
Entity mention matching and candidate generation.

Builds an Aho-Corasick automaton from entity surface forms (title + aliases +
inflections) and finds every span in free text, attaching the candidate titles
whose surface matched. Ranking happens later in `linker.py`.

Matching is case-insensitive and constrained to word boundaries.
"""

import json
from collections import defaultdict
from dataclasses import dataclass

import ahocorasick


@dataclass
class Mention:
    start: int        # char offset, inclusive
    end: int          # char offset, exclusive
    surface: str      # matched text from the original
    candidate_titles: list


class EntityMentionMatcher:
    """Aho-Corasick dictionary of entity surface forms -> candidate titles."""

    def __init__(self, lowercase=True, min_len=2):
        self.lowercase = lowercase
        self.min_len = min_len                 # skip very short surfaces (noise)
        self._surfaces = defaultdict(set)      # normalized surface -> set(title)
        self._automaton = None

    # ---- building -----------------------------------------------------------
    def _norm(self, surface):
        s = surface.strip()
        return s.lower() if self.lowercase else s

    def add_surface(self, surface, title):
        if not surface:
            return
        key = self._norm(surface)
        if len(key) >= self.min_len:
            self._surfaces[key].add(title)

    def add_entity(self, title, surfaces):
        for s in surfaces:
            self.add_surface(s, title)

    def add_entities_jsonl(self, path):
        """Surfaces = title + aliases + inflections, all mapped to the title."""
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                title = e.get("title")
                if not title:
                    continue
                surfaces = [title, e.get("label")]
                surfaces += e.get("aliases") or []
                surfaces += e.get("inflections") or []
                self.add_entity(title, [s for s in surfaces if s])
                n += 1
        return n

    def build(self):
        a = ahocorasick.Automaton()
        for key, titles in self._surfaces.items():
            a.add_word(key, (len(key), sorted(titles)))   # len for offset recovery
        a.make_automaton()
        self._automaton = a
        return self

    @classmethod
    def from_entities_jsonl(cls, path, **kw):
        g = cls(**kw)
        g.add_entities_jsonl(path)
        return g.build()

    # ---- matching -----------------------------------------------------------
    def detect(self, text, resolve="longest"):
        """Find entity mentions in `text`.
        resolve="longest" keeps leftmost-longest non-overlapping spans (default);
        "all" keeps every boundary-valid match (may overlap).
        """
        if self._automaton is None:
            raise RuntimeError("call build() before detect()")
        haystack = text.lower() if self.lowercase else text
        matches = []
        for end_idx, (klen, titles) in self._automaton.iter(haystack):
            start = end_idx - klen + 1
            end = end_idx + 1
            if self._on_word_boundary(text, start, end):
                matches.append(Mention(start, end, text[start:end], titles))
        if resolve != "all":
            matches = self._leftmost_longest(matches)
        return matches

    @staticmethod
    def _is_word_char(ch):
        return ch.isalnum() or ch == "_"

    def _on_word_boundary(self, text, start, end):
        if start > 0 and self._is_word_char(text[start - 1]) and self._is_word_char(text[start]):
            return False
        if end < len(text) and self._is_word_char(text[end])  and self._is_word_char(text[end - 1]):
            return False
        return True

    @staticmethod
    def _leftmost_longest(matches):
        matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        kept, covered_to = [], -1
        for m in matches:
            if m.start >= covered_to:
                kept.append(m)
                covered_to = m.end
        return kept


def _example():
    g = EntityMentionMatcher()
    g.add_entity("Beļģija", ["Beļģija", "Beļģijas Karaliste"])
    g.add_entity("Vācija", ["Vācija", "Vāciju"])
    g.add_entity("Ķīna", ["Ķīna", "Ķīnas Tautas Republika"])
    g.build()
    text = "Beļģija robežojas ar Vāciju, bet Ķīnas Tautas Republika ir liela."
    for m in g.detect(text):
        print(f"[{m.start:3d}:{m.end:3d}] {m.surface!r:28} -> {m.candidate_titles}")


if __name__ == "__main__":
    _example()
