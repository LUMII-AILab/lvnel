"""
Named Entity Linking pipeline (inference).
- EntityMentionMatcher   text  -> spans + candidate titles
- KnowledgeBase          title -> entity profile + popularity prior
- CandidateRanker        choose one title per mention

A CandidateRanker implements `scores(text, start, end, candidates) -> [float]`;
the base turns those into a ranking (argmax, or alpha-blended with the popularity
prior).
"""

import json
from dataclasses import dataclass

from lvnel.entity_mention_matcher import EntityMentionMatcher


class KnowledgeBase:
    """title -> entity dict."""

    def __init__(self, entities):
        self.entities = entities

    @classmethod
    def load(cls, path):
        entities = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                entities[e["title"]] = e
        return cls(entities)

    def get(self, title):
        return self.entities.get(title)

    def candidates(self, candidate_titles):
        return [self.entities[t] for t in candidate_titles if t in self.entities]


# Shared context/profile builders -- identical to training, so inference matches.
def _trim_window(text, start, end, window):
    left = text[max(0, start - window):start]
    right = text[end:end + window]
    if " " in left:
        left = left[left.index(" ") + 1:]
    if " " in right:
        right = right[:right.rindex(" ")]
    return left, text[start:end], right


def marked_context(text, start, end, window=400, markers=("[E]", "[/E]")):
    left, span, right = _trim_window(text, start, end, window)
    o, c = markers
    return f"{left} {o} {span} {c} {right}"


def plain_context(text, start, end, window=500):
    left, span, right = _trim_window(text, start, end, window)
    return f"{left} {span} {right}"


def entity_profile(e, use_lead=True):
    desc = (e.get("desc_wiki") or "").strip()
    lead = (e.get("lead") or "").strip() if use_lead else ""
    return " ".join(p for p in (desc, lead) if p)


class CandidateRanker:
    """Pick one candidate title for a mention.

    Subclasses implement scores() -> one number per candidate (higher = better).
    rank() is shared: alpha=None ranks by raw score (doc_count tie-break); alpha
    set blends `alpha*score + (1-alpha)*prior` (prior = doc_count normalized over
    the group), matching the blend used in training.
    """

    alpha = None

    def scores(self, text, start, end, candidates):
        raise NotImplementedError

    def rank(self, text, start, end, candidates):
        """[(candidate, final_score)] sorted best-first."""
        if not candidates:
            return []
        if len(candidates) == 1:
            return [(candidates[0], 1.0)]
        s = self.scores(text, start, end, candidates)
        dc = [c.get("doc_count") or 0 for c in candidates]
        if self.alpha is None:
            final = list(s)
        else:
            total = sum(dc) or 1
            a = self.alpha
            final = [a * s[i] + (1 - a) * dc[i] / total for i in range(len(candidates))]
        order = sorted(range(len(candidates)), key=lambda i: (final[i], dc[i]), reverse=True)
        return [(candidates[i], final[i]) for i in order]

    def pick(self, text, start, end, candidates):
        ranked = self.rank(text, start, end, candidates)
        if not ranked or self.is_nil(ranked):
            return None
        return ranked[0][0]["title"]

    def is_nil(self, ranked):
        """Whether the mention links to no candidate. Only NIL-aware methods override."""
        return False


class PriorRanker(CandidateRanker):
    """Popularity baseline: pick the most common entity (no context, no model)."""

    def scores(self, text, start, end, candidates):
        return [c.get("doc_count") or 0 for c in candidates]


@dataclass
class LinkedMention:
    start: int
    end: int
    surface: str
    title: str               # chosen entity title (None if no candidates / NIL)
    id: str
    label: str               # chosen entity's label, for readability
    candidate_titles: list

    def __str__(self):
        n = len(self.candidate_titles)
        amb = f" [{n} cand]" if n > 1 else ""
        return f"[{self.start:4d}:{self.end:4d}] {self.surface!r:24} -> {self.id} {self.title} ({self.label}){amb}"


class EntityLinker:
    """text -> mentions -> linked entities, with a swappable candidate ranker."""

    def __init__(self, knowledge_base: KnowledgeBase, mention_matcher: EntityMentionMatcher, ranker=None):
        self.knowledge_base = knowledge_base
        self.matcher = mention_matcher
        self.ranker = ranker or PriorRanker()

    @classmethod
    def load(cls, entities_jsonl, ranker=None):
        """Build the full pipeline: knowledge base + mention matcher + ranker."""
        knowledge_base = KnowledgeBase.load(entities_jsonl)
        matcher = EntityMentionMatcher.from_entities_jsonl(entities_jsonl)
        return cls(knowledge_base, matcher, ranker)

    def link(self, text, resolve="longest"):
        out = []
        for m in self.matcher.detect(text, resolve=resolve):
            cands = self.knowledge_base.candidates(m.candidate_titles)
            ranked = self.ranker.rank(text, m.start, m.end, cands)
            chosen = None if (not ranked or self.ranker.is_nil(ranked)) else ranked[0][0]
            out.append(LinkedMention(
                m.start, m.end, m.surface,
                chosen["title"] if chosen else None,
                chosen["id"] if chosen else None,
                (chosen.get("label") or chosen["title"]) if chosen else None,
                [c["title"] for c in cands],
            ))
        return out

    def analyze(self, text, resolve="longest"):
        """JSON-serializable result for UIs: per-mention ranked candidates
        (chosen first, with score) + the distinct linked entities."""
        mentions, entities = [], {}
        for m in self.matcher.detect(text, resolve=resolve):
            ranked = self.ranker.rank(text, m.start, m.end, self.knowledge_base.candidates(m.candidate_titles))
            nil = self.ranker.is_nil(ranked)
            cands = [{
                "id": c["id"],
                "title": c["title"],
                "label": c.get("label") or c["title"],
                "description": c.get("description"),
                "type": c.get("type"),
                "doc_count": c.get("doc_count") or 0,
                "score": round(float(score), 4),
                "selected": i == 0 and not nil,
            } for i, (c, score) in enumerate(ranked)]
            chosen = None if (nil or not cands) else cands[0]
            mentions.append({"start": m.start, "end": m.end, "surface": m.surface,
                             "title": chosen["title"] if chosen else None,
                             "id": chosen["id"] if chosen else None,
                             "nil": nil, "candidates": cands})
            if chosen:
                e = entities.setdefault(chosen["title"], {**chosen, "count": 0})
                e["count"] += 1
        doc_entities = sorted(entities.values(), key=lambda e: (-e["count"], -e["doc_count"]))
        return {"text": text, "mentions": mentions, "entities": doc_entities}


if __name__ == "__main__":
    linker = EntityLinker.load("data/wiki_v2.entities.jsonl")
    text = "Beļģija un Vācija ir Eiropas valstis. Ķīnas Tautas Republika atrodas Āzijā."
    print(*linker.link(text), sep="\n")
