"""
Cross-encoder reranker (inference).

`CrossEncoder` wraps a fine-tuned `AutoModelForSequenceClassification`
(num_labels=1) that scores a `[context | candidate profile]` pair as one relevance
 logit -- a generic pair-scorer that owns no entity-linking preprocessing.
"""

import json
import math
import os

import torch
from huggingface_hub import snapshot_download
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

from lvnel.linker import CandidateRanker, marked_context, entity_profile


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model_path(model_name_or_path):
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    local_model_dir = snapshot_download(model_name_or_path)
    return local_model_dir


class CrossEncoder:
    """Load a trained reranker and score (context, profiles) -> softmax."""

    def __init__(self, model_dir, device=None, batch_size=64, max_len=512):
        self._torch = torch
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.max_len = max_len
        model_dir = get_model_path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1).to(self.device).eval()

    def logits(self, context, profiles):
        """Raw per-candidate logits for one mention (no normalization)."""
        torch = self._torch
        out = []
        for i in range(0, len(profiles), self.batch_size):
            chunk = profiles[i:i + self.batch_size]
            enc = self.tok([context] * len(chunk), chunk, truncation="only_first",
                           max_length=self.max_len, padding=True,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                out += self.model(**enc).logits.squeeze(-1).float().cpu().tolist()
        return out

    def score(self, context, profiles):
        """Per-group softmax over a mention's candidate profiles."""
        if not profiles:
            return []
        if len(profiles) == 1:
            return [1.0]
        lg = self.logits(context, profiles)
        m = max(lg)                                    # stable softmax
        exps = [math.exp(x - m) for x in lg]
        z = sum(exps)
        return [e / z for e in exps]


class CrossEncoderRanker(CandidateRanker):
    """Trained cross-encoder reranker: ranks by the per-mention softmax."""

    def __init__(self, model_dir, device=None, batch_size=64, max_len=512, window=400, markers=("[E]", "[/E]")):
        model_dir = get_model_path(model_dir)
        self.window = window
        self.markers = markers
        self.enc = CrossEncoder(model_dir, device=device, batch_size=batch_size, max_len=max_len)

    def scores(self, text, start, end, candidates):
        ctx = marked_context(text, start, end, self.window, self.markers)
        return self.enc.score(ctx, [entity_profile(c) for c in candidates])


class NilCrossEncoderRanker(CrossEncoderRanker):
    """NIL-aware cross-encoder: same encoder, but each logit is an absolute
    "is this a match?" score (trained with a fixed NIL anchor). Ranks by raw
    logits and abstains when the best candidate falls below `nil_threshold`."""

    def __init__(self, model_dir, nil_threshold=None, **kw):
        model_dir = get_model_path(model_dir)
        super().__init__(model_dir, **kw)
        ranker_config = json.load(open(os.path.join(model_dir, "ranker_config.json")))
        self.nil_threshold = nil_threshold if nil_threshold is not None else ranker_config["nil_threshold"]

    def scores(self, text, start, end, candidates):
        ctx = marked_context(text, start, end, self.window, self.markers)
        return self.enc.logits(ctx, [entity_profile(c) for c in candidates])

    def rank(self, text, start, end, candidates):
        # Every candidate needs its real logit so the NIL threshold applies
        # uniformly (no single-candidate shortcut).
        if not candidates:
            return []
        s = self.scores(text, start, end, candidates)
        dc = [c.get("doc_count") or 0 for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (s[i], dc[i]), reverse=True)
        return [(candidates[i], s[i]) for i in order]

    def is_nil(self, ranked):
        return bool(ranked) and ranked[0][1] < self.nil_threshold
