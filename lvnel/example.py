import json

from lvnel.linker import EntityLinker
from lvnel.ce_ranker import NilCrossEncoderRanker

linker = EntityLinker.load(
    "data/wiki_v2.entities.jsonl",
    NilCrossEncoderRanker("AiLab-IMCS-UL/lv-ce_nil_ranker-wiki_v2-m18"),
)

text = "Bijušais bibliotēkas vadītājs Andris Vilks. Finanšu ministrs Andris Vilks."
for m in linker.link(text):
    print(m) # span -> chosen title (+ candidates)
"""
[   9:  20] 'bibliotēkas'            -> Q7075 Bibliotēka (bibliotēka)
[  21:  29] 'vadītājs'               -> Q1162163 Direktors (direktors)
[  30:  42] 'Andris Vilks'           -> Q16351421 Andris Vilks (filologs) (Andris Vilks) [3 cand]
[  52:  60] 'ministrs'               -> Q83307 Ministrs (ministrs)
[  61:  73] 'Andris Vilks'           -> Q2671932 Andris Vilks (ekonomists) (Andris Vilks) [3 cand]
"""

result = linker.analyze(text)  # JSON: per-mention ranked candidates + doc entities
print(json.dumps(result, indent=2, ensure_ascii=False))
