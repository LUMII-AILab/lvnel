# Latvian Named Entity Linking
Links entity mentions in Latvian text to a Wikidata-derived knowledge base (~131k entities keyed by Wikipedia title).

## Setup

Run everything from the repo root. Assumes `uv`, `wget`, and `zstd` are installed.

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
./data/download.sh
```

The cross-encoder rankers download their model weights from the Hugging Face Hub
automatically on first use; only the knowledge base above is fetched manually.

## Pipeline

Pipeline (`lvnel/linker.py`):
1. `EntityMentionMatcher` finds mention spans and entity candidates based on entity aliases and inflections.
2. `KnowledgeBase` loads candidate entity descriptions and popularity counts.
3. `CandidateRanker` scores the candidates for each mention.
4. `EntityLinker` returns the chosen entity for each mention.

## Usage

Example usage: [lvnel/example.py](example.py)

Available rankers:
- `PriorRanker()` — popularity baseline, no model (the default if none is passed).
- `CrossEncoderRanker("AiLab-IMCS-UL/lv-ce_ranker-wiki_v2-m8")` — context reranker.
- `NilCrossEncoderRanker("AiLab-IMCS-UL/lv-ce_nil_ranker-wiki_v2-m18")` — reranker that can abstain (NIL).

Quick check with the popularity baseline (no model download):

```bash
python -m lvnel.linker
```

Web demo (switch rankers from the dropdown):

```bash
python -m lvnel.demo        # open the printed local URL
```
