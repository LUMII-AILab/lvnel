"""
Gradio visual demo for the entity linker.

Paste text -> highlighted mention spans (hover a span to see all candidates, the
selected entity first, with its score) -> a table of the distinct linked entities.

The ranking method is switchable from the dropdown; each method's linker is
built lazily and cached, all sharing one knowledge base and mention matcher.
"""

import html
import logging
from functools import cache

import gradio as gr

from lvnel.ce_ranker import CrossEncoderRanker, NilCrossEncoderRanker
from lvnel.entity_mention_matcher import EntityMentionMatcher
from lvnel.linker import KnowledgeBase, EntityLinker, PriorRanker

ENTITIES = "data/wiki_v2.entities.jsonl"
SAMPLE =("Beļģija un Vācija ir Eiropas valstis. Ķīnas Tautas Republika atrodas "
          "Āzijā. Bijušais bibliotēkas vadītājs Andris Vilks. Finanšu ministrs ir Andris Vilks.")
RANKERS = {
    "prior": lambda: PriorRanker(),
    "cross-encoder": lambda: CrossEncoderRanker("AiLab-IMCS-UL/lv-ce_ranker-wiki_v2-m8"),
    "cross-encoder-nil": lambda: NilCrossEncoderRanker("AiLab-IMCS-UL/lv-ce_nil_ranker-wiki_v2-m18"),
}
DEFAULT_RANKER = "cross-encoder-nil"

CSS = """
.doc { line-height: 2.2; font-size: 15px; }
.m { background: #ffe9a8; border-radius: 3px; padding: 0 2px; position: relative; cursor: help; }
.m.amb { background: #ffd27f; }
.m.nil { background: #eee; color: #999; text-decoration: line-through; }
.tip .nil-note { color: #a00; font-size: 11px; padding: 2px 6px; }
.m:hover .tip { display: block; }
.tip { display: none; position: absolute; left: 0; top: 1.9em; z-index: 1000;
       width: 340px; background: #fff; border: 1px solid #888; border-radius: 4px;
       padding: 4px; box-shadow: 0 2px 10px rgba(0,0,0,.25); white-space: normal;
       text-align: left; font-size: 13px; line-height: 1.4; }
.cand { padding: 4px 6px; border-bottom: 1px solid #eee; }
.cand.sel { background: #e7f5e7; }
.cand .title { color: #06c; font-size: 11px; }
.cand .score { float: right; color: #888; font-size: 11px; }
.cand .desc { color: #555; font-size: 11px; }
.ent { border-collapse: collapse; font-size: 13px; }
.ent td, .ent th { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
.ent th { background: #f4f4f4; }
.nel-out, .nel-out * { overflow: visible !important; }
"""


@cache
def _pipeline():
    """Knowledge base + mention matcher, loaded once and shared by every method."""
    return KnowledgeBase.load(ENTITIES), EntityMentionMatcher.from_entities_jsonl(ENTITIES)


@cache
def get_linker(ranker_name):
    """The linker for one method, built (and cached) on first use."""
    knowledge_base, matcher = _pipeline()
    return EntityLinker(knowledge_base, matcher, RANKERS[ranker_name]())


def _esc(s):
    return html.escape(s or "")


def _cand_html(c):
    cls = "cand sel" if c["selected"] else "cand"
    return (f"<div class='{cls}'>"
            f"<span class='score'>{c['score']}</span>"
            f"<b>{_esc(c['label'])}</b> <span class='title'>{_esc(c['title'])}</span>"
            f"<div class='desc'>{_esc(c['description'])}</div></div>")


def _mention_html(text, m):
    surface = _esc(text[m["start"]:m["end"]])
    if m.get("nil"):
        cls = " nil"
        note = "<div class='nil-note'>→ NIL (no matching entity)</div>"
    else:
        cls = " amb" if len(m["candidates"]) > 1 else ""
        note = ""
    tip = ("<span class='tip'>" + note
           + "".join(_cand_html(c) for c in m["candidates"]) + "</span>")
    return f"<span class='m{cls}'>{surface}{tip}</span>"


def render_text(result):
    text, pos, out = result["text"], 0, []
    for m in sorted(result["mentions"], key=lambda m: m["start"]):
        if m["start"] < pos:
            continue
        out.append(_esc(text[pos:m["start"]]))
        out.append(_mention_html(text, m))
        pos = m["end"]
    out.append(_esc(text[pos:]))
    body = "".join(out).replace("\n", "<br>")
    return f"<div class='doc'>{body}</div>"


def render_entities(result):
    ents = result["entities"]
    if not ents:
        return "<p><i>No linked entities.</i></p>"
    rows = "".join(
        f"<tr><td><b>{_esc(e['label'])}</b></td><td>{_esc(e['title'])}</td>"
        f"<td>{e['count']}</td><td>{_esc(e['type'])}</td>"
        f"<td>{_esc(e['description'])}</td></tr>" for e in ents)
    return ("<table class='ent'><thead><tr><th>entity</th><th>title</th>"
            "<th>mentions</th><th>type</th><th>description</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def analyze(text, method):
    try:
        result = get_linker(method).analyze(text or "")
    except Exception as e:
        return f"<p style='color:#c00'>{_esc(method)}: {_esc(str(e))}</p>", ""
    return render_text(result), render_entities(result)


def build_ui():
    with gr.Blocks(title="Latvian NEL") as demo:
        gr.Markdown("## Latvian Named Entity Linking")
        with gr.Row():
            text = gr.Textbox(SAMPLE, lines=6, label="Text", scale=4)
            method = gr.Dropdown(list(RANKERS.keys()), value=DEFAULT_RANKER, label="Ranker", scale=1)
        btn = gr.Button("Analyze", variant="primary")
        marked = gr.HTML(label="Linked text", elem_classes="nel-out")
        gr.Markdown("### Document entities")
        entities = gr.HTML(elem_classes="nel-out")
        btn.click(analyze, [text, method], [marked, entities])
    return demo


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    build_ui().launch(css=CSS)
