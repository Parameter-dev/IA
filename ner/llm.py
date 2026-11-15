"""Helpers for Task 4: NER with a chat LLM (no API).

Workflow:
1. ``build_llm_prompt`` creates the text that is pasted into the chat.
2. The chat answer is saved by hand in ``results/llm/raw_llm_response.txt``.
3. ``parse_llm_response`` reads that answer (JSONL, tolerant to code fences).
4. ``align_entities_to_text`` recomputes ``start``/``end`` on the original text,
   because LLMs cannot count characters reliably.
"""

import json
import re
from typing import Any, Dict, List, Optional


def build_llm_prompt(
    docs: List[Dict[str, Any]],
    label_descriptions: Dict[str, str],
) -> str:
    """Build a zero-shot NER prompt for a chat LLM."""
    labels_block = "\n".join(f"- {lbl}: {desc}" for lbl, desc in label_descriptions.items())
    docs_block = "\n".join(
        json.dumps({"id": d["id"], "text": d["text"]}, ensure_ascii=False) for d in docs
    )
    return (
        "You are a Named Entity Recognition system for administrative documents "
        "(Spanish and Catalan).\n"
        "Extract ALL entity mentions from each document using ONLY these labels:\n"
        f"{labels_block}\n\n"
        "Rules:\n"
        "- Copy every entity text EXACTLY as it appears in the document (same characters).\n"
        "- Include every mention, also repeated ones.\n"
        "- Do not compute character offsets.\n"
        "- Answer ONLY with JSONL: one JSON object per line, no explanations, no code fences.\n"
        'Format: {"id": "<document id>", "entities": [{"text": "...", "label": "..."}]}\n\n'
        "Documents (one JSON object per line):\n"
        f"{docs_block}\n"
    )


def parse_llm_response(raw: str) -> List[Dict[str, Any]]:
    """Parse the chat answer: one JSON object per line (code fences are ignored)."""
    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r",\s*$", "", line)
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "id" in obj:
            records.append(obj)
    return records


def align_entities_to_text(
    text: str, entities: List[Dict[str, Any]], valid_labels: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Recompute ``start``/``end`` by locating each entity text in ``text``.

    Repeated mentions are matched to successive occurrences. Entities whose
    text does not appear literally in the document (hallucinations) or whose
    label is not valid are dropped.
    """
    aligned: List[Dict[str, Any]] = []
    cursors: Dict[str, int] = {}

    for ent in entities:
        ent_text = ent.get("text")
        label = ent.get("label")
        if not ent_text or not label:
            continue
        if valid_labels is not None and label not in valid_labels:
            continue

        start = text.find(ent_text, cursors.get(ent_text, 0))
        if start == -1:
            start = text.find(ent_text)
        if start == -1:
            continue

        end = start + len(ent_text)
        cursors[ent_text] = end
        aligned.append({"start": start, "end": end, "text": ent_text, "label": label})

    return aligned


def llm_records_to_predictions(
    parsed: List[Dict[str, Any]],
    reference_dataset: List[Dict[str, Any]],
    valid_labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build prediction records in the gold order (documents the LLM skipped get no entities)."""
    by_id = {rec["id"]: rec for rec in parsed}
    predictions: List[Dict[str, Any]] = []
    for gold in reference_dataset:
        rec = by_id.get(gold["id"], {})
        predictions.append(
            {
                "id": gold["id"],
                "text": gold["text"],
                "entities": align_entities_to_text(
                    gold["text"], rec.get("entities", []), valid_labels
                ),
            }
        )
    return predictions
