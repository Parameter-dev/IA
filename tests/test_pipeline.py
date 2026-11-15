"""Smoke tests for the NER pipeline (no model download, no course data needed)."""

import pymupdf

from ner.evaluation import evaluate_ner_relaxed, evaluate_ner_strict
from ner.inference import get_dataset_labels, predict_dataset
from ner.llm import (
    align_entities_to_text,
    llm_records_to_predictions,
    parse_llm_response,
)
from ner.pdf import extract_pdf_dataset
from ner.report import build_results_table
from ner.utils import load_jsonl, save_responses_to_jsonl

TEXT = "Ana Perez vive en Girona. Ana Perez firmo el 2018-04-09."
GOLD = [
    {
        "id": "test-0001",
        "text": TEXT,
        "entities": [
            {"start": 0, "end": 9, "text": "Ana Perez", "label": "PERSON_NAME"},
            {"start": 18, "end": 24, "text": "Girona", "label": "ADDRESS"},
            {"start": 26, "end": 35, "text": "Ana Perez", "label": "PERSON_NAME"},
            {"start": 43, "end": 53, "text": "2018-04-09", "label": "DATE"},
        ],
    }
]


class FakeModel:
    """Mimics GLiNER.predict_entities; echoes the labels it receives."""

    def __init__(self):
        self.received_labels = None

    def predict_entities(self, text, labels, threshold=0.5, flat_ner=True):
        self.received_labels = labels
        return [{"start": 0, "end": 9, "text": "Ana Perez", "label": labels[0], "score": 0.9}]


def test_load_jsonl_n_and_roundtrip(tmp_path):
    path = tmp_path / "x.jsonl"
    save_responses_to_jsonl([{"id": i} for i in range(5)], path)
    assert len(load_jsonl(path)) == 5
    assert load_jsonl(path, n=2) == [{"id": 0}, {"id": 1}]


def test_get_dataset_labels():
    assert get_dataset_labels(GOLD) == ["ADDRESS", "DATE", "PERSON_NAME"]


def test_predict_dataset_with_descriptions_restores_labels():
    model = FakeModel()
    desc = {"PERSON_NAME": "a person", "DATE": "a date"}
    preds = predict_dataset(model, GOLD, labels=desc)
    assert model.received_labels == ["a person", "a date"]
    assert preds[0]["entities"][0]["label"] == "PERSON_NAME"


def test_evaluation_with_paths(tmp_path):
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    save_responses_to_jsonl(GOLD, gold_path)
    save_responses_to_jsonl(predict_dataset(FakeModel(), GOLD, labels=["PERSON_NAME"]), pred_path)
    strict = evaluate_ner_strict(gold_path, pred_path, print_results=False)
    relaxed = evaluate_ner_relaxed(gold_path, pred_path, print_results=False)
    assert strict["micro"]["true_positives"] == 1
    assert strict["micro"]["false_negatives"] == 3
    assert relaxed["unique"]["micro"]["true_positives"] == 1


def test_pdf_extraction_follows_reference_order(tmp_path):
    for doc_id, name in (("test-0002", "clean"), ("test-0001", "scanned")):
        suffix = "_scanned" if name == "scanned" else ""
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), f"hello {doc_id}")
        pdf.save(tmp_path / f"{doc_id}{suffix}.pdf")
    ref = [{"id": "test-0001"}, {"id": "test-0002"}]
    docs = extract_pdf_dataset(tmp_path, ref)
    assert [d["id"] for d in docs] == ["test-0001", "test-0002"]
    assert "hello test-0001" in docs[0]["text"]


def test_llm_alignment_repeated_mentions_and_hallucinations():
    ents = [
        {"text": "Ana Perez", "label": "PERSON_NAME"},
        {"text": "Ana Perez", "label": "PERSON_NAME"},
        {"text": "Madrid", "label": "ADDRESS"},  # not in text -> dropped
        {"text": "Girona", "label": "BAD_LABEL"},  # invalid label -> dropped
    ]
    aligned = align_entities_to_text(TEXT, ents, valid_labels=["PERSON_NAME", "ADDRESS"])
    assert [(e["start"], e["end"]) for e in aligned] == [(0, 9), (26, 35)]


def test_llm_parse_and_predictions_follow_gold_order():
    raw = (
        "```json\n"
        '{"id": "test-0001", "entities": [{"text": "Girona", "label": "ADDRESS"}]},\n'
        "```\n"
    )
    preds = llm_records_to_predictions(parse_llm_response(raw), GOLD)
    assert preds[0]["entities"] == [
        {"start": 18, "end": 24, "text": "Girona", "label": "ADDRESS"}
    ]


def test_report_skips_empty_files(tmp_path):
    (tmp_path / "data" / "plain").mkdir(parents=True)
    (tmp_path / "data" / "plain" / "validation.jsonl").write_text("")
    table, missing = build_results_table(tmp_path)
    assert len(missing) == 7
    assert table.count("\n") == 1  # header only
