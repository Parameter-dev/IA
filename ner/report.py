"""Build the results table (README section 3) from the prediction files in results/."""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ner.evaluation import evaluate_ner_relaxed, evaluate_ner_strict

PLAIN = "data/plain/validation.jsonl"
PLAIN_SMALL = "data/plain/validation_small.jsonl"

# (input, model, label descriptions, extraction, gold file, predictions file)
EXPERIMENTS: List[Tuple[str, str, str, str, str, str]] = [
    ("Plain text", "`gliner_small-v2.5`", "No", "—", PLAIN, "results/plain/gliner_small_validation.jsonl"),
    ("Plain text", "`gliner_small-v2.5`", "Yes", "—", PLAIN, "results/plain/gliner_small_descriptions_validation.jsonl"),
    ("Plain text", "`gliner_large-v2.5`", "No", "—", PLAIN, "results/plain/gliner_large_validation.jsonl"),
    ("Clean PDF", "`gliner_small-v2.5`", "No", "PyMuPDF", PLAIN, "results/pdf_clean/gliner_small_validation.jsonl"),
    ("Scanned PDF", "`gliner_small-v2.5`", "No", "PyMuPDF", PLAIN, "results/pdf_scanned/gliner_small_validation.jsonl"),
    ("Scanned PDF", "`gliner_small-v2.5`", "No", "PyMuPDF + OCR", PLAIN, "results/pdf_scanned/gliner_small_ocr_validation.jsonl"),
    ("Plain text (validation_small)", "LLM chat (no API)", "Yes (prompt)", "—", PLAIN_SMALL, "results/llm/llm_validation.jsonl"),
]

START = "<!-- RESULTS_TABLE_START -->"
END = "<!-- RESULTS_TABLE_END -->"


def _has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _fmt(m: Dict[str, Any]) -> str:
    return f"{m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f}"


def build_results_table(root: Path) -> Tuple[str, List[str]]:
    """Return (markdown table, list of experiments that could not be evaluated)."""
    lines = [
        "| Input | Model | Label descriptions | Extraction | Evaluation | P | R | F1 |",
        "| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |",
    ]
    missing: List[str] = []

    for inp, model, desc, extraction, gold_rel, pred_rel in EXPERIMENTS:
        gold, pred = root / gold_rel, root / pred_rel
        if not _has_content(gold) or not _has_content(pred):
            missing.append(f"{pred_rel} (gold: {gold_rel})")
            continue
        strict = evaluate_ner_strict(gold, pred, print_results=False)
        relaxed = evaluate_ner_relaxed(gold, pred, print_results=False)
        rows = [
            ("Strict", strict["micro"]),
            ("Relaxed occurrences", relaxed["occurrences"]["micro"]),
            ("Relaxed unique", relaxed["unique"]["micro"]),
        ]
        for name, micro in rows:
            lines.append(f"| {inp} | {model} | {desc} | {extraction} | {name} | {_fmt(micro)} |")

    return "\n".join(lines), missing


def update_readme(root: Path, table: str) -> bool:
    """Replace the table between the README markers. Returns True if updated."""
    readme = root / "README.md"
    content = readme.read_text(encoding="utf-8")
    if START not in content or END not in content:
        return False
    new = re.sub(
        re.escape(START) + r".*?" + re.escape(END),
        lambda _: f"{START}\n{table}\n{END}",
        content,
        flags=re.DOTALL,
    )
    readme.write_text(new, encoding="utf-8")
    return True


def main(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parent.parent
    table, missing = build_results_table(root)
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    if missing:
        print("\nWARNING - not evaluated (file missing or empty):")
        for item in missing:
            print(f"  - {item}")
    if update_readme(root, table):
        print("\nREADME.md results table updated.")
    else:
        print("\nREADME.md markers not found; table only written to results/summary.md")
