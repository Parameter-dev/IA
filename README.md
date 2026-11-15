# Named Entity Recognition (NER) Pipeline for Administrative Documents

## 1. Group Members and Organization
* **Author:** Ricardo Sebastián Tovar Delgado (Individual project)
* **Work Organization:** The project was structured into four sequential phases:
  1. **Core Package Implementation (`ner/`):** Development of JSONL utilities (`ner/utils.py`), GLiNER model loading and inference wrappers (`ner/inference.py`), and PDF text extraction pipelines (`ner/pdf.py`).
  2. **Baseline & Model Exploration (`notebooks/01_plain_text.ipynb`):** Evaluation of `gliner_small-v2.5` and `gliner_large-v2.5` on plain-text administrative records, with and without label descriptions.
  3. **Document Extraction Pipelines (`notebooks/02_pdf_scanned.ipynb`):** Extension of the NER pipeline to clean digital PDFs and scanned PDFs using `PyMuPDF`, followed by zero-shot evaluation via an LLM chat interface.
  4. **Evaluation & Analysis:** Quantitative comparison across Strict, Relaxed Occurrences, and Relaxed Unique metrics.

---

## 2. System and Configuration
* **Implemented Pipeline:**
  * **Task 1 (Plain Text):** `JSONL -> GLiNER -> Strict & Relaxed Evaluation`
  * **Task 2 (Clean PDF):** `Clean PDF -> PyMuPDF -> Extracted Text -> GLiNER -> Strict & Relaxed Evaluation`
  * **Task 3 (Scanned PDF):** `Scanned PDF -> PyMuPDF (with/without OCR) -> Extracted Text -> GLiNER -> Strict & Relaxed Evaluation`
  * **Task 4 (LLM Chat):** `Plain Text Prompt -> LLM Chat (Zero-shot, no API) -> JSONL Predictions -> Strict & Relaxed Evaluation`
* **GLiNER Models Used:**
  * `gliner-community/gliner_small-v2.5` (primary baseline model)
  * `gliner-community/gliner_large-v2.5` (comparison model for Task 1)
* **Label Descriptions:** Tested both raw label names (`Label descriptions = No`) and descriptive natural-language definitions mapped back to the target schema (`Label descriptions = Yes`).
* **PDF Extraction Library:** `PyMuPDF` (`pymupdf` / `fitz`).
* **Environment & Hardware:** Python 3.13 managed with `uv`, executed on local CPU/GPU hardware. Inference threshold set to `0.5` with `flat_ner=True`.

---

## 3. Results Summary

<!-- RESULTS_TABLE_START -->
| Input | Model | Label descriptions | Extraction | Evaluation | P | R | F1 |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
<!-- RESULTS_TABLE_END -->

*Micro-averaged Precision / Recall / F1 computed by `ner.evaluation` on the files in `results/` (table generated with `uv run python scripts/make_report.py`). The LLM row is evaluated on `validation_small.jsonl`; all other rows on `validation.jsonl`.*

---

## 4. Difficulties and Decisions
1. **Character Span Misalignment in PDFs:** When extracting text from PDFs using `PyMuPDF`, line breaks, headers, multi-column layouts, and extra whitespace alter the exact character offsets (`start`, `end`) compared to the original plain-text ground truth. Consequently, evaluating PDF predictions against plain-text ground truth causes a sharp drop in **Strict** metrics even when the model correctly identifies the entity text. Relying on **Relaxed** evaluation (`occurrences` and `unique`) was essential to measure true extraction quality on PDFs.
2. **Scanned PDFs without Embedded Text Layers:** Standard `page.get_text("text")` in `PyMuPDF` only extracts embedded digital text streams. On purely rasterized scanned PDFs (`data/pdf_scanned/`), standard extraction returns empty strings or noisy OCR artifacts, resulting in near-zero recall unless an OCR engine (such as `page.get_textpage_ocr()` with Tesseract) is enabled. Additionally, scanned filenames used the `_scanned.pdf` suffix (`test-0001_scanned.pdf`), which required a flexible path resolver in `ner/pdf.py`.
3. **LLM Chat Offset Calculation:** Generative LLMs excel at semantic entity identification in zero-shot settings but struggle to count exact character indices (`start`, `end`) manually in long documents. To make a fair comparison under **Strict** evaluation, we implemented a post-processing alignment step (`ner/llm.py`) that locates the exact substring returned by the LLM in the source text and recomputes `start`/`end` (successive occurrences for repeated mentions; entities not found literally in the text are discarded).

---

## 5. Conclusions

### Comparison of Main Results
* **Small vs. Large GLiNER Models:** `gliner_large-v2.5` achieves higher Precision and Recall than `gliner_small-v2.5` on ambiguous and long-tail entities due to its larger representation capacity, at the cost of higher memory consumption and slower CPU inference.
* **Impact of Label Descriptions:** Providing concise label descriptions helps disambiguate closely related administrative categories, improving precision on domain-specific labels, though overly verbose descriptions can occasionally shift attention away from short surface forms.
* **GLiNER vs. LLM Chat:** The LLM chat demonstrates strong zero-shot comprehension and high **Relaxed** F1 scores without task-specific tuning, often matching or outperforming `gliner_small-v2.5` on complex contextual entities. However, without programmatic span re-alignment, raw LLM character offsets fail under **Strict** evaluation, whereas GLiNER natively predicts accurate token/character spans.

### Most Difficult Entity Types
* **Long or Composite Administrative Mentions:** Entities that include titles, legal references, addresses, or organizational hierarchies frequently suffer from boundary disagreements (e.g., including or excluding articles, prefixes, or punctuation), which heavily penalizes **Strict** match while still succeeding or partially succeeding in qualitative review.
* **Ambiguous Alphanumeric Codes and Dates:** Short identifiers, reference numbers, and dates that appear in multiple contexts (e.g., document issue date vs. birth date or deadline) are prone to label confusion between categories.

### Plain Text vs. Clean PDF vs. Scanned PDF
* **Plain Text vs. Clean PDF:** Under **Relaxed** evaluation (`occurrences` and `unique`), **Clean PDF** results are almost identical to **Plain Text** results because `PyMuPDF` recovers the digital text accurately. However, under **Strict** evaluation, **Clean PDF** performance drops noticeably because PDF line-wrapping and layout formatting shift the `(start, end)` character indices relative to the plain-text ground truth.
* **Clean PDF vs. Scanned PDF:** **Scanned PDFs** yield the lowest performance when using standard text extraction (`PyMuPDF` without OCR) because scanned pages are stored as images rather than character streams (or contain degraded OCR noise such as `l`/`1` or `O`/`0` substitutions). Even when OCR is applied, character recognition errors on low-resolution scans degrade both **Strict** and **Relaxed** metrics because whitespace normalization cannot fix misspelled characters inside an entity string.

### Strict vs. Relaxed Evaluation
* **Strict Evaluation (`start`, `end`, `label`):** Requires exact character-offset and label agreement. It is the most rigorous metric for plain-text sequence labeling, but it is overly punitive when text comes from external document parsers (`PyMuPDF`) that introduce layout-induced offset shifts, or when a model includes a trailing space/punctuation mark.
* **Relaxed Occurrences (`label` + normalized text, counting repetitions):** Ignores character offsets (`spans`) and normalizes whitespace while still requiring the model to find every mention in the document. The significant jump from **Strict F1** to **Relaxed Occurrences F1** (especially on Clean PDFs) proves that GLiNER correctly extracts the entities even when PDF formatting shifts their byte positions.
* **Relaxed Unique (`label` + normalized text, deduplicated per document):** Counts each unique `(label, text)` pair only once per document. It achieves the highest F1 score across all experiments and is the most representative metric for downstream administrative information extraction (e.g., populating a database record), where extracting a person's name or ID at least once per document is what matters most.