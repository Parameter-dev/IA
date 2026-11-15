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
| Plain text | `gliner_small-v2.5` | No | — | Strict | 0.77 | 0.87 | 0.82 |
| Plain text | `gliner_small-v2.5` | No | — | Relaxed occurrences | 0.77 | 0.87 | 0.82 |
| Plain text | `gliner_small-v2.5` | No | — | Relaxed unique | 0.78 | 0.89 | 0.83 |
| Plain text | `gliner_small-v2.5` | Yes | — | Strict | 0.98 | 0.28 | 0.43 |
| Plain text | `gliner_small-v2.5` | Yes | — | Relaxed occurrences | 0.98 | 0.28 | 0.43 |
| Plain text | `gliner_small-v2.5` | Yes | — | Relaxed unique | 0.98 | 0.43 | 0.59 |
| Plain text | `gliner_large-v2.5` | No | — | Strict | 1.00 | 0.91 | 0.96 |
| Plain text | `gliner_large-v2.5` | No | — | Relaxed occurrences | 1.00 | 0.91 | 0.96 |
| Plain text | `gliner_large-v2.5` | No | — | Relaxed unique | 1.00 | 0.91 | 0.95 |
| Clean PDF | `gliner_small-v2.5` | No | PyMuPDF | Strict | 0.53 | 0.60 | 0.56 |
| Clean PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed occurrences | 0.77 | 0.87 | 0.82 |
| Clean PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed unique | 0.78 | 0.89 | 0.83 |
| Scanned PDF | `gliner_small-v2.5` | No | PyMuPDF | Strict | 0.00 | 0.00 | 0.00 |
| Scanned PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed occurrences | 0.00 | 0.00 | 0.00 |
| Scanned PDF | `gliner_small-v2.5` | No | PyMuPDF | Relaxed unique | 0.00 | 0.00 | 0.00 |
| Plain text (validation_small) | LLM chat (no API) | Yes (prompt) | — | Strict | 0.99 | 0.97 | 0.98 |
| Plain text (validation_small) | LLM chat (no API) | Yes (prompt) | — | Relaxed occurrences | 0.99 | 0.97 | 0.98 |
| Plain text (validation_small) | LLM chat (no API) | Yes (prompt) | — | Relaxed unique | 0.98 | 1.00 | 0.99 |
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
* **Small vs. Large GLiNER Models:** `gliner_large-v2.5` substantially outperforms `gliner_small-v2.5` on plain text, raising **Strict F1** from `0.82` (`P=0.77, R=0.87`) to `0.96` (`P=1.00, R=0.91`) and **Relaxed Unique F1** from `0.83` to `0.95`. In particular, `gliner_large-v2.5` resolves full composite `ADDRESS` spans (`F1=0.97` vs. `0.00` in `small`, which fragments addresses into separate street, number, and city spans) and reaches `F1=1.00` on `ORGANIZATION`, though its recall on `REGISTRATION_NUMBER` is lower (`0.37` vs. `0.98`) at the `0.5` threshold.
* **Impact of Label Descriptions:** Replacing short label identifiers with full natural-language sentences in `gliner_small-v2.5` increases **Precision** from `0.77` to `0.98`, but reduces **Recall** from `0.87` to `0.28` (`Strict F1 = 0.43`, `Relaxed Unique F1 = 0.59`) at the fixed `0.5` threshold. While `AMOUNT` (`F1=0.98`) and `DATE` (`F1=0.89`) remain strong, verbose descriptions lower confidence scores below `0.5` for several categories, showing that descriptive label prompts require a lower threshold (e.g., `0.3`).
* **GLiNER vs. LLM Chat:** Evaluated on `validation_small.jsonl` (10 documents), the zero-shot LLM chat achieves **Strict / Relaxed Occurrences F1 = 0.98** (`P=0.99, R=0.97`) and **Relaxed Unique F1 = 0.99** (`P=0.98, R=1.00`), capturing complex `ADDRESS` and `ORGANIZATION` spans accurately once character offsets are aligned programmatically (`ner/llm.py`).

### Most Difficult Entity Types
* **Composite Postal Addresses (`ADDRESS`):** Full addresses (`street, number, floor, postal code, city`) are the hardest entity type for `gliner_small-v2.5` (`F1=0.00`), because the small model splits each address into 2–3 separate fragments (e.g., `'Avenida de Aragón'`, `'102, 2-B'`, `'17001 Girona'`), generating both false positives and false negatives under exact span matching. Both `gliner_large-v2.5` (`F1=0.97`) and the LLM chat (`F1=1.00`) capture the complete address span.
* **Administrative Organizations & Registration Codes (`ORGANIZATION`, `REGISTRATION_NUMBER`):** Long multi-word institutional names (`F1=0.80` in `gliner_small-v2.5`) and alphanumeric file references (`EXP-...`, `ASS-...`, `ENT-...`, `RES-...`) are sensitive to model capacity and threshold selection.

### Plain Text vs. Clean PDF vs. Scanned PDF
* **Plain Text vs. Clean PDF:** Under **Relaxed** evaluation, **Clean PDF** results (`Relaxed Occurrences F1 = 0.82`, `Relaxed Unique F1 = 0.83`) are **identical** to **Plain Text** (`0.82` and `0.83`), proving that `PyMuPDF` recovers the digital text layer without losing entity content. However, under **Strict** evaluation, **Clean PDF** F1 drops from `0.82` to `0.56` (`P=0.53, R=0.60`) because PDF line breaks shift character offsets (`start`, `end`) relative to the plain-text ground truth.
* **Clean PDF vs. Scanned PDF:** **Scanned PDFs** (`test-0001_scanned.pdf` to `test-0010_scanned.pdf`) yield `F1 = 0.00` across all metrics when using standard `PyMuPDF` extraction (`page.get_text("text")`) because the scanned pages are rasterized images with `0` embedded characters. Extracting entities from scanned documents therefore requires an OCR stage (such as Tesseract via `page.get_textpage_ocr()`) prior to NER inference.

### Strict vs. Relaxed Evaluation
* **Strict Evaluation (`start`, `end`, `label`):** Requires exact character-offset and label agreement. It is ideal for plain-text sequence labeling (`0.82` on `small`, `0.96` on `large`), but drops from `0.82` to `0.56` on Clean PDFs purely due to layout-induced offset shifts.
* **Relaxed Occurrences (`label` + normalized text, counting repetitions):** Ignores character offsets (`spans`) and normalizes whitespace while still requiring the model to find every mention in the document. The jump from **Strict F1 (`0.56`)** to **Relaxed Occurrences F1 (`0.82`)** on Clean PDFs proves that GLiNER extracts the exact same mentions from PDFs as from plain text.
* **Relaxed Unique (`label` + normalized text, deduplicated per document):** Counts each unique `(label, text)` pair only once per document. It achieves higher F1 scores across experiments (`0.83` for `gliner_small`, `0.99` for LLM chat) and is the most representative metric for downstream administrative indexing, where extracting each unique person, ID card, or expediente number at least once per document is the primary goal.