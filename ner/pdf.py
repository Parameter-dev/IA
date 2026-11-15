from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pymupdf


def extract_text_from_pdf(
    pdf_path: Union[str, Path], use_ocr: bool = False, ocr_language: str = "eng"
) -> str:
    """Extract the text of a PDF with PyMuPDF (optionally using Tesseract OCR)."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages_text: List[str] = []
    with pymupdf.open(path) as doc:
        for page in doc:
            if use_ocr:
                tp = page.get_textpage_ocr(language=ocr_language, full=True)
                pages_text.append(page.get_text("text", textpage=tp))
            else:
                pages_text.append(page.get_text("text"))

    return "\n".join(pages_text).strip()


def resolve_pdf_path(pdf_dir: Path, doc_id: str) -> Path:
    """Find ``<id>.pdf`` or ``<id>_scanned.pdf`` inside ``pdf_dir``."""
    for candidate in (pdf_dir / f"{doc_id}.pdf", pdf_dir / f"{doc_id}_scanned.pdf"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No PDF found for '{doc_id}' in {pdf_dir}")


def extract_pdf_dataset(
    pdf_dir: Union[str, Path],
    reference_dataset: Optional[List[Dict[str, Any]]] = None,
    use_ocr: bool = False,
    ocr_language: str = "eng",
) -> List[Dict[str, Any]]:
    """Extract the text of every PDF, keeping the order of ``reference_dataset``.

    Keeping the gold order matters: the evaluation compares prediction ``i``
    with gold document ``i``.
    """
    directory = Path(pdf_dir)
    records: List[Dict[str, Any]] = []

    if reference_dataset is not None:
        for item in reference_dataset:
            pdf_path = resolve_pdf_path(directory, item["id"])
            text = extract_text_from_pdf(
                pdf_path, use_ocr=use_ocr, ocr_language=ocr_language
            )
            records.append({"id": item["id"], "text": text})
    else:
        for pdf_path in sorted(directory.glob("*.pdf")):
            doc_id = pdf_path.stem.replace("_scanned", "")
            records.append(
                {
                    "id": doc_id,
                    "text": extract_text_from_pdf(
                        pdf_path, use_ocr=use_ocr, ocr_language=ocr_language
                    ),
                }
            )

    return records
