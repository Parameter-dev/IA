from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pymupdf


def extract_text_from_pdf(
    pdf_path: Union[str, Path], use_ocr: bool = False, ocr_language: str = "eng"
) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo PDF: {path}")

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
    candidates = [
        pdf_dir / f"{doc_id}.pdf",
        pdf_dir / f"{doc_id}_scanned.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró el PDF para el id '{doc_id}' en {pdf_dir}"
    )


def extract_pdf_dataset(
    pdf_dir: Union[str, Path],
    reference_dataset: Optional[List[Dict[str, Any]]] = None,
    use_ocr: bool = False,
) -> List[Dict[str, Any]]:
    directory = Path(pdf_dir)
    extracted_records: List[Dict[str, Any]] = []

    if reference_dataset is not None:
        for record in reference_dataset:
            doc_id = record["id"]
            pdf_path = resolve_pdf_path(directory, doc_id)
            text = extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
            new_record = dict(record)
            new_record["text"] = text
            extracted_records.append(new_record)
    else:
        for pdf_path in sorted(directory.glob("*.pdf")):
            doc_id = pdf_path.stem.replace("_scanned", "")
            text = extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
            extracted_records.append({"id": doc_id, "text": text})

    return extracted_records