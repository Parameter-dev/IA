from pathlib import Path
from typing import List, Dict, Any
import pymupdf


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo PDF: {pdf_path}")

    text_pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text_pages.append(page.get_text("text"))

    return "\n".join(text_pages)


def extract_pdf_dataset(
        jsonl_records: List[Dict[str, Any]],
        pdf_dir: str | Path
) -> List[Dict[str, Any]]:
    pdf_dir = Path(pdf_dir)
    extracted_dataset = []

    for record in jsonl_records:
        doc_id = record["id"]
        pdf_file = pdf_dir / f"{doc_id}.pdf"

        extracted_text = extract_text_from_pdf(pdf_file)

        new_record = dict(record)
        new_record["text"] = extracted_text
        extracted_dataset.append(new_record)

    return extracted_dataset