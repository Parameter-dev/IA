import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


def load_jsonl(
    file_path: Union[str, Path], n: Optional[int] = None
) -> List[Dict[str, Any]]:
    path = Path(file_path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if n is not None and len(records) >= n:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_responses_to_jsonl(
    responses: Iterable[Dict[str, Any]], output_path: Union[str, Path]
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in responses:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")