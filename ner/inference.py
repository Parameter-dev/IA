from typing import Any, Dict, List, Optional, Union

from gliner import GLiNER
from tqdm import tqdm

# Natural-language descriptions used for the "label descriptions" experiments
# (GLiNER receives the description instead of the bare label name; the original
# label is restored in the output so the evaluation still works).
LABEL_DESCRIPTIONS: Dict[str, str] = {
    "PERSON_NAME": "Full name of a natural person (first name and surnames)",
    "ID_CARD_NUMBER": "National identity document number of a person (DNI or NIE)",
    "ADDRESS": "Full postal address: street, number, floor, postal code and city",
    "REGISTRATION_NUMBER": "Administrative file or registration number assigned to a request",
    "DATE": "Calendar date",
    "AMOUNT": "Monetary amount with its currency",
    "ORGANIZATION": "Name of a public body, office, company or institution",
}


def load_gliner_model(
    model_name: str = "gliner-community/gliner_small-v2.5",
    device: Optional[str] = None,
) -> GLiNER:
    """Load a GLiNER model. If ``device`` is None, CUDA is used when available."""
    model = GLiNER.from_pretrained(model_name)
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def predict_entities(
    model: GLiNER,
    text: str,
    labels: Union[List[str], Dict[str, str]],
    threshold: float = 0.5,
    flat_ner: bool = True,
) -> List[Dict[str, Any]]:
    """Predict entities in ``text``.

    ``labels`` can be a list of label names or a ``{label: description}`` dict.
    With a dict, the descriptions are sent to the model and mapped back to the
    original label names in the output.
    """
    if not text or not text.strip():
        return []

    if isinstance(labels, dict):
        desc_to_label = {desc: lbl for lbl, desc in labels.items()}
        query_labels = list(labels.values())
    else:
        desc_to_label = {}
        query_labels = list(labels)

    raw_preds = model.predict_entities(
        text, query_labels, threshold=threshold, flat_ner=flat_ner
    )
    return [
        {
            "start": ent["start"],
            "end": ent["end"],
            "text": ent["text"],
            "label": desc_to_label.get(ent["label"], ent["label"]),
            "score": float(ent.get("score", 1.0)),
        }
        for ent in raw_preds
    ]


def get_dataset_labels(dataset: List[Dict[str, Any]]) -> List[str]:
    """Return the sorted set of entity labels found in a gold dataset."""
    labels_set = set()
    for record in dataset:
        if "labels" in record and isinstance(record["labels"], list):
            labels_set.update(record["labels"])
        for ent in record.get("entities", []):
            if "label" in ent:
                labels_set.add(ent["label"])
    return sorted(labels_set)


def predict_dataset(
    model: GLiNER,
    dataset: List[Dict[str, Any]],
    labels: Optional[Union[List[str], Dict[str, str]]] = None,
    threshold: float = 0.5,
    flat_ner: bool = True,
) -> List[Dict[str, Any]]:
    """Run GLiNER over every record (``id`` + ``text``) and return predictions."""
    if labels is None:
        labels = get_dataset_labels(dataset)

    predictions: List[Dict[str, Any]] = []
    for record in tqdm(dataset, desc="Running GLiNER inference"):
        text = record.get("text", "")
        pred_ents = predict_entities(
            model=model,
            text=text,
            labels=labels,
            threshold=threshold,
            flat_ner=flat_ner,
        )
        predictions.append({"id": record["id"], "text": text, "entities": pred_ents})
    return predictions
