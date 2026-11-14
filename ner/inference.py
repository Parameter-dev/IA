from typing import Any, Dict, List, Optional, Union
from gliner import GLiNER
from tqdm import tqdm


def load_gliner_model(
    model_name: str = "gliner-community/gliner_small-v2.5",
    device: Optional[str] = None,
) -> GLiNER:
    model = GLiNER.from_pretrained(model_name)
    if device:
        model = model.to(device)
    return model


def predict_entities(
    model: GLiNER,
    text: str,
    labels: Union[List[str], Dict[str, str]],
    threshold: float = 0.5,
    flat_ner: bool = True,
) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    if isinstance(labels, dict):
        # Usamos las descripciones para guiar al modelo y luego restauramos el nombre de la etiqueta
        desc_to_label = {desc: lbl for lbl, desc in labels.items()}
        query_labels = list(labels.values())
        raw_preds = model.predict_entities(
            text, query_labels, threshold=threshold, flat_ner=flat_ner
        )
        entities = []
        for ent in raw_preds:
            entities.append(
                {
                    "start": ent["start"],
                    "end": ent["end"],
                    "text": ent["text"],
                    "label": desc_to_label.get(ent["label"], ent["label"]),
                    "score": float(ent.get("score", 1.0)),
                }
            )
        return entities
    else:
        raw_preds = model.predict_entities(
            text, labels, threshold=threshold, flat_ner=flat_ner
        )
        return [
            {
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"],
                "label": ent["label"],
                "score": float(ent.get("score", 1.0)),
            }
            for ent in raw_preds
        ]


def get_dataset_labels(dataset: List[Dict[str, Any]]) -> List[str]:
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
    if labels is None:
        labels = get_dataset_labels(dataset)

    predictions: List[Dict[str, Any]] = []
    for record in tqdm(dataset, desc="Running GLiNER inference"):
        text = record.get("text", "")
        record_labels = labels if labels is not None else record.get("labels", [])
        pred_ents = predict_entities(
            model=model,
            text=text,
            labels=record_labels,
            threshold=threshold,
            flat_ner=flat_ner,
        )
        predictions.append(
            {
                "id": record["id"],
                "text": text,
                "entities": pred_ents,
            }
        )
    return predictions