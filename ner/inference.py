from typing import List, Dict, Any
from gliner import GLiNER
from tqdm import tqdm


def load_gliner_model(model_name: str = "gliner-community/gliner_small-v2.5") -> GLiNER:
    return GLiNER.from_pretrained(model_name)


def predict_entities(
        model: GLiNER,
        text: str,
        labels: List[str],
        threshold: float = 0.5
) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    raw_entities = model.predict_entities(text, labels, threshold=threshold)

    formatted_entities = []
    for ent in raw_entities:
        formatted_entities.append({
            "start": ent["start"],
            "end": ent["end"],
            "text": ent["text"],
            "label": ent["label"],
            "score": float(ent.get("score", 1.0))
        })
    return formatted_entities


def predict_dataset(
        model: GLiNER,
        dataset: List[Dict[str, Any]],
        labels: List[str],
        threshold: float = 0.5
) -> List[Dict[str, Any]]:
    predictions = []
    for record in tqdm(dataset, desc="Predicting entities"):
        doc_id = record.get("id")
        text = record.get("text", "")
        doc_labels = record.get("label_types", labels)

        predicted_ents = predict_entities(model, text, doc_labels, threshold=threshold)

        predictions.append({
            "id": doc_id,
            "text": text,
            "entities": predicted_ents
        })
    return predictions