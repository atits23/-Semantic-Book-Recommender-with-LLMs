"""Fiction/non-fiction prediction using natural language inference."""

from .checkpoints import predict_cached
from .config import CLASSIFICATION_MODEL

LABELS = ["Fiction", "Non-fiction"]
HYPOTHESIS = "This book is {}."


def classify_books(books, batch_size=16):
    classifier = None

    def predict(texts):
        nonlocal classifier
        if classifier is None:
            import torch
            from transformers import pipeline

            # The checkpoint uses float16; float32 is much faster on many CPUs.
            classifier = pipeline(
                "zero-shot-classification",
                model=CLASSIFICATION_MODEL,
                device=-1,
                dtype=torch.float32,
            )
        results = classifier(
            texts,
            candidate_labels=LABELS,
            hypothesis_template=HYPOTHESIS,
            multi_label=False,
            batch_size=batch_size,
        )
        return [
            {"predicted_category": r["labels"][0], "category_confidence": float(r["scores"][0])}
            for r in results
        ]

    # Include the title but deliberately do not leak the source genre into the classifier.
    texts = (books.title + ". " + books.description).tolist()
    predictions = predict_cached(
        texts,
        CLASSIFICATION_MODEL,
        {"labels": LABELS, "hypothesis": HYPOTHESIS, "dtype": "float32"},
        predict,
        "classification",
        batch_size,
    )
    result = books.copy()
    for col in ["predicted_category", "category_confidence"]:
        result[col] = [p[col] for p in predictions]
    result["category_needs_review"] = result.category_confidence < 0.65
    return result
