"""Estimate seven emotions from descriptions, not from the full books."""

import re

import numpy as np

from .checkpoints import predict_cached
from .config import EMOTION_MODEL, EMOTIONS


def description_chunks(text, max_words=100):
    """Bound sentence chunks so later parts of long descriptions are also scored."""
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.split()
        chunks.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
    return chunks or [text]


def score_emotions(books, batch_size=16):
    classifier = None

    def predict(texts):
        nonlocal classifier
        if classifier is None:
            from transformers import pipeline

            classifier = pipeline("text-classification", model=EMOTION_MODEL, device=-1, top_k=None)
        groups = [description_chunks(text) for text in texts]
        flat = [chunk for group in groups for chunk in group]
        predictions = classifier(flat, truncation=True, max_length=512, batch_size=batch_size)
        output, offset = [], 0
        for group in groups:
            # Mean gives an overall tone. Unlike max pooling, scores still sum to one.
            scores = [
                {item["label"]: float(item["score"]) for item in row}
                for row in predictions[offset : offset + len(group)]
            ]
            output.append(
                {emotion: float(np.mean([row[emotion] for row in scores])) for emotion in EMOTIONS}
            )
            offset += len(group)
        return output

    predictions = predict_cached(
        books.description.tolist(),
        EMOTION_MODEL,
        {"aggregation": "mean", "max_words": 100, "max_tokens": 512},
        predict,
        "emotions",
        batch_size,
    )
    result = books.copy()
    for emotion in EMOTIONS:
        result[emotion] = [row[emotion] for row in predictions]
    result["dominant_emotion"] = result[list(EMOTIONS)].idxmax(axis=1)
    return result
