"""Shared paths and model choices. No API keys or hosted inference."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
RAW = ROOT / "data/raw/books.csv"
CLEAN = ROOT / "data/processed/books_clean.csv"
CLASSIFIED = ROOT / "data/processed/books_classified.csv"
ENRICHED = ROOT / "data/processed/books_enriched.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFICATION_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
TONES = {
    "Any tone": None,
    "Happy": "joy",
    "Sad": "sadness",
    "Suspenseful": "fear",
    "Surprising": "surprise",
    "Angry": "anger",
    "Calm / neutral": "neutral",
}


def make_directories():
    for path in [RAW.parent, CLEAN.parent, ARTIFACTS, REPORTS, ARTIFACTS / "checkpoints"]:
        path.mkdir(parents=True, exist_ok=True)
