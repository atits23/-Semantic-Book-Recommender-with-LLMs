"""LangChain documents + local Sentence Transformers + persistent Chroma."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import ARTIFACTS, EMBEDDING_MODEL, ENRICHED, TONES


class LocalEmbeddings(Embeddings):
    """Small explicit LangChain adapter; both sides use the same normalization."""

    def __init__(self, model_name=EMBEDDING_MODEL):
        import torch

        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def data_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def open_store(collection, embeddings, directory=None):
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(directory or ARTIFACTS / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=Settings(anonymized_telemetry=False),
    )


def build_index(path=ENRICHED, embeddings=None, artifacts=ARTIFACTS):
    books = pd.read_csv(path, dtype={"isbn13": str, "book_id": str}).fillna("")
    if books.empty or books.book_id.duplicated().any():
        raise ValueError("The index requires nonempty data with unique book IDs")
    fingerprint = data_fingerprint(path)
    model_hash = hashlib.sha256(EMBEDDING_MODEL.encode()).hexdigest()[:8]
    collection = f"books-{fingerprint[:16]}-{model_hash}"
    store = open_store(collection, embeddings or LocalEmbeddings(), artifacts / "chroma")
    for start in range(0, len(books), 128):
        batch = books.iloc[start : start + 128]
        docs = [
            Document(
                page_content=row.description,
                metadata={"book_id": str(row.book_id), "category": row.predicted_category},
            )
            for row in batch.itertuples()
        ]
        store.add_documents(docs, ids=batch.book_id.astype(str).tolist())
    # Publish the manifest only after all inserts succeed. Older collections remain intact.
    manifest = {
        "collection": collection,
        "embedding_model": EMBEDDING_MODEL,
        "data_sha256": fingerprint,
        "book_count": len(books),
        "distance": "cosine",
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    temporary = artifacts / "index.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(artifacts / "index.json")
    return manifest


def rank_candidates(candidates, tone="Any tone", weight=0.25, top_k=12):
    """Blend only within retrieved candidates; never search the whole corpus by mood."""
    if tone not in TONES:
        raise ValueError("Unknown tone")
    if not 0 <= weight <= 1:
        raise ValueError("Tone weight must be between zero and one")
    result = candidates.copy()
    if result.empty:
        return result
    emotion = TONES[tone]
    # Map cosine [-1, 1] onto [0, 1] before blending with emotion probabilities.
    semantic = (result.similarity.clip(-1, 1) + 1) / 2
    result["rank_score"] = (
        semantic if emotion is None else (1 - weight) * semantic + weight * result[emotion]
    )
    return (
        result.sort_values(["rank_score", "similarity", "book_id"], ascending=[False, False, True])
        .head(top_k)
        .reset_index(drop=True)
    )


class BookRecommender:
    def __init__(self, path=ENRICHED, embeddings=None, artifacts=ARTIFACTS):
        manifest_path = artifacts / "index.json"
        if not manifest_path.exists() or not Path(path).exists():
            raise FileNotFoundError("Prepare the catalog first: python -m scripts.build_pipeline")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["data_sha256"] != data_fingerprint(path):
            raise ValueError(
                "Catalog changed since indexing. Run python -m scripts.build_pipeline --stage index"
            )
        if self.manifest["embedding_model"] != EMBEDDING_MODEL:
            raise ValueError("Embedding model changed. Rebuild the index.")
        self.books = pd.read_csv(path, dtype={"book_id": str, "isbn13": str}).fillna("")
        self.by_id = self.books.set_index("book_id", drop=False)
        self.store = open_store(
            self.manifest["collection"], embeddings or LocalEmbeddings(), artifacts / "chroma"
        )
        if len(self.store.get(include=[])["ids"]) != len(self.books):
            raise ValueError("Index is incomplete. Rebuild the index.")

    def search(self, query, category="All", tone="Any tone", top_k=12, weight=0.25):
        query = query.strip()
        if not query:
            raise ValueError("Describe the kind of book you would like to read.")
        if len(query) > 1500:
            raise ValueError("Please keep your search under 1,500 characters.")
        if category not in ["All", "Fiction", "Non-fiction"]:
            raise ValueError("Unknown category")
        if tone not in TONES or not 1 <= int(top_k) <= 30:
            raise ValueError("Invalid tone or result count")
        eligible = (
            len(self.books)
            if category == "All"
            else int((self.books.predicted_category == category).sum())
        )
        if eligible == 0:
            return pd.DataFrame()
        where = None if category == "All" else {"category": category}
        hits = self.store.similarity_search_with_score(
            query, k=min(eligible, max(50, int(top_k) * 4)), filter=where
        )
        records = []
        for doc, distance in hits:
            record = self.by_id.loc[doc.metadata["book_id"]].to_dict()
            record["similarity"] = float(1 - distance)
            records.append(record)
        return rank_candidates(pd.DataFrame(records), tone, float(weight), int(top_k))
