"""Real Chroma persistence with tiny deterministic embeddings; no model downloads."""

import pandas as pd
import pytest
from langchain_core.embeddings import Embeddings

from book_recommender.vector_search import BookRecommender, build_index


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] if "space" in t else [0.0, 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_persistence_filters_rebuild_and_stale_data(tmp_path):
    path = tmp_path / "books.csv"
    books = pd.DataFrame(
        [
            {
                "book_id": "001",
                "isbn13": "001",
                "description": "space adventure",
                "predicted_category": "Fiction",
                "joy": 0.1,
            },
            {
                "book_id": "002",
                "isbn13": "002",
                "description": "space science",
                "predicted_category": "Non-fiction",
                "joy": 0.2,
            },
            {
                "book_id": "003",
                "isbn13": "003",
                "description": "family story",
                "predicted_category": "Fiction",
                "joy": 0.9,
            },
        ]
    )
    books.to_csv(path, index=False)
    embeddings = TinyEmbeddings()
    build_index(path, embeddings, tmp_path)
    build_index(path, embeddings, tmp_path)  # upsert must not duplicate IDs
    engine = BookRecommender(path, embeddings, tmp_path)
    assert len(engine.store.get()["ids"]) == 3
    hits = engine.search("space", "Non-fiction", top_k=2)
    assert hits.book_id.tolist() == ["002"]
    assert hits.similarity.iloc[0] == pytest.approx(1)
    assert engine.search("space", "Fiction", top_k=1).book_id.tolist() == ["001"]
    with pytest.raises(ValueError, match="Describe"):
        engine.search("  ")
    with pytest.raises(ValueError, match="category"):
        engine.search("space", "invalid")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Catalog changed"):
        BookRecommender(path, embeddings, tmp_path)
