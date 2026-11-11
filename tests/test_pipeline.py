import pandas as pd
import pytest

from book_recommender.checkpoints import predict_cached
from book_recommender.data_exploration import clean_books
from book_recommender.sentiment_analysis import description_chunks
from book_recommender.vector_search import rank_candidates


def test_cleaning_handles_missing_text_duplicate_isbns_and_leading_zero():
    rows = [
        {"isbn13": "0123456789012", "title": "A", "description": "word " * 25},
        {"isbn13": "0123456789012", "title": "Duplicate", "description": "word " * 25},
        {"isbn13": "1234567890123", "title": "Empty", "description": None},
        {"isbn13": "bad", "title": "Bad", "description": "word " * 25},
    ]
    books, report = clean_books(pd.DataFrame(rows))
    assert books.book_id.tolist() == ["0123456789012"]
    assert books.authors.iloc[0] == "Unknown author"
    assert report["duplicate_isbns"] == 1
    assert report["invalid_identifier_or_title"] == 1
    assert report["short_or_missing_description"] == 1


def test_missing_required_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        clean_books(pd.DataFrame({"title": ["A"]}))


def test_chunks_preserve_end_of_long_description():
    text = " ".join(["word"] * 250 + ["ending"])
    chunks = description_chunks(text)
    assert max(len(c.split()) for c in chunks) <= 100
    assert " ".join(chunks) == text


def test_mood_reranking_and_zero_weight():
    books = pd.DataFrame(
        [
            {"book_id": "a", "similarity": 0.9, "joy": 0.1},
            {"book_id": "b", "similarity": 0.8, "joy": 0.9},
        ]
    )
    assert rank_candidates(books, "Happy", weight=0).book_id.tolist() == ["a", "b"]
    assert rank_candidates(books, "Happy", weight=0.4).book_id.tolist() == ["b", "a"]
    assert rank_candidates(books, "Any tone", weight=0.4).book_id.tolist() == ["a", "b"]
    with pytest.raises(ValueError):
        rank_candidates(books, "Unknown")


def test_checkpoint_resumes_and_invalidates_changed_model(monkeypatch, tmp_path):
    import book_recommender.checkpoints as cp

    monkeypatch.setattr(cp, "ARTIFACTS", tmp_path)
    (tmp_path / "checkpoints").mkdir()
    calls = []

    def predict(texts):
        calls.extend(texts)
        return [{"value": t.upper()} for t in texts]

    assert predict_cached(["one"], "m1", {}, predict, "test")[0]["value"] == "ONE"
    predict_cached(["one", "two"], "m1", {}, predict, "test")
    assert calls == ["one", "two"]
    predict_cached(["one"], "m2", {}, predict, "test")
    assert calls == ["one", "two", "one"]


def test_html_escapes_catalog_text():
    from app import render_cards

    row = {
        "title": "<script>alert(1)</script>",
        "description": "<b>text</b>",
        "category_confidence": 0.9,
        "similarity": 0.8,
        "rank_score": 0.9,
    }
    rendered = render_cards(pd.DataFrame([row]))
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_cache_keeps_original_order_after_length_batching(monkeypatch, tmp_path):
    import book_recommender.checkpoints as cp

    monkeypatch.setattr(cp, "ARTIFACTS", tmp_path)
    (tmp_path / "checkpoints").mkdir()
    texts = ["a much longer description", "short", "short"]
    predictions = predict_cached(texts, "model", {}, lambda batch: batch, "order", 1)
    assert predictions == texts
    # A partial last line should not prevent valid cached records being reused.
    with (tmp_path / "checkpoints/order.jsonl").open("a") as handle:
        handle.write('{"key": "broken"')
    predictions = predict_cached(
        texts, "model", {}, lambda batch: pytest.fail("Unexpected inference"), "order", 1
    )
    assert predictions == texts
