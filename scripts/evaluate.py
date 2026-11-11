"""Retrieval sanity check and optional human-labelled relevance evaluation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from book_recommender.config import REPORTS
from book_recommender.vector_search import BookRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judgments", type=str, help="JSON list: query and relevant_isbns (human labels)"
    )
    args = parser.parse_args()
    engine = BookRecommender()
    sample = engine.books.sample(min(20, len(engine.books)), random_state=42)
    hits, latencies = 0, []
    for row in sample.itertuples():
        start = time.perf_counter()
        result = engine.search(row.description[:1000], top_k=1)
        hits += int(result.book_id.iloc[0] == row.book_id)
        latencies.append(time.perf_counter() - start)
    report = {
        "catalog_size": len(engine.books),
        "sanity_queries": len(sample),
        "description_self_retrieval_hit_at_1": hits / len(sample),
        "median_search_seconds": float(np.median(latencies)),
        "p95_search_seconds": float(np.percentile(latencies, 95)),
        "interpretation": "Self-retrieval checks indexing, not recommendation quality. Timing excludes model initialization.",
    }
    if args.judgments:
        judgments = json.loads(Path(args.judgments).read_text(encoding="utf-8"))
        if not judgments or any(not j.get("relevant_isbns") for j in judgments):
            raise ValueError("Fill in human relevance judgments before computing quality metrics")
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(engine.books.description)
        metrics = {"semantic": [], "tfidf": []}
        for item in judgments:
            relevant = set(map(str, item["relevant_isbns"]))
            if not relevant.issubset(set(engine.books.book_id)):
                raise ValueError("Judgments contain ISBNs absent from this catalog")
            semantic = engine.search(item["query"], top_k=5).book_id.tolist()
            scores = cosine_similarity(vectorizer.transform([item["query"]]), matrix).ravel()
            lexical = engine.books.iloc[np.argsort(-scores)[:5]].book_id.tolist()
            for name, ranking in [("semantic", semantic), ("tfidf", lexical)]:
                flags = [int(book in relevant) for book in ranking]
                dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(flags))
                ideal = sum(1 / np.log2(i + 2) for i in range(min(5, len(relevant))))
                metrics[name].append(
                    {
                        "precision_at_5": sum(flags) / 5,
                        "recall_at_5": sum(flags) / len(relevant),
                        "ndcg_at_5": dcg / ideal,
                    }
                )
        report["human_labeled_query_count"] = len(judgments)
        report["human_evaluation"] = {
            name: pd.DataFrame(rows).mean().to_dict() for name, rows in metrics.items()
        }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
