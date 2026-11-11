"""Run all stages, or inspect/re-run one stage at a time."""

import argparse
import json
import time

import pandas as pd

from book_recommender.config import CLASSIFIED, CLEAN, ENRICHED, REPORTS, make_directories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "clean", "classify", "emotions", "index"], default="all"
    )
    parser.add_argument(
        "--limit", type=int, help="Deterministic subset for a first CPU run; omit for full catalog"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1 or args.batch_size < 1:
        parser.error("Limit and batch size must be positive")
    make_directories()
    import torch

    torch.set_num_threads(4)
    stages = ["clean", "classify", "emotions", "index"] if args.stage == "all" else [args.stage]
    timings = {}
    for stage in stages:
        start = time.perf_counter()
        print(f"\nStarting {stage}", flush=True)
        if stage == "clean":
            from book_recommender.data_exploration import run

            _, report = run()
            print(json.dumps(report, indent=2))
        elif stage == "classify":
            from book_recommender.text_classification import classify_books

            books = pd.read_csv(CLEAN, dtype={"isbn13": str, "book_id": str})
            if args.limit:
                books = books.head(args.limit)
            classify_books(books, args.batch_size).to_csv(CLASSIFIED, index=False)
        elif stage == "emotions":
            from book_recommender.sentiment_analysis import score_emotions

            books = pd.read_csv(CLASSIFIED, dtype={"isbn13": str, "book_id": str})
            score_emotions(books, args.batch_size).to_csv(ENRICHED, index=False)
        else:
            from book_recommender.vector_search import build_index

            print(build_index())
        timings[stage] = round(time.perf_counter() - start, 2)
        print(f"Finished {stage} in {timings[stage]} seconds", flush=True)
    run_report = {
        "stage": args.stage,
        "classification_limit": args.limit,
        "batch_size": args.batch_size,
        "seconds": timings,
        "note": "Wall time for this invocation; cached predictions may be reused. Not a cold-start benchmark.",
    }
    (REPORTS / f"timings_{args.stage}.json").write_text(
        json.dumps(run_report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
