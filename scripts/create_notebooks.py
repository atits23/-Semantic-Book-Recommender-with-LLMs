"""Generate the educational notebooks. Actual work lives in importable modules."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / "book_recommender").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from book_recommender.config import RAW, CLEAN, CLASSIFIED, ENRICHED, REPORTS
"""


def save(name, cells):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
        for kind, text in cells
    ]
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    nbf.write(notebook, folder / name)


def main():
    save(
        "01_data_exploration.ipynb",
        [
            (
                "md",
                "# 01 Data exploration and cleaning\n\n**Question:** Is the raw catalog suitable for semantic search? Inspect missing values, identifiers, and description lengths before modeling. Use the project virtual environment as this notebook's kernel. First run `python -m scripts.download_data` in the project root.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Inspect the raw data\nISBNs are identifiers, not numbers. Preserve leading zeros by reading them as strings. The dataset includes several metadata fields that are useful for displaying and auditing recommendations.",
            ),
            (
                "code",
                'raw = pd.read_csv(RAW, dtype={"isbn13": str, "isbn10": str})\nprint(raw.shape)\nraw.head(3)',
            ),
            ("code", 'raw.isna().sum().sort_values(ascending=False).to_frame("missing_rows")'),
            (
                "md",
                "## Apply explicit cleaning rules\nKeep descriptions of at least 20 words, valid 13-digit IDs, and nonempty titles. Deduplicate by ISBN; do not remove records merely because covers or authors are missing. Twenty words is a heuristic worth testing, not an objective definition of quality.",
            ),
            (
                "code",
                "from book_recommender.data_exploration import run\nbooks, report = run()\nreport",
            ),
            (
                "code",
                'from IPython.display import Image, display\ndisplay(Image(filename=str(REPORTS / "data_exploration.png")))',
            ),
            (
                "md",
                "## Check the result\nEach book must map to one unique vector ID. Different editions can still share descriptions, so unique ISBNs do not guarantee unique semantic content.",
            ),
            (
                "code",
                'assert books.book_id.is_unique\nassert books.description_words.min() >= 20\nbooks[["book_id", "title", "description_words"]].head()',
            ),
            (
                "md",
                "## Think about the tradeoff\nWhich useful short blurbs did cleaning remove? Compare thresholds of 10, 20, and 30 using `clean_books(raw, min_words=...)` before deciding. The next notebook predicts a broad fiction category.",
            ),
        ],
    )
    save(
        "02_text_classification.ipynb",
        [
            (
                "md",
                "# 02 Zero shot fiction classification\n\n**Question:** Can a pretrained model assign a broad category without training on this book dataset? We use natural language inference to compare candidate label hypotheses.",
            ),
            ("code", SETUP),
            (
                "code",
                'books = pd.read_csv(CLEAN, dtype={"isbn13": str, "book_id": str})\nbooks[["title", "categories"]].head()',
            ),
            (
                "md",
                "## From labels to hypotheses\nThe premise is the title and description. The candidates become `This book is Fiction.` and `This book is Non-fiction.` The NLI model compares entailment evidence. Zero-shot means no task-specific training on these books; the pretrained model has seen other labelled NLI data.",
            ),
            (
                "code",
                'from book_recommender.text_classification import classify_books\nexample = classify_books(books.head(8), batch_size=4)\nexample[["title", "categories", "predicted_category", "category_confidence", "category_needs_review"]]',
            ),
            (
                "md",
                "## Review uncertainty\nThe winning score is relative to these two labels. It is not calibrated confidence in factual correctness. Biographies written as narratives, poetry, and mixed collections may be difficult. We flag scores below 0.65 for manual review.",
            ),
            (
                "code",
                'example.sort_values("category_confidence")[["title", "description", "predicted_category", "category_confidence"]].head(3)',
            ),
            (
                "md",
                "## Build the catalog\nFor all books run `python -m scripts.build_pipeline --stage classify` in the root. Use `--limit 200` for a first run. Predictions are cached by text/model/settings; rerunning a completed batch does not repeat inference. The demonstration above deliberately does not overwrite the full catalog file.",
            ),
            (
                "code",
                'if CLASSIFIED.exists():\n    catalog = pd.read_csv(CLASSIFIED)\n    display(catalog.predicted_category.value_counts().to_frame("books"))\nelse:\n    print("Run the classify stage to create the catalog.")',
            ),
            (
                "md",
                "## Evaluation exercise\nManually label a random set of books, blind to predictions. Calculate precision/recall for both categories and inspect a confusion matrix. Avoid treating imperfect source genres as unquestioned ground truth.",
            ),
        ],
    )
    save(
        "03_sentiment_analysis.ipynb",
        [
            (
                "md",
                "# 03 Emotion analysis\n\n**Question:** How can readers prefer a mood among semantically relevant books? We estimate seven emotions from blurbs using a pretrained DistilRoBERTa classifier. This is more specific than positive/negative sentiment.",
            ),
            ("code", SETUP),
            (
                "code",
                'from book_recommender.config import EMOTIONS\nfrom book_recommender.sentiment_analysis import description_chunks, score_emotions\nbooks = pd.read_csv(CLASSIFIED, dtype={"isbn13": str, "book_id": str})\ndescription_chunks(books.description.iloc[0])[:3]',
            ),
            (
                "md",
                "## Why split descriptions\nA transformer has a token limit. Sentence chunks capped at 100 words let later parts of long blurbs contribute. A word cap is not a token guarantee, so token truncation is retained. This sentence splitter is intentionally simple and can mishandle abbreviations.",
            ),
            (
                "code",
                'example = score_emotions(books.head(5), batch_size=4)\nexample[["title", *EMOTIONS, "dominant_emotion"]]',
            ),
            (
                "md",
                "## Aggregate chunk predictions\nWe average each emotion's probabilities across chunks, giving an overall description tone. Maximum pooling instead highlights isolated intense moments. Mean probabilities sum to approximately one; neither aggregation guarantees the tone of the full book.",
            ),
            (
                "code",
                'import numpy as np\nassert np.allclose(example[list(EMOTIONS)].sum(axis=1), 1, atol=1e-5)\nexample.set_index("title")[list(EMOTIONS)].plot.barh(stacked=True, figsize=(10, 5))',
            ),
            (
                "md",
                "## Prepare all scores\nRun `python -m scripts.build_pipeline --stage emotions`. The dashboard uses Happy→joy, Sad→sadness, Suspenseful→fear, Surprising→surprise, Angry→anger, and Calm→neutral. Fear and neutral are imperfect proxies for suspense and calm.",
            ),
            (
                "code",
                'if ENRICHED.exists():\n    enriched = pd.read_csv(ENRICHED)\n    display(enriched.dominant_emotion.value_counts().to_frame("books"))',
            ),
            (
                "md",
                "## Reflection\nCompare a cheerful blurb for a tragic story with a neutral factual description. Why might classifier probabilities disagree with a reader? Explain domain shift and marketing language in your interview.",
            ),
        ],
    )
    save(
        "04_vector_search.ipynb",
        [
            (
                "md",
                "# 04 Semantic vector search\n\n**Question:** How do we match descriptions by meaning rather than exact words? MiniLM maps queries and book blurbs into the same 384-dimensional space. LangChain supplies document and vector-store interfaces; Chroma stores the actual vectors.",
            ),
            ("code", SETUP),
            (
                "code",
                'from book_recommender.vector_search import LocalEmbeddings\nembedding = LocalEmbeddings()\nvectors = embedding.embed_documents(["A voyage to distant planets", "An adventure in outer space", "Cooking soup at home"])\nimport numpy as np\nnp.asarray(vectors).shape',
            ),
            (
                "md",
                "## Cosine similarity\nFor normalized vectors, the dot product equals cosine similarity. A score is geometric similarity, not the chance that a reader will like the book. MiniLM truncates long descriptions at its configured sequence length; check it below.",
            ),
            (
                "code",
                'print("Token limit:", embedding.model.max_seq_length)\nnp.asarray(vectors) @ np.asarray(vectors).T',
            ),
            (
                "md",
                "## Persistent storage\nRun `python -m scripts.build_pipeline --stage index` once after enrichment. One document per book carries its identifier and category in metadata. This avoids parsing IDs out of text and keeps repeated runs idempotent.",
            ),
            (
                "code",
                "from book_recommender.vector_search import BookRecommender\nengine = BookRecommender(embeddings=embedding)\nengine.manifest",
            ),
            (
                "code",
                'results = engine.search("A journey through space and the search for other life", category="All", top_k=5)\nresults[["title", "predicted_category", "similarity", "rank_score"]]',
            ),
            (
                "md",
                "## Filter before retrieval and rerank after\nThe category is a Chroma metadata filter. Mood blends a normalized similarity with an emotion probability within a semantic shortlist. Increasing mood influence can lower topical relevance, so show both controls and scores.",
            ),
            (
                "code",
                'engine.search("A family rebuilding their lives", tone="Happy", weight=0.4, top_k=5)[["title", "similarity", "joy", "rank_score"]]',
            ),
            (
                "md",
                "## Extension exercise\nTry chunked embeddings for long blurbs, deduplicating hits by ISBN. Compare against the single-vector baseline using human judgments before deciding it is better.",
            ),
        ],
    )
    save(
        "05_evaluation.ipynb",
        [
            (
                "md",
                "# 05 Evaluation and honest reporting\n\n**Question:** What evidence would establish that this recommender works? Separate software correctness, retrieval sanity, and user relevance. A perfect sanity check is not a perfect recommendation system.",
            ),
            ("code", SETUP),
            (
                "md",
                "## Run the tests\nThe integration tests use real persisted Chroma with tiny deterministic embeddings to isolate ID mapping, filters, rebuilds, and stale-data detection. They do not validate pretrained model quality.",
            ),
            (
                "code",
                'import subprocess\nresult = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)\nprint(result.stdout)\nassert result.returncode == 0, result.stderr',
            ),
            (
                "md",
                "## Self retrieval and latency\nRun `python -m scripts.evaluate`. It searches with sampled book descriptions and asks whether the source book ranks first. Timings exclude initial model loading. Near-duplicate descriptions can produce ties.",
            ),
            (
                "code",
                'import json\npath = REPORTS / "evaluation.json"\nif path.exists():\n    display(json.loads(path.read_text()))\nelse:\n    print("Run python -m scripts.evaluate first.")',
            ),
            (
                "md",
                "## Human relevance labels\nInspect `data/evaluation_queries.json`. Fill relevant ISBNs by judging a pooled set of semantic and TF-IDF results without knowing which system retrieved them. Keep a separate held-out query set for final reporting. Empty labels are intentionally rejected by the evaluator.",
            ),
            (
                "code",
                'judgments = json.loads((ROOT / "data/evaluation_queries.json").read_text())\npd.DataFrame(judgments)',
            ),
            (
                "md",
                "## Metrics\nPrecision@5 is relevant retrieved books divided by five. Recall@5 is relevant retrieved books divided by all relevant books in the judged set. nDCG discounts lower ranks and divides by the ideal ranking. Incomplete relevance labels distort recall; do not claim dataset-wide accuracy from a small subjective test.",
            ),
            (
                "md",
                "## Final demonstration\nStart `python app.py`. Show a semantic query, category constraint, tone reranking, and score explanation. Report actual catalog size and measured latency from `reports/`; mention truncation, weak labels, and cold-start limitations. This is inference using pretrained models, not custom LLM training.",
            ),
        ],
    )


if __name__ == "__main__":
    main()
