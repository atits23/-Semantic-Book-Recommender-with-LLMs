"""Audit and clean the 7k Books CSV, keeping a reproducible cleaning report."""

import json

import pandas as pd

from .config import CLEAN, RAW, REPORTS, make_directories


def clean_books(raw: pd.DataFrame, min_words: int = 20):
    required = {"isbn13", "title", "description"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    books = raw.copy()
    report = {
        "input_rows": len(books),
        "missing_values": raw.isna().sum().to_dict(),
        "minimum_description_words": min_words,
    }
    for col in ["isbn13", "title", "description", "authors", "categories", "thumbnail"]:
        if col not in books:
            books[col] = ""
        books[col] = books[col].fillna("").astype(str).str.strip()
    # ISBN is an identifier, never an integer or float in saved files.
    books["isbn13"] = books.isbn13.str.replace(r"\.0$", "", regex=True)
    valid = books.isbn13.str.fullmatch(r"\d{13}") & books.title.ne("")
    report["invalid_identifier_or_title"] = int((~valid).sum())
    books = books.loc[valid].copy()
    books["description"] = books.description.str.replace(r"\s+", " ", regex=True)
    books["description_words"] = books.description.str.split().str.len()
    report["short_or_missing_description"] = int((books.description_words < min_words).sum())
    books = books.loc[books.description_words >= min_words].copy()
    report["duplicate_isbns"] = int(books.isbn13.duplicated().sum())
    books = books.drop_duplicates("isbn13", keep="first").reset_index(drop=True)
    books["authors"] = books.authors.replace("", "Unknown author")
    books["categories"] = books.categories.replace("", "Uncategorized")
    books["book_id"] = books.isbn13
    # Stable ordering makes subset runs and debugging repeatable.
    report["output_rows"] = len(books)
    return books, report


def run(input_path=RAW, output_path=CLEAN):
    make_directories()
    raw = pd.read_csv(input_path, dtype={"isbn13": str, "isbn10": str})
    books, report = clean_books(raw)
    if books.empty:
        raise ValueError("No usable books remain after cleaning. Check the input CSV.")
    books.to_csv(output_path, index=False)
    (REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    books.description_words.clip(upper=500).hist(bins=35, ax=axes[0], color="#167d8d")
    axes[0].set(title="Description lengths", xlabel="Words (capped at 500)", ylabel="Books")
    books.categories.value_counts().head(10).sort_values().plot.barh(ax=axes[1], color="#d79844")
    axes[1].set(title="Most common source categories", xlabel="Books", ylabel="")
    fig.tight_layout()
    fig.savefig(REPORTS / "data_exploration.png", dpi=160)
    plt.close(fig)
    return books, report
