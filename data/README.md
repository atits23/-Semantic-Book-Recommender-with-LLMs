# Data provenance

Use Dylan Castillo's [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata), the dataset used in the tutorial. `python -m scripts.download_data` downloads its public archive into `data/raw/books.csv`. If Kaggle changes its download endpoint, download the CSV manually from that page.

The project does not redistribute the source CSV through Git. Consult the dataset's current license and the underlying description/image rights before redistribution. Model licenses are separate from data rights.

`data/processed/` contains generated CSVs. ISBNs are strings. Categories from the source are retained for audit; the dashboard uses model-predicted fiction/non-fiction labels. Missing descriptions and descriptions with fewer than 20 words are removed; missing authors and covers are not reasons to discard a book.

`evaluation_queries.json` is a human-labelling template, not measured ground truth. Inspect the catalog and fill each `relevant_isbns` list with independently judged relevant ISBNs before running relevance evaluation. Include several relevant books per query, pool results from both systems, and judge without seeing which system retrieved them. Empty labels intentionally cause evaluation to fail rather than produce invented accuracy.
