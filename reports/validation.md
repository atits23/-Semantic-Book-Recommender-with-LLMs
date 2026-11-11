# Validation report

Verified on 11 November 2026 with Python 3.12.14 on Windows. This report describes the complete local catalog, not the initial 200-book smoke test.

## Catalog

- Raw dataset: **6,810** rows.
- Cleaned, classified, emotion-scored, and indexed: **5,729** books.
- Removed for missing or fewer than 20 description words: **1,081** rows.
- Predicted Fiction: **3,788**; predicted Non-fiction: **1,941**.
- Category predictions below the 0.65 review threshold: **519**.
- All seven emotion probabilities are present. Per-book sums range from approximately 0.9999998 to 1.0000002, within floating-point tolerance.
- Persistent Chroma index and model caches are built locally. The app was tested with `HF_HUB_OFFLINE=1`.

The models used are MiniLM for embeddings, DeBERTa base zero-shot v2.0 for categories, and Hartmann DistilRoBERTa for emotions. Exact downloaded revision hashes are recorded in `model_versions.json`; the installed Python environment is recorded in `../requirements-lock.txt`.

## Checks completed

- **8 tests passed**, including actual Chroma persistence, category filtering, repeat upserts, stale-data rejection, cache recovery/order preservation, cleaning, chunking, mood ranking, and HTML escaping. `tests.xml` contains the test-run record.
- Ruff checks passed and Python source compiled successfully.
- `pip check` reported no broken requirements.
- **All 5 notebooks executed successfully** against the final catalog and current model configuration. Outputs are saved in the notebooks.
- A headless Edge browser submitted a natural-language query and received nine actual book recommendations.
- Empty-query validation displayed the expected error.
- Selecting Fiction returned Fiction cards.
- Desktop and 390-pixel mobile layouts were checked; no mobile horizontal overflow and no browser JavaScript errors were observed.

## Retrieval sanity check

For 20 sampled books, querying with each book's description recovered the original book at rank one in **20 of 20** cases. The median measured search time was approximately **0.094 seconds** and the 95th percentile approximately **0.332 seconds**. These timings exclude initial model loading and depend on hardware and concurrent activity.

This is an index/retrieval sanity check, **not recommendation accuracy**. Natural-language relevance and category accuracy still require a representative independently labelled evaluation set. The provided relevance-label template is intentionally unfilled; the evaluator will not invent relevance metrics.

## Runtime notes

The final full classification run took approximately 53.6 minutes on this CPU. Emotion-cache completion took approximately 13.7 minutes, partly overlapping classification. Final embedding/index creation took approximately 3.8 minutes. These are observations from this machine, not promised setup times.

The final `timings_all.json` records a rebuild with classification and emotion predictions already cached; its sub-second inference-stage times must not be presented as cold-start preprocessing times. The application reuses the persisted index and needs only a query embedding during search.

## Start the application

Run `.venv/Scripts/python.exe app.py` from the project directory, or double-click `run_dashboard.bat` on Windows. The local address is normally http://127.0.0.1:7860. Do not rebuild the catalog while serving it; stop and restart the app around rebuilds.
