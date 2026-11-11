"""Gradio entry point. Run from the project root with: python app.py."""

import html
import logging
import time
from functools import lru_cache

import pandas as pd

from book_recommender.config import EMOTIONS, TONES

CSS = """
.gradio-container {max-width: 1240px !important; margin:auto;}
#masthead {padding: 32px 0 18px; border-bottom: 1px solid #d9ddd5; margin-bottom: 22px;}
#masthead .eyebrow {color:#177a76; font-size:12px; font-weight:700; letter-spacing:2px;}
#masthead h1 {font-family: Georgia,serif; font-size:48px; font-weight:400; color:#193e3a; margin:10px 0;}
#masthead p {font-size:16px; color:#576a64; max-width:680px; line-height:1.6;}
.book-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;}
.book-card {background:#fffef9;border:1px solid #dfe3d9;border-radius:14px;padding:22px;}
.book-card .number {color:#a77836;font-family:Georgia,serif;font-size:27px;}
.book-card h3 {font-family:Georgia,serif;font-size:23px;line-height:1.25;color:#193e3a;margin:12px 0 8px;}
.book-card .author {font-size:13px;color:#65726a;margin-bottom:16px;}
.book-card p {color:#48574f;font-size:14px;line-height:1.65;}
.book-card .tag {display:inline-block;background:#eaf2ed;color:#24574b;padding:4px 8px;
 border-radius:6px;font-size:11px;margin:0 5px 8px 0;}
.book-card .score {border-top:1px solid #edf0e9;padding-top:12px;margin-top:15px;font-size:12px;color:#65726a;}
.book-card summary {cursor:pointer;color:#177a76;font-size:13px;}
.empty {padding:45px;text-align:center;border:1px dashed #c9d4c9;border-radius:14px;color:#65726a;}
@media(max-width:900px){.book-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:600px){.book-grid{grid-template-columns:1fr;}#masthead h1{font-size:35px;}}
"""
EMPTY = '<div class="empty">Your next chapter starts with a search.<br>Describe a theme, a feeling, or a kind of story.</div>'
LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_recommender():
    from book_recommender.vector_search import BookRecommender

    return BookRecommender()


def render_cards(books):
    if books.empty:
        return '<div class="empty">No books match this category. Try All or rebuild with a larger catalog.</div>'
    cards = []
    for i, row in enumerate(books.to_dict("records"), 1):

        def esc(key, record=row):
            return html.escape(str(record.get(key, "")))

        preview = html.escape(" ".join(str(row["description"]).split()[:42]))
        if len(str(row["description"]).split()) > 42:
            preview += "…"
        warning = " · uncertain category" if float(row["category_confidence"]) < 0.65 else ""
        cards.append(f"""<article class="book-card"><div class="number">{i:02d}</div>
        <h3>{esc("title")}</h3><div class="author">{esc("authors").replace(";", ", ")}</div>
        <span class="tag">{esc("predicted_category")}{warning}</span>
        <span class="tag">{esc("dominant_emotion")}</span><p>{preview}</p>
        <details><summary>Read the full description</summary><p>{esc("description")}</p>
        <p>ISBN {esc("isbn13")} · Source category: {esc("categories")}</p></details>
        <div class="score">Semantic similarity {float(row["similarity"]):.3f} · Ranking score {float(row["rank_score"]):.3f}</div></article>""")
    return '<div class="book-grid">' + "".join(cards) + "</div>"


def recommend(query, category, tone, top_k, weight):
    import gradio as gr

    started = time.perf_counter()
    try:
        recommender = get_recommender()
        books = recommender.search(query, category, tone, int(top_k), weight)
        columns = [
            "title",
            "authors",
            "isbn13",
            "predicted_category",
            "category_confidence",
            "similarity",
            "rank_score",
            *EMOTIONS,
        ]
        table = books[columns].round(4) if not books.empty else pd.DataFrame(columns=columns)
        status = f"**{len(books)} recommendations** · {time.perf_counter() - started:.2f}s · {len(recommender.books):,} books in the catalog"
        return render_cards(books), status, table
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Recommendation failed")
        raise gr.Error(
            "Could not load the local recommender. Check the terminal and the setup steps in README.md."
        ) from exc


def create_app():
    import gradio as gr

    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="stone", font=["Arial", "sans-serif"])
    with gr.Blocks(title="Between the Lines | Book Recommender", theme=theme, css=CSS) as demo:
        gr.HTML("""<header id="masthead"><div class="eyebrow">BETWEEN THE LINES / A SEMANTIC BOOK FINDER</div>
        <h1>Find a story that feels right.</h1><p>Start with an idea, not a title. Discover books by what they are about,
        then shape your shortlist with a category and a mood.</p></header>""")
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="What would you like to read?",
                    lines=3,
                    placeholder="A story about unlikely friendships, second chances, and finding a place to belong",
                )
            with gr.Column(scale=1):
                category = gr.Dropdown(
                    ["All", "Fiction", "Non-fiction"], value="All", label="Category"
                )
                tone = gr.Dropdown(list(TONES), value="Any tone", label="Emotional tone")
        with gr.Accordion("Fine-tune your shortlist", open=False):
            count = gr.Slider(3, 24, value=9, step=3, label="Number of recommendations")
            weight = gr.Slider(
                0,
                0.6,
                value=0.25,
                step=0.05,
                label="Mood influence",
                info="0 keeps semantic order. Higher values favor your selected tone within the closest matches.",
            )
        submit = gr.Button("Find my next read", variant="primary", size="lg")
        gr.Examples(
            examples=[
                ["A detective investigating a mysterious disappearance"],
                ["Understanding the universe, stars, and the origins of life"],
                ["A family rebuilding their lives after loss"],
                ["Practical advice on creativity and developing better habits"],
            ],
            inputs=[query],
        )
        status = gr.Markdown("Search the catalog using your own words.")
        results = gr.HTML(EMPTY)
        with gr.Accordion("Explore the recommendation scores", open=False):
            gr.Markdown(
                "Similarity and ranking scores are **not accuracy percentages**. Emotion scores describe the book's blurb, not your likely reaction. Fiction labels are model predictions."
            )
            table = gr.Dataframe(
                interactive=False, label="Scores for this shortlist", show_search="filter"
            )
        with gr.Accordion("How this works", open=False):
            gr.Markdown(
                "Your query is embedded locally with MiniLM. Chroma searches book descriptions, applying the category filter first. "
                "An optional mood preference reranks the closest matches using precomputed emotion scores. "
                "No paid API, account, or generated plot summary is involved. English descriptions work best. "
                "A suspenseful mood uses fear as an imperfect proxy; calm uses neutral. This is content-based retrieval, not personalized reading-history prediction."
            )
        for event in [submit.click, query.submit]:
            event(
                recommend,
                inputs=[query, category, tone, count, weight],
                outputs=[results, status, table],
                api_name=False,
            )
        gr.Markdown(
            "Built with LangChain · Chroma · Hugging Face Transformers · Gradio  |  Local inference, zero API fees"
        )
    return demo


if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False, inbrowser=True
    )
