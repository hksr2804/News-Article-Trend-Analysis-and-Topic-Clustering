"""
src/dashboard.py — Streamlit dashboard for News Article Topic Analysis.

Public API (pure functions, also used in tests)
------------------------------------------------
filter_by_date(df, start_date, end_date)          → pd.DataFrame
filter_by_topics(df, selected_topics, threshold)  → pd.DataFrame
load_topic_names(path, num_topics)                → dict[str, str]
build_bar_chart(keywords)                         → matplotlib.figure.Figure | None
build_wordcloud(keywords)                         → PIL.Image.Image | None

Entry point
-----------
run()   — Streamlit app; call via `streamlit run src/dashboard.py`
"""

import json
import os
import sys
from datetime import date

# Ensure project root is on path — required when launched as `streamlit run src/dashboard.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # must come before pyplot import; non-interactive backend for st.pyplot()
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

import importlib
import config
importlib.reload(config)  # force reload so Streamlit always uses the current config.py
from src.keywords import extract_keywords


# ---------------------------------------------------------------------------
# Pure logic functions (testable without Streamlit)
# ---------------------------------------------------------------------------

def filter_by_date(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Return rows where start_date <= date <= end_date (inclusive)."""
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask].reset_index(drop=True)


def filter_by_topics(
    df: pd.DataFrame,
    selected_topics: list[int],
    threshold: float,
) -> pd.DataFrame:
    """Return rows matching any of the selected topics at or above the threshold.

    If selected_topics is empty, all rows are returned (no topic filtering).
    A row matches if ANY of the selected topic columns >= threshold.
    """
    if not selected_topics:
        return df.reset_index(drop=True)

    mask = pd.Series(False, index=df.index)
    for i in selected_topics:
        col = f"topic_{i}"
        mask = mask | (df[col] >= threshold)
    return df[mask].reset_index(drop=True)


def load_topic_names(path: str, num_topics: int = config.NUM_TOPICS) -> dict[str, str]:
    """Load topic name mapping from a JSON file.

    Returns default placeholders {"0": "Topic 0", ...} if file is missing or malformed.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        return {str(i): f"Topic {i}" for i in range(num_topics)}


def build_bar_chart(keywords: list[tuple[str, float]]) -> "matplotlib.figure.Figure | None":
    """Build a horizontal bar chart of keyword TF-IDF scores.

    Returns a matplotlib Figure, or None if keywords is empty.
    """
    if not keywords:
        return None

    labels = [kw for kw, _ in keywords]
    scores = [score for _, score in keywords]

    fig, ax = plt.subplots(figsize=(8, max(3, len(keywords) * 0.6)))
    ax.barh(labels[::-1], scores[::-1], color="steelblue")
    ax.set_xlabel("TF-IDF Score")
    ax.set_title("Top Keywords")
    fig.tight_layout()
    return fig


def build_wordcloud(keywords: list[tuple[str, float]]) -> "PIL.Image.Image | None":
    """Build a word cloud image from keyword TF-IDF scores.

    Returns a PIL Image, or None if keywords is empty.
    """
    if not keywords:
        return None

    freq = {word: score for word, score in keywords}
    wc = WordCloud(width=800, height=400, background_color="white")
    wc.generate_from_frequencies(freq)
    return wc.to_image()


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

@st.cache_data
def _load_articles() -> pd.DataFrame:
    """Load and cache the articles-with-topics parquet once per session."""
    df = pd.read_parquet(config.TOPICS_DATA_PATH)
    # Ensure date column is Python date objects for comparisons
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.date
    return df


def run() -> None:
    """Streamlit dashboard entry point."""
    st.title("News Article Topic Analysis")
    st.markdown("Explore trending keywords by date range and topic.")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = _load_articles()
    topic_names = load_topic_names(config.TOPIC_NAMES_PATH, num_topics=config.NUM_TOPICS)

    min_date = df["date"].min()
    max_date = df["date"].max()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    start_date = st.sidebar.date_input(
        "From",
        value=None,
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.sidebar.date_input(
        "To",
        value=None,
        min_value=min_date,
        max_value=max_date,
    )

    topic_options = {
        topic_names[str(i)]: i for i in range(config.NUM_TOPICS)
    }
    selected_labels = st.sidebar.multiselect(
        "Topics",
        options=list(topic_options.keys()),
        default=[],
    )
    selected_topics = [topic_options[lbl] for lbl in selected_labels]

    # ── Require filters before showing results ────────────────────────────────
    if start_date is None or end_date is None or not selected_topics:
        st.info("Select a date range and at least one topic in the sidebar to see results.")
        return
    if start_date > end_date:
        st.warning("'From' date must be before 'To' date.")
        return

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = filter_by_date(df, start_date, end_date)
    filtered = filter_by_topics(filtered, selected_topics, config.THRESHOLD)

    st.markdown(f"**{len(filtered):,} articles** match your filters.")

    if len(filtered) == 0:
        st.warning(
            "No articles match the selected filters. "
            "Try widening the date range or selecting more topics."
        )
        return

    # ── Keyword extraction + chart ─────────────────────────────────────────────
    st.subheader("Top Trending Keywords")
    with st.spinner("Extracting keywords..."):
        keywords = extract_keywords(
            texts=filtered["processed_text"],
            top_n=config.TOP_N_KEYWORDS,
            max_features=config.TFIDF_MAX_FEATURES,
            max_df=config.TFIDF_MAX_DF,
            min_df=config.TFIDF_MIN_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            candidate_multiplier=config.TFIDF_CANDIDATE_MULTIPLIER,
            news_stopwords=config.NEWS_STOPWORDS,
        )

    fig = build_bar_chart(keywords)
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough text data to extract keywords for the current filters.")


if __name__ == "__main__":
    run()
