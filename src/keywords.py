"""
src/keywords.py — TF-IDF keyword extraction with substring deduplication.

Called from dashboard.py on filtered articles (never during training).

Public API
----------
dynamic_min_df(tfidf_min_df, n_docs)        → int
deduplicate_keywords(candidates)             → list[tuple[str, float]]
extract_keywords(texts, top_n, ...)          → list[tuple[str, float]]
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def dynamic_min_df(tfidf_min_df: int, n_docs: int) -> int:
    """Return effective min_df scaled to corpus size.

    Formula: min(tfidf_min_df, max(1, n_docs // 10))
    Prevents TF-IDF from filtering everything when the filtered corpus is small.
    """
    return min(tfidf_min_df, max(1, n_docs // 10))


def deduplicate_keywords(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Remove unigrams that appear as a word component inside a bigram phrase.

    Rule: if token A is a word in phrase B (and A != B), remove A.
    Example: "trump" inside "donald trump" → remove "trump", keep "donald trump".

    Preserves original ordering of surviving entries.
    """
    if not candidates:
        return []

    # Collect all phrases (space-separated multi-word terms)
    all_phrases = [kw for kw, _ in candidates]

    # Build set of all individual words that appear inside any multi-word phrase
    substrings_to_remove: set[str] = set()
    for phrase in all_phrases:
        words = phrase.split()
        if len(words) > 1:
            for word in words:
                substrings_to_remove.add(word)

    # Keep a keyword if it is NOT in the remove set,
    # OR if it is itself a multi-word phrase (bigrams are never removed by this rule)
    result = [
        (kw, score)
        for kw, score in candidates
        if kw not in substrings_to_remove or " " in kw
    ]
    return result


def extract_keywords(
    texts: pd.Series,
    top_n: int,
    max_features: int,
    max_df: float,
    min_df: int,
    ngram_range: tuple,
    candidate_multiplier: int,
    news_stopwords: set | None = None,
) -> list[tuple[str, float]]:
    """Fit a fresh TF-IDF vectorizer on *texts* and return top-N deduplicated keywords.

    Parameters
    ----------
    texts : pd.Series of preprocessed article strings
    top_n : number of keywords to return after deduplication
    max_features : TF-IDF vocabulary cap
    max_df : ignore terms appearing in > max_df fraction of documents
    min_df : base min document frequency (scaled down dynamically for small corpora)
    ngram_range : (min_n, max_n) for n-gram extraction
    candidate_multiplier : extract top_n * candidate_multiplier candidates before dedup
    news_stopwords : set of domain-specific words to exclude from results

    Returns
    -------
    list of (keyword, tfidf_score) tuples, at most top_n items.

    May return fewer than top_n items if deduplication or TF-IDF filtering
    (max_df/min_df constraints) leaves insufficient candidates. Callers should
    handle the short-result case (e.g. skip chart rendering when result is empty).

    Returns [] silently when texts is empty, all-NaN, or TF-IDF produces an
    empty vocabulary (e.g. very narrow date/topic filter in the dashboard).
    """
    if news_stopwords is None:
        news_stopwords = set()

    texts = texts.dropna()
    if texts.empty:
        return []

    effective_min_df = dynamic_min_df(min_df, len(texts))
    n_candidates = top_n * candidate_multiplier

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=max_df,
        min_df=effective_min_df,
        ngram_range=ngram_range,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Empty vocabulary: all terms filtered by max_df/min_df constraints.
        # Happens with very narrow date/topic filters. Return empty gracefully.
        return []

    # Mean TF-IDF score across all documents for each term
    mean_scores = tfidf_matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    # Pick top candidates by mean score, excluding news-domain stopwords.
    # Exact-match only — bigrams like "new york" / "donald trump" / "white house"
    # are NOT blocked even though they contain stopword tokens.
    top_indices = mean_scores.argsort()[::-1][:n_candidates]
    candidates = [
        (feature_names[i], float(mean_scores[i]))
        for i in top_indices
        if feature_names[i] not in news_stopwords
    ]

    deduplicated = deduplicate_keywords(candidates)
    return deduplicated[:top_n]
