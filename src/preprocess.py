"""
src/preprocess.py — spaCy text preprocessing pipeline

Loads output/clean_data.parquet, processes each article through spaCy
(lemmatize, lowercase, remove stopwords/punctuation/numbers/short tokens),
and saves output/processed_data.parquet.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import spacy

import config


def preprocess_texts(texts, nlp):
    """
    Process a list of raw text strings through the spaCy pipeline.

    For each document, keeps tokens that are:
    - Not a stopword
    - Not punctuation
    - Not a number
    - Alphabetic only (no special chars)
    - Longer than 2 characters

    Each kept token is lemmatized and lowercased.
    Returns a list of space-separated processed strings, one per input text.
    """
    results = []
    for doc in nlp.pipe(texts, batch_size=500):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.is_alpha
            and len(token) > 2
        ]
        results.append(" ".join(tokens))
    return results


def run():
    print("[preprocess] Loading clean data ...")
    df = pd.read_parquet(config.CLEAN_DATA_PATH)
    print(f"[preprocess] Rows loaded: {len(df):,}")

    print("[preprocess] Loading spaCy model (en_core_web_sm) ...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Guard against any null text rows so nlp.pipe() never receives None
    df["text"] = df["text"].fillna("")

    texts = df["text"].tolist()
    total = len(texts)
    processed = []

    print(f"[preprocess] Processing {total:,} articles (this may take 5-10 min) ...")

    # Process in chunks so we can log progress every 10K articles
    chunk_size = 10_000
    for start in range(0, total, chunk_size):
        chunk = texts[start:start + chunk_size]
        processed.extend(preprocess_texts(chunk, nlp))
        done = min(start + chunk_size, total)
        print(f"[preprocess]   {done:,} / {total:,} articles processed")

    # Build output frame with explicit index reset so date alignment is guaranteed
    df_out = df[["date"]].copy().reset_index(drop=True)
    df_out["processed_text"] = processed

    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH) or ".", exist_ok=True)
    print(f"[preprocess] Saving to {config.PROCESSED_DATA_PATH} ...")
    df_out.to_parquet(config.PROCESSED_DATA_PATH, index=False)
    print("[preprocess] Done.")


if __name__ == "__main__":
    run()
