"""
src/lda_model.py — LDA topic modeling: train, transform, and save artifacts.

Usage (called from train.py):
    from src.lda_model import run
    run()
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import config


def train_lda(texts, num_topics, max_features, cv_max_df, cv_min_df,
              lda_max_iter, lda_batch_size):
    """Fit CountVectorizer and LDA on the given texts.

    Returns:
        (lda, cv): fitted LatentDirichletAllocation and CountVectorizer
    """
    cv = CountVectorizer(
        max_features=max_features,
        max_df=cv_max_df,
        min_df=cv_min_df,
    )
    dtm = cv.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method="online",
        random_state=42,
        max_iter=lda_max_iter,
        batch_size=lda_batch_size,
    )
    lda.fit(dtm)
    return lda, cv


def get_topic_distribution(lda, cv, texts):
    """Transform texts into a (n_docs, num_topics) probability array.

    Returns:
        numpy ndarray of shape (len(texts), lda.n_components)
    """
    dtm = cv.transform(texts)
    return lda.transform(dtm)


def build_topics_df(df, topic_matrix, num_topics):
    """Assemble the output DataFrame.

    Args:
        df: DataFrame with columns [processed_text, date]
        topic_matrix: numpy array of shape (n_docs, num_topics)
        num_topics: number of topics

    Returns:
        DataFrame with columns: processed_text, date, topic_0, ..., topic_N
    """
    topic_cols = {f"topic_{i}": topic_matrix[:, i] for i in range(num_topics)}
    result = df[["processed_text", "date"]].copy().reset_index(drop=True)
    for col, values in topic_cols.items():
        result[col] = values
    return result


def init_topic_names_file(path, num_topics):
    """Write placeholder topic names JSON — skips if file already exists.

    Creates parent directory if needed. Safe to call standalone without
    pre-creating the models/ directory.

    Args:
        path: destination file path
        num_topics: number of topics
    """
    if os.path.exists(path):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    placeholders = {str(i): f"Topic {i}" for i in range(num_topics)}
    with open(path, "w") as f:
        json.dump(placeholders, f, indent=2)


def save_artifacts(lda, cv, topics_df, lda_path, cv_path, topics_path):
    """Save LDA model, CountVectorizer, and topics DataFrame to disk.

    Args:
        lda: fitted LatentDirichletAllocation
        cv: fitted CountVectorizer
        topics_df: DataFrame from build_topics_df
        lda_path: path for lda_model.joblib
        cv_path: path for count_vectorizer.joblib
        topics_path: path for articles_with_topics.parquet
    """
    for path in (lda_path, cv_path, topics_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    joblib.dump(lda, lda_path)
    joblib.dump(cv, cv_path)
    topics_df.to_parquet(topics_path, index=False)


def run():
    """Orchestrate Step 3: load preprocessed data, train LDA, save all artifacts.

    Returns None. train.py determines whether to call this by checking for
    the output file (config.TOPICS_DATA_PATH) rather than inspecting a return value.
    """
    print("Loading processed data...")
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    df["processed_text"] = df["processed_text"].fillna("")

    # Use Series (not .tolist()) to avoid a full in-memory copy of ~189K strings.
    # CountVectorizer accepts any iterable, including a pandas Series.
    texts = df["processed_text"]

    print(f"Training LDA ({config.NUM_TOPICS} topics) on {len(texts):,} articles...")
    lda, cv = train_lda(
        texts=texts,
        num_topics=config.NUM_TOPICS,
        max_features=config.MAX_FEATURES,
        cv_max_df=config.CV_MAX_DF,
        cv_min_df=config.CV_MIN_DF,
        lda_max_iter=config.LDA_MAX_ITER,
        lda_batch_size=config.LDA_BATCH_SIZE,
    )

    print("Computing topic distributions...")
    topic_matrix = get_topic_distribution(lda, cv, texts)

    topics_df = build_topics_df(df, topic_matrix, config.NUM_TOPICS)

    print("Saving models and output parquet...")
    save_artifacts(
        lda, cv, topics_df,
        config.LDA_MODEL_PATH,
        config.COUNT_VECTORIZER_PATH,
        config.TOPICS_DATA_PATH,
    )

    init_topic_names_file(config.TOPIC_NAMES_PATH, config.NUM_TOPICS)

    print(f"Done. Articles saved to {config.TOPICS_DATA_PATH}")
    print(f"Models saved to {config.LDA_MODEL_PATH}, {config.COUNT_VECTORIZER_PATH}")
    print(f"Topic names placeholder: {config.TOPIC_NAMES_PATH}")
