"""
train.py — Pipeline orchestrator for the News Dashboard.

Runs the three training steps in order:
  1. src/ingest.py  — PySpark batch ingestion → output/clean_data.parquet
  2. src/preprocess.py — spaCy cleaning → output/processed_data.parquet
  3. src/lda_model.py  — LDA training  → output/articles_with_topics.parquet + models/

Each step is skipped if its output already exists, unless --retrain is passed.

Usage:
    python train.py             # skip steps whose output exists
    python train.py --retrain   # force-run all steps
"""

import argparse
import os

import config
from src import ingest, preprocess, lda_model


def run_pipeline(retrain=False):
    """Orchestrate the three pipeline steps.

    Args:
        retrain: If True, run all steps even when output files already exist.
    """
    # ── Step 1 — Ingestion ────────────────────────────────────────────────────
    if retrain or not os.path.exists(config.CLEAN_DATA_PATH):
        print("[train] Step 1/3 — Running ingestion ...")
        ingest.run()
    else:
        print(f"[train] Step 1/3 — Skipping ingestion ({config.CLEAN_DATA_PATH} already exists)")

    # ── Step 2 — Preprocessing ────────────────────────────────────────────────
    if retrain or not os.path.exists(config.PROCESSED_DATA_PATH):
        print("[train] Step 2/3 — Running preprocessing ...")
        preprocess.run()
    else:
        print(f"[train] Step 2/3 — Skipping preprocessing ({config.PROCESSED_DATA_PATH} already exists)")

    # ── Step 3 — LDA training ─────────────────────────────────────────────────
    if retrain or not os.path.exists(config.TOPICS_DATA_PATH):
        print("[train] Step 3/3 — Running LDA training ...")
        lda_model.run()
    else:
        print(f"[train] Step 3/3 — Skipping LDA training ({config.TOPICS_DATA_PATH} already exists)")

    print("[train] Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="News Dashboard pipeline orchestrator"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-run of all steps even if output files already exist",
    )
    args = parser.parse_args()
    run_pipeline(retrain=args.retrain)
