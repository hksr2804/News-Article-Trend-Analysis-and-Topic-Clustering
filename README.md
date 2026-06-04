# News Article Trend Analysis & Topic Clustering

An end-to-end NLP pipeline and interactive dashboard that discovers hidden topics across **~190,000 HuffPost news articles (2012–2022)** and lets users explore trending keywords by date range and topic.

**Live demo:** https://github.com/hksr2804/News-Article-Trend-Analysis-and-Topic-Clustering (deployed on Streamlit Community Cloud)

---

## Overview

The project ingests ~210K raw news articles, cleans them down to 189,814 usable records, uses **Latent Dirichlet Allocation (LDA)** to discover 8 latent topics, and serves an interactive **Streamlit** dashboard. Users filter articles by date and topic; the app computes **TF-IDF** keywords live on the filtered subset and renders them as a bar chart and word cloud.

The heavy ML pipeline runs **once offline** (~20 min) and persists results to disk, so the dashboard stays fast — it only reads pre-computed topic probabilities and runs lightweight TF-IDF at query time.

## Key Results & Highlights

- Processed and modeled **189,814 articles** spanning a 10-year period.
- Selected the optimal topic count (**k=8**) empirically via **coherence-score evaluation** across k=5–10 (best CV coherence: 0.4417).
- Built a fully **tested codebase — 101 passing tests** developed test-first (TDD).
- **Deployed to production** on Streamlit Community Cloud with auto-redeploy on every push.
- Designed a custom **domain-stopword + bigram-deduplication** keyword pipeline to surface meaningful trends instead of generic filler words.

## Architecture

```
Raw JSON (~210K articles)
   │  src/ingest.py      — PySpark: read, clean, filter, cast dates → Parquet
   ▼
clean_data.parquet (189,814 rows)
   │  src/preprocess.py  — spaCy: lemmatize, strip stopwords/punctuation/numbers
   ▼
processed_data.parquet
   │  src/lda_model.py   — scikit-learn: CountVectorizer + LDA (8 topics)
   ▼
articles_with_topics.parquet  +  models/ (LDA + vectorizer)
   │  src/dashboard.py   — Streamlit: filter + live TF-IDF + charts
   ▼
Interactive dashboard
```

## Tech Stack

| Area | Technologies |
|---|---|
| Data ingestion | **PySpark** (distributed batch processing → Parquet) |
| NLP preprocessing | **spaCy** (`en_core_web_sm`, lemmatization, tokenization) |
| Topic modeling | **scikit-learn** — CountVectorizer + Latent Dirichlet Allocation |
| Topic-count selection | **gensim** coherence scoring |
| Keyword extraction | **scikit-learn** TfidfVectorizer (unigrams + bigrams) |
| Dashboard / UI | **Streamlit** |
| Visualization | **matplotlib**, **wordcloud**, **Pillow** |
| Data & serialization | **pandas**, **PyArrow/Parquet**, **joblib** |
| Testing | **pytest** (101 tests) |
| Deployment | **GitHub** + **Streamlit Community Cloud** |

## How It Works

1. **Ingest** — PySpark reads the raw JSON, drops incomplete rows, concatenates `headline` + `short_description`, casts dates, and writes typed Parquet.
2. **Preprocess** — spaCy lemmatizes each article and removes stopwords, punctuation, numbers, and short/noisy tokens.
3. **Topic modeling** — CountVectorizer builds a 189,814 × 5,000 document-term matrix; LDA learns 8 topics. Each article gets a probability distribution over the 8 topics, saved alongside its date.
4. **Dashboard** — User picks a date range and topics. The app filters articles (topic match = probability ≥ 0.2), runs TF-IDF on that subset, deduplicates unigrams contained in bigrams, removes domain stopwords, and renders the top keywords as a bar chart (top 5) and word cloud (top 20).

The 8 discovered topics were manually labeled after inspecting top words: *Health & Wellness, Politics, News & Current Affairs, Entertainment, Human Interest, Public Affairs, Travel & Leisure, Society.*

## Getting Started

**Run the dashboard** (uses committed pre-trained models — no retraining needed):

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

**Retrain the full pipeline** (requires the raw dataset, Java 17, plus PySpark & spaCy):

```bash
pip install pyspark==4.1.1 spacy==3.8.14
python -m spacy download en_core_web_sm

python train.py            # skips steps whose output already exists
python train.py --retrain  # force re-run every step
```

> The dataset (`data/News_Category_Dataset_v3.json`, HuffPost News Category Dataset from Kaggle) is excluded from the repo. Only the final topic-tagged Parquet and trained models are committed so the dashboard runs out of the box.

## Project Structure

```
config.py            # All tunable parameters and file paths
train.py             # Pipeline orchestrator (ingest → preprocess → LDA)
src/
  ingest.py          # PySpark ingestion + cleaning
  preprocess.py      # spaCy lemmatization
  lda_model.py       # CountVectorizer + LDA training
  keywords.py        # TF-IDF keyword extraction + deduplication
  dashboard.py       # Streamlit app
models/              # Trained LDA model, vectorizer, topic names
output/              # Final topic-tagged dataset (Parquet)
tests/               # pytest suite (101 tests)
```

## Notable Engineering Decisions

- **Offline LDA, online TF-IDF** — topic distributions are pre-computed once; keyword extraction runs live so it adapts to each filtered subset (e.g. surfacing "covid"/"vaccine" for Health articles in 2020).
- **Coherence-driven topic count** — k=8 chosen from quantitative evaluation, not guesswork.
- **Domain stopwords + bigram preservation** — a curated news-domain stoplist removes generic words ("say", "people", "year") while exact-match logic keeps meaningful bigrams ("new york", "white house") intact.
- **Dynamic `min_df`** — scales the TF-IDF document-frequency floor for narrow filters so small subsets don't produce an empty vocabulary.
