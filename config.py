# config.py — all tunable parameters for the News Dashboard pipeline

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/News_Category_Dataset_v3.json"

# ── Output paths ─────────────────────────────────────────────────────────────
CLEAN_DATA_PATH     = "output/clean_data.parquet"
PROCESSED_DATA_PATH = "output/processed_data.parquet"
TOPICS_DATA_PATH    = "output/articles_with_topics.parquet"

# ── Model paths ───────────────────────────────────────────────────────────────
LDA_MODEL_PATH        = "models/lda_model.joblib"
COUNT_VECTORIZER_PATH = "models/count_vectorizer.joblib"
TOPIC_NAMES_PATH      = "models/topic_names.json"

# ── LDA / CountVectorizer ─────────────────────────────────────────────────────
NUM_TOPICS     = 8
MAX_FEATURES   = 5000
CV_MAX_DF      = 0.85
CV_MIN_DF      = 3
LDA_MAX_ITER   = 15
LDA_BATCH_SIZE = 512

# ── Topic filtering ───────────────────────────────────────────────────────────
THRESHOLD = 0.2   # min topic probability for an article to match a topic

# ── TF-IDF keyword extraction ─────────────────────────────────────────────────
TOP_N_KEYWORDS             = 5
TOP_N_WORDCLOUD            = 20
TFIDF_MAX_FEATURES         = 5000
TFIDF_MAX_DF               = 0.5
TFIDF_MIN_DF               = 3
TFIDF_NGRAM_RANGE          = (1, 2)
TFIDF_CANDIDATE_MULTIPLIER = 10  # extract TOP_N * 10 candidates before deduplication

# News-domain stopwords: common words that appear across all topics and carry
# no discriminating meaning. Complements spaCy's grammatical stopwords.
NEWS_STOPWORDS = {
    # High-frequency generic words (appear in >40% of all filter combos)
    "new", "say", "year", "people", "day", "time",
    # Generic verbs
    "like", "know", "want", "find", "look", "get", "need", "way",
    "work", "make", "come", "go", "tell", "watch", "check", "back",
    "help", "think", "face", "play", "news", "report",
    "call", "take", "feel", "let", "happen", "give", "read",
    "learn", "matter", "wear", "write", "sign", "buy",
    # Generic adjectives
    "good", "thing", "life", "love", "old", "right", "late",
    "well", "bad", "big", "sure",
    "best", "easy", "favorite", "great", "high", "little", "long", "small", "social",
    # Generic nouns
    "photo", "photos", "video", "week", "home", "star", "world",
    "woman", "man", "host", "white", "post",
    "reason", "place", "moment", "problem", "sale", "care",
    "red", "heart", "dress", "team", "story", "street",
    # Platform / media names (appear as article formatting artifacts)
    "huffpost", "pinterest", "tumblr",
    # HuffPost section name leaking as standalone token
    "rise",
    # Partial place names (appear standalone when "New York" bigram is filtered)
    "york",
    # Partial proper nouns (appear alone when bigram doesn't rank high enough)
    "donald",
    # Explicit CTA bigrams from old HuffPost article formatting
    "check huffpost", "twitter facebook", "huffpost rise",
    # Words appearing in >25% of all year×topic combos at top-20 (wordcloud expansion)
    "state", "child",
    "black", "change", "kid", "live", "food", "show", "family",
    "country", "stop", "end", "try",
}
