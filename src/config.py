# config.py
"""
Configuration file for the NLP Earnings Report Analysis project.
Contains paths, parameters and constants used throughout the project.
"""

import os

# Directory paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'results')
DOC_DIR = os.path.join(ROOT_DIR, "docs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

# Model paths
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, 'embeddings', 'tfidf_5000')
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment', 'loughran_mcdonald')
TOPIC_MODEL_PATH = os.path.join(MODEL_DIR, 'topics', 'lda_model')
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, 'features', 'combined_features')

# Data paths
RAW_DATA_PATH = os.path.join(DATA_DIR, 'ExpTask2Data.csv.gz')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Data split parameters
TEST_SIZE = 0.2       # Percentage of data for test set
VAL_SIZE = 0.15       # Percentage of data for validation set
RANDOM_STATE = 42     
CV_FOLDS = 5          # Number of cross-validation folds
SAMPLE_SIZE = 5000    # Sample size for topic model tuning

# Model parameters
VOCAB_SIZE = 5000
NUM_TOPICS = 40
MAX_FEATURES = 5000   # Maximum number of features to use in NLP models
MAX_DOC_FREQ = 0.95   # Maximum document frequency for TF-IDF vectorizer
NGRAM_RANGE = (1, 2)  # Use both unigrams and bigrams
LARGE_RETURN_THRESHOLD = 0.05  # 5% threshold for binary classification

# Topic modeling parameters
TOPIC_RANGE_MIN = 10  # Minimum number of topics to try
TOPIC_RANGE_MAX = 10  # Maximum number of topics to try
TOPIC_RANGE_STEP = 5  # Step size for number of topics
TOPIC_WORD_PRIOR = 0.01  # Alpha prior for word-topic distributions
TOPIC_DOC_PRIOR = None  # Alpha prior for document-topic distributions (auto)
DOC_TOPIC_PRIOR_FACTOR = 50.0  # Factor to calculate default doc-topic prior as 50/num_topics
OPTIMAL_TOPICS = 40   # Default number of topics if optimized value not available
LDA_MAX_ITER = 15     # Maximum number of iterations for LDA
LDA_LEARNING_DECAY = 0.7  # Learning decay rate for LDA
LDA_LEARNING_OFFSET = 10.0  # Learning offset for LDA

# Hyperparameter tuning
N_ITER_SEARCH = 20    # Number of iterations for randomized search

# Regularization parameters
ALPHA_MIN = 0.0001     # Minimum alpha value for Lasso/Ridge regularization
ALPHA_MAX = 1.0        # Maximum alpha value for regularization
ALPHA_STEPS = 10       # Number of steps in regularization grid search
N_ALPHAS = 50          # Number of alpha values to try in regularization path

# Visualization
MAX_WORD_CLOUD_WORDS = 100
FIGURE_DPI = 300

# Text processing parameters
MIN_TOKEN_LENGTH = 3      # Minimum length for tokens to keep
MIN_SENTENCE_TOKENS = 5   # Minimum number of tokens for a sentence to be valid
MAX_FINANCIAL_NUM_RATIO = 0.4  # Maximum ratio of financial numbers in a sentence
