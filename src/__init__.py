"""
NLP Earnings Report Analysis - Main Package
This package contains tools for analyzing earnings report texts using NLP techniques.
"""

# Import core modules
from . import data
from . import nlp
from . import dashboard
from . import models
from . import utils

# Import key functions for convenience
from .config import (
    ROOT_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR, DOC_DIR,
    EMBEDDING_MODEL_PATH, SENTIMENT_MODEL_PATH, TOPIC_MODEL_PATH, FEATURE_EXTRACTOR_PATH,
    RAW_DATA_PATH, PROCESSED_DATA_DIR
)

__all__ = [
    'data', 
    'nlp', 
    'dashboard', 
    'models',
    'utils',
    'ROOT_DIR',
    'DATA_DIR',
    'MODEL_DIR',
    'OUTPUT_DIR',
    'DOC_DIR'
]