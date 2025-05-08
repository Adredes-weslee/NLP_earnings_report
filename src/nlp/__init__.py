"""
Core NLP module for earnings report text analysis.
Contains tools for text embedding, sentiment analysis, topic modeling,
and feature extraction from financial texts.
"""

# Import NLP components
from .embedding import EmbeddingProcessor
from .sentiment import SentimentAnalyzer
from .topic_modeling import TopicModeler
from .feature_extraction import FeatureExtractor

__all__ = [
    'EmbeddingProcessor',
    'SentimentAnalyzer',
    'TopicModeler',
    'FeatureExtractor'
]