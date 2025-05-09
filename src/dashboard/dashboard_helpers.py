"""
Utility functions for the NLP earnings report dashboard.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from wordcloud import WordCloud



# Import configuration values
from ..config import (MODEL_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR, 
                  EMBEDDING_MODEL_PATH, SENTIMENT_MODEL_PATH, TOPIC_MODEL_PATH,
                  FEATURE_EXTRACTOR_PATH, MAX_WORD_CLOUD_WORDS, FIGURE_DPI)

# The environment variable is now set in app.py
# This comment is kept for reference

logger = logging.getLogger('dashboard.utils')

# Import NLP components
from ..nlp.embedding import EmbeddingProcessor
from ..nlp.sentiment import SentimentAnalyzer
from ..nlp.topic_modeling import TopicModeler
from ..nlp.feature_extraction import FeatureExtractor


def load_models() -> Dict[str, Any]:
    """
    Load all available models for the dashboard.
    
    Returns:
        Dict[str, Any]: Dictionary of loaded models.
    """
    models = {}
    
    try:
        # Try to load embedding model
        try:
            embedding_path = EMBEDDING_MODEL_PATH
            models['embedding'] = EmbeddingProcessor.load(embedding_path)
            logger.info(f"Embedding model loaded from {embedding_path}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {str(e)}")
        
        # Try to load sentiment model
        try:
            sentiment_path = SENTIMENT_MODEL_PATH
            models['sentiment'] = SentimentAnalyzer.load(sentiment_path)
            logger.info(f"Sentiment model loaded from {sentiment_path}")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {str(e)}")
        
        # Try to load topic model
        try:
            topic_path = TOPIC_MODEL_PATH
            models['topic'] = TopicModeler.load(topic_path)
            logger.info(f"Topic model loaded from {topic_path}")
        except Exception as e:
            logger.warning(f"Failed to load topic model: {str(e)}")
        
        # Try to load feature extractor
        try:
            feature_path = FEATURE_EXTRACTOR_PATH
            # Add this code here to ensure directory exists with proper permissions
            features_dir = os.path.dirname(feature_path)
            os.makedirs(features_dir, exist_ok=True)
            # Try to fix permissions if on Windows
            if os.name == 'nt':
                try:
                    import subprocess
                    subprocess.run(['icacls', features_dir, '/grant', f'{os.getenv("USERNAME")}:(F)'], 
                                capture_output=True)
                except Exception as perm_error:
                    logger.warning(f"Could not set permissions: {str(perm_error)}")
            
            models['feature_extractor'] = FeatureExtractor.load(feature_path)
            logger.info(f"Feature extractor loaded from {feature_path}")
        except Exception as e:
            logger.warning(f"Failed to load feature extractor: {str(e)}")
        
        # Try to load sample data
        try:
            # Get most recent training data from processed directory
            import glob
            train_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "train_*.csv"))
            if train_files:
                # Sort by modification time and get the most recent
                most_recent = max(train_files, key=os.path.getmtime)
                models['sample_data'] = pd.read_csv(most_recent)
                logger.info(f"Loaded {len(models['sample_data'])} sample data records from {most_recent}")
            else:
                logger.warning(f"No training data files found in {PROCESSED_DATA_DIR}")
        except Exception as e:
            logger.warning(f"Failed to load sample data: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
    
    return models


def get_available_models() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get list of all available models in the model zoo.
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary of available models by type.
    """
    model_types = {
        'embedding': [],
        'sentiment': [],
        'topic': [],
        'feature_extractor': []
    }
    
    # In a real application, this would search a models directory
    # For now, we'll add some demo models
    
    # Add embedding models
    model_types['embedding'].append({
        'name': 'FinBERT Embeddings',
        'description': 'Financial domain-specific BERT embeddings',
        'version': '1.0.0',
        'created_at': '2023-10-15'
    })
    
    model_types['embedding'].append({
        'name': 'Sentence Transformer',
        'description': 'General purpose sentence embeddings',
        'version': '2.0.0',
        'created_at': '2023-11-22'
    })
    
    # Add sentiment models
    model_types['sentiment'].append({
        'name': 'Financial Lexicon Analyzer',
        'description': 'Finance-specific sentiment lexicon',
        'version': '1.2.0',
        'created_at': '2023-09-05'
    })
    
    model_types['sentiment'].append({
        'name': 'Earnings Report Sentiment',
        'description': 'Fine-tuned BERT model for earnings sentiment',
        'version': '0.9.1',
        'created_at': '2023-12-10'
    })
    
    # Add topic models
    model_types['topic'].append({
        'name': 'LDA Topic Model',
        'description': 'Latent Dirichlet Allocation model with 20 topics',
        'version': '1.0.0',
        'created_at': '2023-08-18'
    })
    
    model_types['topic'].append({
        'name': 'BERTopic Model',
        'description': 'BERT embeddings with HDBSCAN clustering',
        'version': '0.8.5',
        'created_at': '2023-11-30'
    })
    
    # Add feature extractors
    model_types['feature_extractor'].append({
        'name': 'Financial Metrics Extractor',
        'description': 'Regex and rule-based financial metrics extraction',
        'version': '1.1.0',
        'created_at': '2023-10-02'
    })
    
    model_types['feature_extractor'].append({
        'name': 'NER-based Feature Extractor',
        'description': 'Named Entity Recognition for financial metrics',
        'version': '0.9.0',
        'created_at': '2023-12-05'
    })
    
    return model_types


def format_topics(topic_model: TopicModeler) -> pd.DataFrame:
    """
    Format topic model data into a DataFrame for visualization.
    
    Args:
        topic_model (TopicModeler): The topic model to format.
        
    Returns:
        pd.DataFrame: Formatted DataFrame of topics.
    """
    if not hasattr(topic_model, 'num_topics') or not topic_model.num_topics:
        return pd.DataFrame()
    
    # Create a DataFrame with topic information
    topics_data = []
    
    for i in range(topic_model.num_topics):
        topic_dict = {
            'topic_id': i,
            'topic_label': f"Topic {i}",
        }
        
        # Add topic words if available
        if hasattr(topic_model, 'get_topic_words'):
            words = topic_model.get_topic_words(i, top_n=10)
            if isinstance(words, list):
                if words and isinstance(words[0], tuple):
                    # (word, score) format
                    word_str = ", ".join([w[0] for w in words[:5]])
                else:
                    word_str = ", ".join(words[:5])
                topic_dict['top_words'] = word_str
        
        # Add topic weight/prevalence if available
        if hasattr(topic_model, 'topic_weights'):
            topic_dict['weight'] = topic_model.topic_weights[i]
        
        topics_data.append(topic_dict)
    
    return pd.DataFrame(topics_data)


def classify_sentiment(text: str, sentiment_model: SentimentAnalyzer) -> Dict[str, float]:
    """
    Classify sentiment using the sentiment model.
    
    Args:
        text (str): Text to analyze.
        sentiment_model (SentimentAnalyzer): The sentiment model to use.
        
    Returns:
        Dict[str, float]: Dictionary of sentiment scores.
    """
    return sentiment_model.analyze(text)


def format_sentiment_result(result: Dict[str, float]) -> pd.DataFrame:
    """
    Format sentiment result for visualization.
    
    Args:
        result (Dict[str, float]): Sentiment analysis results.
        
    Returns:
        pd.DataFrame: Formatted DataFrame of sentiment results.
    """
    # Convert to DataFrame for easier visualization
    sentiment_df = pd.DataFrame({
        'Dimension': list(result.keys()),
        'Score': list(result.values())
    })
    
    # Sort by absolute value of score to show strongest sentiments first
    sentiment_df['AbsScore'] = abs(sentiment_df['Score'])
    sentiment_df = sentiment_df.sort_values('AbsScore', ascending=False)
    sentiment_df = sentiment_df.drop('AbsScore', axis=1)
    
    return sentiment_df


def extract_topic_visualization(topic_model: TopicModeler) -> str:
    """
    Extract interactive visualization HTML from topic model if available.
    
    Args:
        topic_model (TopicModeler): The topic model.
        
    Returns:
        str: HTML string of visualization or empty string if not available.
    """
    if hasattr(topic_model, 'get_visualization_html'):
        return topic_model.get_visualization_html()
    return ""


def get_feature_importance_plot(feature_extractor: FeatureExtractor) -> Optional[plt.Figure]:
    """
    Get feature importance plot from feature extractor.
    
    Args:
        feature_extractor (FeatureExtractor): The feature extractor.
        
    Returns:
        Optional[plt.Figure]: Matplotlib figure or None if not available.
    """
    if not hasattr(feature_extractor, 'get_feature_importance'):
        return None
    
    # Get feature importance data
    try:
        feature_importance = feature_extractor.get_feature_importance()
        
        if not feature_importance:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        if isinstance(feature_importance, dict):
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
        else:
            # Assume it's a list of (feature, importance) tuples
            features = [f[0] for f in feature_importance]
            importance = [f[1] for f in feature_importance]
            
        # Sort features by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx[-15:]]  # Top 15 features
        importance = [importance[i] for i in sorted_idx[-15:]]
        
        # Plot horizontal bar chart
        ax.barh(features, importance)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error in get_feature_importance_plot: {str(e)}")
        return None


def get_wordcloud_for_topic(topic_model: TopicModeler, topic_id: int) -> Optional[bytes]:
    """
    Generate a word cloud image for a topic.
    
    Args:
        topic_model (TopicModeler): The topic model.
        topic_id (int): Topic ID to visualize.
        
    Returns:
        Optional[bytes]: Base64-encoded image or None if not available.
    """
    if not hasattr(topic_model, 'get_topic_words'):
        return None
    
    try:
        words = topic_model.get_topic_words(topic_id, top_n=50)
        
        if not words:
            return None
        
        # Create word cloud
        word_dict = {}
        
        if isinstance(words[0], tuple):
            # (word, weight) format
            for word, weight in words:
                word_dict[word] = weight
        else:
            # Just words, use decreasing weights
            for i, word in enumerate(words):
                word_dict[word] = (len(words) - i) / len(words)
          # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=MAX_WORD_CLOUD_WORDS
        ).generate_from_frequencies(word_dict)
        
        # Convert to image
        img = wordcloud.to_image()
        
        # Save to buffer
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        logger.error(f"Error in get_wordcloud_for_topic: {str(e)}")
        return None


def create_prediction_simulator(models: Dict[str, Any], sample_text: str) -> Dict[str, Any]:
    """
    Create a prediction simulator configuration.
    
    Args:
        models (Dict[str, Any]): Dictionary of loaded models.
        sample_text (str): Sample text to use for initial display.
        
    Returns:
        Dict[str, Any]: Dictionary with simulator configuration.
    """
    simulator = {
        'has_models': len(models) > 0,
        'initial_text': sample_text,
        'available_models': {
            'sentiment': 'sentiment' in models,
            'topics': 'topic' in models,
            'features': 'feature_extractor' in models
        }
    }
    
    # Add prediction functions
    if 'sentiment' in models:
        simulator['predict_sentiment'] = lambda text: models['sentiment'].analyze(text)
    else:
        simulator['predict_sentiment'] = lambda text: None
    
    if 'topic' in models:
        simulator['predict_topics'] = lambda text: models['topic'].extract_topics([text])
    else:
        simulator['predict_topics'] = lambda text: None
    
    if 'feature_extractor' in models:
        simulator['extract_features'] = lambda text: models['feature_extractor'].extract_features(text)
    else:
        simulator['extract_features'] = lambda text: None
    
    return simulator


def create_topic_explorer(topic_model: Optional[TopicModeler]) -> Dict[str, Any]:
    """
    Create a topic explorer configuration.
    
    Args:
        topic_model (Optional[TopicModeler]): The topic model or None.
        
    Returns:
        Dict[str, Any]: Dictionary with explorer configuration.
    """
    if topic_model is None:
        return {'has_model': False}
    
    explorer = {
        'has_model': True,
        'num_topics': topic_model.num_topics if hasattr(topic_model, 'num_topics') else 0,
        'topics_df': format_topics(topic_model),
        'visualization_html': extract_topic_visualization(topic_model),
        'get_topic_words': lambda topic_id, n=10: topic_model.get_topic_words(topic_id, n) if hasattr(topic_model, 'get_topic_words') else [],
        'get_wordcloud': lambda topic_id: get_wordcloud_for_topic(topic_model, topic_id)
    }
    
    return explorer