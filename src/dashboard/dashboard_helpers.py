"""Utility functions for the NLP earnings report dashboard.

This module provides helper functions for the earnings report dashboard application.
It contains utilities for loading models, formatting data for visualization,
generating visualizations, and creating interactive components for the dashboard.

The module bridges the gap between the underlying NLP components (sentiment analysis,
topic modeling, feature extraction) and their presentation in the Streamlit dashboard.

Examples:
    Loading models for the dashboard:
    
    >>> from src.dashboard.dashboard_helpers import load_models
    >>> models = load_models()
    >>> print(f"Loaded {len(models)} models")
    
    Creating interactive components:
    
    >>> topic_explorer = create_topic_explorer(topic_model)
    >>> prediction_simulator = create_prediction_simulator(models, sample_text)
"""
import glob
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

# Import NLP components
from ..nlp.embedding import EmbeddingProcessor
from ..nlp.sentiment import SentimentAnalyzer
from ..nlp.topic_modeling import TopicModeler
from ..nlp.feature_extraction import FeatureExtractor
from ..nlp.nlp_processing import NLPProcessor

logger = logging.getLogger('dashboard.utils')

def load_models() -> Dict[str, Any]:
    """Load all available models for the dashboard application.
    
    Attempts to load all necessary NLP models for the dashboard including:
    - Embedding model for text representation
    - Sentiment analyzer for sentiment analysis
    - Topic model for topic extraction and visualization
    - Feature extractor for financial feature extraction
    
    The function handles exceptions for each model separately, allowing the 
    dashboard to function even if some models fail to load. Successful and 
    failed loads are logged appropriately.
    
    Args:
        None
        
    Returns:
        Dict[str, Any]: Dictionary of successfully loaded models where:
            - Keys are model types ('embedding', 'sentiment', 'topic', etc.)
            - Values are the corresponding model objects
            
    Example:
        >>> models = load_models()
        >>> if 'sentiment' in models:
        ...     sentiment_result = models['sentiment'].analyze(text)
        
    Notes:
        If a model fails to load, a warning is logged but execution continues.
        The function also attempts to load sample data, which is included in 
        the return dictionary under the key 'sample_data'.
    """
    models = {}
    
    try:
        # Try to load embedding model
        try:
            embedding_path = EMBEDDING_MODEL_PATH
            models['embedding'] = EmbeddingProcessor.load(embedding_path)
            logger.info(f"Embedding model loaded from {embedding_path}")
            
            # Store the NLPProcessor for reuse by other components
            if hasattr(models['embedding'], 'nlp_processor') and models['embedding'].nlp_processor is not None:
                models['nlp_processor'] = models['embedding'].nlp_processor
                logger.info("Using NLPProcessor from embedding model")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {str(e)}")
        
        # Try to load sentiment model
        try:
            sentiment_path = SENTIMENT_MODEL_PATH
            models['sentiment'] = SentimentAnalyzer.load(sentiment_path)
            
            # Connect to NLPProcessor if available
            if 'nlp_processor' in models and hasattr(models['sentiment'], 'set_nlp_processor'):
                models['sentiment'].set_nlp_processor(models['nlp_processor'])
                
            logger.info(f"Sentiment model loaded from {sentiment_path}")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {str(e)}")
        
        # Try to load topic model
        try:
            topic_path = TOPIC_MODEL_PATH
            models['topic'] = TopicModeler.load(topic_path)
            
            # Connect to NLPProcessor if available
            if 'nlp_processor' in models and hasattr(models['topic'], 'nlp_processor'):
                models['topic'].nlp_processor = models['nlp_processor']
                
            logger.info(f"Topic model loaded from {topic_path}")
        except Exception as e:
            logger.warning(f"Failed to load topic model: {str(e)}")
        
        # Try to load feature extractor
        try:
            feature_path = FEATURE_EXTRACTOR_PATH
            models['feature_extractor'] = FeatureExtractor.load(feature_path)
            
            # Connect to NLPProcessor if available and not already present
            if 'nlp_processor' in models and not hasattr(models['feature_extractor'], 'nlp_processor'):
                models['feature_extractor'].nlp_processor = models['nlp_processor']
                
            logger.info(f"Feature extractor loaded from {feature_path}")
        except Exception as e:
            # Try alternative paths with timestamp suffixes
            logger.warning(f"Failed to load feature extractor from primary path: {str(e)}")
            try:
                feature_dir = os.path.dirname(FEATURE_EXTRACTOR_PATH)
                feature_base = os.path.basename(FEATURE_EXTRACTOR_PATH)
                alternative_paths = glob.glob(os.path.join(feature_dir, f"{feature_base}_*"))
                
                if alternative_paths:
                    # Sort by modification time to get the newest
                    newest_path = max(alternative_paths, key=os.path.getmtime)
                    models['feature_extractor'] = FeatureExtractor.load(newest_path)
                    
                    # Connect to NLPProcessor if available
                    if 'nlp_processor' in models:
                        models['feature_extractor'].nlp_processor = models['nlp_processor']
                        
                    logger.info(f"Feature extractor loaded from alternative path: {newest_path}")
                else:
                    logger.warning(f"No alternative feature extractor paths found")
            except Exception as alt_e:
                logger.warning(f"Failed to load feature extractor from any path: {str(alt_e)}")
        
        # Try to load sample data
        try:
            # First try to load the smaller sample dataset specifically created for the dashboard
            sample_file = os.path.join(PROCESSED_DATA_DIR, "sample_train_edad7fda80.csv")
            
            if os.path.exists(sample_file):
                # Load the smaller sample file if it exists
                models['sample_data'] = pd.read_csv(sample_file)
                logger.info(f"Loaded {len(models['sample_data'])} sample data records from smaller sample file: {sample_file}")
            else:
                # Fall back to looking for any train files, but limit the number of rows
                train_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "train_*.csv"))
                if train_files:
                    # Sort by modification time and get the most recent
                    most_recent = max(train_files, key=os.path.getmtime)
                    
                    # Load just 1000 rows from the large file to keep memory usage reasonable
                    models['sample_data'] = pd.read_csv(most_recent, nrows=1000)
                    logger.info(f"Loaded {len(models['sample_data'])} sample data records (limited) from {most_recent}")
                    logger.warning("Using limited rows from full training dataset. Consider running create_dashboard_sample.py to create a proper sample.")
                else:
                    logger.warning(f"No training data files found in {PROCESSED_DATA_DIR}")
        except Exception as e:
            logger.warning(f"Failed to load sample data: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
    
    return models


def get_available_models() -> Dict[str, List[Dict[str, Any]]]:
    """Get list of all available models in the model zoo for the dashboard."""
    model_types = {
        'embedding': [],
        'sentiment': [],
        'topic': [],
        'feature_extractor': []
    }
    
    # Add only the models that actually exist in your implementation
    
    # Add embedding model
    model_types['embedding'].append({
        'name': 'TF-IDF Embeddings',
        'description': 'Text vectorization using TF-IDF with 5000 features',
        'version': '1.0.0',
        'created_at': datetime.now().strftime('%Y-%m-%d')
    })
    
    # Add sentiment model
    model_types['sentiment'].append({
        'name': 'Loughran-McDonald Lexicon',
        'description': 'Finance-specific sentiment lexicon with positive/negative/uncertainty scores',
        'version': '1.0.0',
        'created_at': datetime.now().strftime('%Y-%m-%d')
    })
    
    # Add topic model
    model_types['topic'].append({
        'name': 'LDA Topic Model',
        'description': 'Latent Dirichlet Allocation model for topic extraction',
        'version': '1.0.0',
        'created_at': datetime.now().strftime('%Y-%m-%d')
    })
    
    # Add feature extractor
    model_types['feature_extractor'].append({
        'name': 'Financial Metrics Extractor',
        'description': 'Extracts financial metrics, embeddings, sentiment and topic features',
        'version': '1.0.0',
        'created_at': datetime.now().strftime('%Y-%m-%d')
    })
    
    return model_types

def format_topics(topic_model: TopicModeler) -> pd.DataFrame:
    """Format topic model data into a DataFrame for visualization."""
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
            try:
                words = topic_model.get_topic_words(i, top_n=10)
                if isinstance(words, list):
                    if words and isinstance(words[0], tuple):
                        # (word, score) format
                        word_str = ", ".join([w[0] for w in words[:5]])
                    else:
                        word_str = ", ".join(words[:5])
                    topic_dict['top_words'] = word_str
            except Exception as e:
                logger.warning(f"Could not get topic words for topic {i}: {str(e)}")
                topic_dict['top_words'] = ""
        
        # Add topic weight/prevalence if available
        if hasattr(topic_model, 'topic_weights'):
            topic_dict['weight'] = topic_model.topic_weights[i]
        
        topics_data.append(topic_dict)
    
    return pd.DataFrame(topics_data)


def classify_sentiment(text: str, sentiment_model: SentimentAnalyzer) -> Dict[str, float]:
    """Analyze sentiment of financial text using the provided sentiment model."""
    if not text or not isinstance(text, str):
        return {"positive": 0, "negative": 0, "uncertainty": 0}
    
    try:
        return sentiment_model.analyze(text)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {"positive": 0, "negative": 0, "uncertainty": 0}


def format_sentiment_result(result: Dict[str, float]) -> pd.DataFrame:
    """Format sentiment analysis results for dashboard visualization."""
    # Handle empty result
    if not result:
        return pd.DataFrame(columns=['Dimension', 'Score'])
        
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
    """Extract or generate interactive visualization HTML from topic model."""
    # First check if we can use a better pyLDAvis visualization
    try:
        import pyLDAvis
        import pyLDAvis.lda_model
        
        if hasattr(topic_model, 'model') and hasattr(topic_model, 'dtm'):
            if hasattr(topic_model, 'nlp_processor') and topic_model.nlp_processor is not None:
                vectorizer = topic_model.nlp_processor.count_vectorizer
                if vectorizer is not None:
                    prepared_data = pyLDAvis.lda_model.prepare(
                        topic_model.model, topic_model.dtm, vectorizer
                    )
                    return pyLDAvis.prepared_data_to_html(prepared_data)
    except Exception as e:
        logger.warning(f"Could not create pyLDAvis visualization: {str(e)}")
    
    # First try to use the model's built-in visualization if available
    if hasattr(topic_model, 'get_visualization_html'):
        try:
            html = topic_model.get_visualization_html()
            if html:
                return html
        except Exception as e:
            logger.warning(f"Built-in visualization failed: {str(e)}")
    
    # If no built-in visualization, create one with Plotly
    try:
        import plotly.graph_objects as go
        import numpy as np
        
        # Get number of topics
        num_topics = topic_model.num_topics if hasattr(topic_model, 'num_topics') else 10
        
        # Function to get topic words
        def get_topic_words(topic_id, num_words=10):
            try:
                if hasattr(topic_model, 'get_topic_words'):
                    return topic_model.get_topic_words(topic_id, num_words)
                elif hasattr(topic_model, 'topic_words') and topic_model.topic_words:
                    if topic_id in topic_model.topic_words:
                        return topic_model.topic_words[topic_id][:num_words]
            except Exception:
                pass
            return []
        
        # Generate topic positions in 2D space
        np.random.seed(42)  # For reproducibility
        x_pos = np.random.normal(0, 1, size=num_topics)
        y_pos = np.random.normal(0, 1, size=num_topics)
        
        # Calculate topic sizes based on word counts
        sizes = [len(get_topic_words(i, 20)) for i in range(num_topics)]
        sizes = [50 + s*5 for s in sizes]  # Scale for visibility
        
        # Create hover texts
        hover_texts = []
        for i in range(num_topics):
            words = get_topic_words(i, 10)
            if isinstance(words, list):
                if words and isinstance(words[0], tuple):
                    topic_words_text = ", ".join([w[0] for w in words[:8]])
                else:
                    topic_words_text = ", ".join([str(w) for w in words[:8]])
            else:
                topic_words_text = str(words)
            hover_texts.append(f"Topic {i}<br>{topic_words_text}")
        
        # Create the scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=list(range(num_topics)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Topic ID")
            ),
            text=[f"{i}" for i in range(num_topics)],
            hovertext=hover_texts,
            hoverinfo='text',
            name='Topics'
        ))
        
        fig.update_layout(
            title="Topic Similarity Map (2D Projection)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            width=800,
            height=600
        )
        
        # Convert to HTML
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
        
    except Exception as e:
        logger.error(f"Error creating topic visualization: {str(e)}")
        return ""


def get_feature_importance_plot(feature_extractor: FeatureExtractor) -> Optional[plt.Figure]:
    """Generate a feature importance visualization from the feature extractor."""
    # Try to get feature importances from feature_extractor using different approaches
    importances = None
    
    try:
        # Try first with get_top_features if available
        if hasattr(feature_extractor, 'get_top_features'):
            top_features = feature_extractor.get_top_features(15)
            if isinstance(top_features, pd.DataFrame) and 'feature' in top_features.columns and 'importance' in top_features.columns:
                features = top_features['feature'].tolist()
                importance = top_features['importance'].tolist()
                importances = (features, importance)
        
        # If not available, try with feature_importances attribute
        if importances is None and hasattr(feature_extractor, 'feature_importances'):
            if isinstance(feature_extractor.feature_importances, pd.DataFrame):
                if 'feature' in feature_extractor.feature_importances.columns and 'importance' in feature_extractor.feature_importances.columns:
                    df = feature_extractor.feature_importances.copy()
                    df['abs_importance'] = df['importance'].abs()
                    df = df.sort_values('abs_importance', ascending=False).head(15)
                    features = df['feature'].tolist()
                    importance = df['importance'].tolist()
                    importances = (features, importance)
                    
        # If still no importances, return None
        if importances is None:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Unpack feature names and importance values
        features, importance = importances
        
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


def get_wordcloud_for_topic(topic_model: TopicModeler, topic_id: int) -> Optional[str]:
    """Generate a word cloud visualization for a specific topic in the topic model."""
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
    """Create a configuration for the earnings report prediction simulator component."""
    simulator = {
        'has_models': len(models) > 0,
        'initial_text': sample_text,
        'available_models': {
            'sentiment': 'sentiment' in models,
            'topics': 'topic' in models,
            'features': 'feature_extractor' in models
        }
    }
    
    # Add prediction functions with improved error handling
    
    # Sentiment prediction function
    def predict_sentiment(text):
        if not text or 'sentiment' not in models:
            return None
        try:
            return models['sentiment'].analyze(text)
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {str(e)}")
            return None
    
    # Topic prediction function
    def predict_topics(text):
        if not text or 'topic' not in models:
            return None
        try:
            # Make sure we have a string input - handle both string and DataFrame inputs
            if isinstance(text, pd.DataFrame):
                if 'text' in text.columns and len(text) > 0:
                    text = text['text'].iloc[0]
                else:
                    text = str(text.iloc[0])
            
            # Use extract_topics with proper single-text handling
            return models['topic'].extract_topics([text])
        except Exception as e:
            logger.error(f"Error in topic prediction: {str(e)}")
            return None
    
    # Feature extraction function
    def extract_features(text):
        if not text or 'feature_extractor' not in models:
            return {}
        try:
            # Make sure we're using the proper input format
            if not isinstance(text, pd.DataFrame):
                text_df = pd.DataFrame({'text': [text]})
            else:
                text_df = text
                
            # Use extract_financial_metrics with the DataFrame input
            return models['feature_extractor'].extract_financial_metrics(text_df)
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    # Set the prediction functions
    simulator['predict_sentiment'] = predict_sentiment
    simulator['predict_topics'] = predict_topics
    simulator['extract_features'] = extract_features
    
    return simulator


def create_topic_explorer(topic_model: Optional[TopicModeler]) -> Dict[str, Any]:
    """Create a configuration for the interactive topic explorer component."""
    if topic_model is None:
        return {'has_model': False}
    
    # Make sure we can safely access num_topics
    num_topics = 0
    if hasattr(topic_model, 'num_topics'):
        num_topics = topic_model.num_topics
    elif hasattr(topic_model, 'model') and hasattr(topic_model.model, 'n_components'):
        num_topics = topic_model.model.n_components
    
    # Safe implementation of get_topic_words
    def get_topic_words(topic_id, n=10):
        try:
            if hasattr(topic_model, 'get_topic_words'):
                return topic_model.get_topic_words(topic_id, n)
            # Fall back to other approaches if needed
            return []
        except Exception as e:
            logger.warning(f"Error getting topic words: {str(e)}")
            return []
    
    # Safe implementation of get_wordcloud
    def get_wordcloud(topic_id):
        try:
            return get_wordcloud_for_topic(topic_model, topic_id)
        except Exception as e:
            logger.warning(f"Error generating wordcloud: {str(e)}")
            return None
    
    explorer = {
        'has_model': True,
        'num_topics': num_topics,
        'topics_df': format_topics(topic_model),
        'visualization_html': extract_topic_visualization(topic_model),
        'get_topic_words': get_topic_words,
        'get_wordcloud': get_wordcloud
    }
    
    return explorer