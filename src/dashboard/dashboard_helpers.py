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
            models['feature_extractor'] = FeatureExtractor.load(feature_path)
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
    """Get list of all available models in the model zoo for the dashboard.
    
    Creates a catalog of available pre-trained models organized by type
    (embedding, sentiment, topic, feature extraction). For each model,
    it provides metadata such as name, description, version, and creation date.
    
    This function can be extended to dynamically scan model directories
    and populate the catalog based on available model files. Currently,
    it returns a static list of demonstration models.
    
    Args:
        None
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary of available models by type where:
            - Keys are model types ('embedding', 'sentiment', 'topic', 'feature_extractor')
            - Values are lists of model information dictionaries containing:
                - name: Display name of the model
                - description: Brief description of the model
                - version: Version string
                - created_at: Creation date string
                
    Example:
        >>> available_models = get_available_models()
        >>> sentiment_models = available_models['sentiment']
        >>> for model in sentiment_models:
        ...     print(f"{model['name']} (v{model['version']}): {model['description']}")
    """
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
    """Format topic model data into a DataFrame for visualization.
    
    This function takes a topic model and formats its topic-word distributions
    into a structured DataFrame suitable for dashboard visualization. It includes
    topic IDs, top words for each topic, and relevance scores.
    
    Args:
        topic_model (TopicModeler): The trained topic model to format.
        
    Returns:
        pd.DataFrame: Formatted DataFrame of topics with columns:
            - topic_id: Numeric ID for each topic
            - top_words: String of top words that best represent the topic
            - word_weights: Dictionary mapping words to their weights in the topic
            - coherence: Topic coherence score if available
            
    Example:
        >>> topics_df = format_topics(topic_model)
        >>> print(f"Found {len(topics_df)} topics")
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
    """Analyze sentiment of financial text using the provided sentiment model.
    
    Processes the input text using the given sentiment analyzer to extract
    emotional tone and financial sentiment indicators. The function is a wrapper
    around the sentiment model's analyze method that provides consistent error
    handling and interface for the dashboard.
    
    Args:
        text (str): Financial text to analyze for sentiment.
        sentiment_model (SentimentAnalyzer): Initialized sentiment analyzer model
            that implements the analyze() method.
        
    Returns:
        Dict[str, float]: Dictionary of sentiment scores where:
            - Keys are sentiment dimensions (e.g., 'positive', 'negative', 'uncertainty')
            - Values are floating-point scores, typically in range [-1.0, 1.0]
            
    Example:
        >>> sentiment_model = SentimentAnalyzer.load(model_path)
        >>> sentiment_scores = classify_sentiment("Revenue increased by 15%", sentiment_model)
        >>> print(f"Positive score: {sentiment_scores.get('positive', 0)}")
    """
    return sentiment_model.analyze(text)


def format_sentiment_result(result: Dict[str, float]) -> pd.DataFrame:
    """Format sentiment analysis results for dashboard visualization.
    
    Converts dictionary-based sentiment scores into a structured DataFrame
    that can be easily used for visualization in the dashboard. The function
    sorts sentiment dimensions by the absolute value of their scores to 
    highlight the strongest sentiment indicators first.
    
    Args:
        result (Dict[str, float]): Sentiment analysis results dictionary where
            keys are sentiment dimensions and values are corresponding scores.
        
    Returns:
        pd.DataFrame: Formatted DataFrame of sentiment results with columns:
            - Dimension: The sentiment dimension name (e.g., 'positive', 'negative')
            - Score: The sentiment score value
            The DataFrame is sorted by the absolute value of the scores
            in descending order.
            
    Example:
        >>> sentiment_scores = {'positive': 0.65, 'negative': -0.12, 'uncertainty': 0.24}
        >>> result_df = format_sentiment_result(sentiment_scores)
        >>> print(result_df.head())
           Dimension  Score
        0  positive    0.65
        1  uncertainty 0.24
        2  negative   -0.12
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
    """Extract or generate interactive visualization HTML from topic model.
    
    Attempts to get HTML-based interactive visualizations from the topic model
    if it supports this capability. If not available, generates a Plotly-based
    visualization showing topic relationships.
    
    Args:
        topic_model (TopicModeler): The initialized topic model object
        
    Returns:
        str: HTML string containing the interactive visualization
    """
    # First try to use the model's built-in visualization if available
    if hasattr(topic_model, 'get_visualization_html'):
        html = topic_model.get_visualization_html()
        if html:
            return html
    
    # If no built-in visualization, create one with Plotly
    try:
        import plotly.graph_objects as go
        import numpy as np
        
        # Get number of topics
        num_topics = topic_model.num_topics if hasattr(topic_model, 'num_topics') else 10
        
        # Function to get topic words
        def get_topic_words(topic_id, num_words=10):
            if hasattr(topic_model, 'get_topic_words'):
                return topic_model.get_topic_words(topic_id, num_words)
            elif hasattr(topic_model, 'topic_words') and topic_model.topic_words:
                if topic_id in topic_model.topic_words:
                    return topic_model.topic_words[topic_id][:num_words]
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
                topic_words_text = ", ".join(words[:8])
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

def get_visualization_html(self) -> str:
    """Generate an interactive HTML visualization of the topic model.
    
    Returns:
        str: HTML content for interactive visualization, or empty string if unavailable
    """
    try:
        if self.method == 'lda':
            # For LDA models, generate pyLDAvis visualization
            import pyLDAvis
            import pyLDAvis.sklearn
            
            if not hasattr(self, '_prepared_vis'):
                # Only prepare visualization once (it's computationally expensive)
                if hasattr(self, 'dtm') and hasattr(self, 'model'):
                    self._prepared_vis = pyLDAvis.sklearn.prepare(
                        self.model, self.dtm, self.vectorizer
                    )
                else:
                    return ""
                    
            # Convert to HTML
            return pyLDAvis.prepared_data_to_html(self._prepared_vis)
            
        # Add other model type visualizations here if needed
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        
    return ""

def get_feature_importance_plot(feature_extractor: FeatureExtractor) -> Optional[plt.Figure]:
    """Generate a feature importance visualization from the feature extractor.
    
    Creates a horizontal bar chart showing the most influential features in the
    feature extraction model. The chart displays the top 15 features ranked by
    their importance scores, which helps in interpreting which textual elements
    are most predictive in the financial analysis models.
    
    The function handles feature importance data in different formats (dictionary
    or list of tuples) and generates a standardized visualization.
    
    Args:
        feature_extractor (FeatureExtractor): The initialized feature extractor
            object that implements the get_feature_importance() method.
        
    Returns:
        Optional[plt.Figure]: Matplotlib figure object containing the feature
            importance bar chart, or None if feature importance information
            is not available or an error occurs.
            
    Example:
        >>> fig = get_feature_importance_plot(feature_extractor)
        >>> if fig:
        ...     st.pyplot(fig)
        ... else:
        ...     st.warning("Feature importance data not available")
            
    Notes:
        The function attempts to extract the top 15 most important features.
        It's designed to be resilient to different feature importance formats
        that might be returned by different feature extractor implementations.
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
    """Generate a word cloud visualization for a specific topic in the topic model.
    
    Creates a visually appealing word cloud where word size corresponds to its
    importance within the specified topic. This provides an intuitive visualization
    of the most relevant terms in each topic, making the abstract topic model
    more interpretable for dashboard users.
    
    The function handles different formats of topic word representations and
    generates a consistent visualization using the WordCloud package.
    
    Args:
        topic_model (TopicModeler): The initialized topic model object that
            implements the get_topic_words() method.
        topic_id (int): Numeric identifier of the topic to visualize.
        
    Returns:
        Optional[bytes]: Base64-encoded PNG image of the word cloud, ready to be
            displayed in the dashboard using st.image() or HTML components.
            Returns None if the topic model doesn't support word retrieval,
            the topic ID is invalid, or an error occurs.
            
    Example:
        >>> wordcloud_img = get_wordcloud_for_topic(topic_model, 5)
        >>> if wordcloud_img:
        ...     st.image(wordcloud_img)
        ... else:
        ...     st.warning("Could not generate word cloud for this topic")
            
    Notes:
        The function uses the MAX_WORD_CLOUD_WORDS configuration parameter to
        limit the number of words displayed in the cloud. The default color
        scheme used is 'viridis' which provides good contrast and readability.
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
    """Create a configuration for the earnings report prediction simulator component.
    
    Sets up an interactive simulator component for the dashboard that allows users
    to input financial text and get predictions from various NLP models. The simulator
    adapts its functionality based on which models are available, enabling sentiment
    analysis, topic modeling, and feature extraction on user-provided text.
    
    This function prepares the simulator configuration including model availability,
    initial sample text, and prediction functions that can be invoked by the dashboard.
    
    Args:
        models (Dict[str, Any]): Dictionary of loaded NLP models where keys are
            model types ('sentiment', 'topic', 'feature_extractor') and values
            are the corresponding model objects.
        sample_text (str): Sample earnings report text to use for initial display
            in the simulator input box.
        
    Returns:
        Dict[str, Any]: Dictionary with simulator configuration containing:
            - has_models: Boolean indicating if any models are available
            - initial_text: Sample text for initial display
            - available_models: Dictionary of model availability by type
            - predict_sentiment: Function for sentiment prediction (or dummy if unavailable)
            - predict_topics: Function for topic prediction (or dummy if unavailable)
            - extract_features: Function for feature extraction (or dummy if unavailable)
            
    Example:
        >>> simulator = create_prediction_simulator(models, "Q2 revenue increased by 15%")
        >>> if simulator['has_models']:
        ...     results = simulator['predict_sentiment'](user_text)
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
        simulator['extract_features'] = lambda text: models['feature_extractor'].extract_financial_metrics(
            pd.DataFrame({'text': [text]})
        ) if text else {}
    else:
        simulator['extract_features'] = lambda text: None
    
    return simulator


def create_topic_explorer(topic_model: Optional[TopicModeler]) -> Dict[str, Any]:
    """Create a configuration for the interactive topic explorer component.
    
    Sets up an interactive topic exploration interface for the dashboard that allows
    users to browse, visualize, and understand topics extracted from earnings reports.
    The explorer provides topic words, distributions, and visualizations to help
    interpret the latent topics discovered in the financial text corpus.
    
    This function prepares the explorer configuration including topic metadata,
    visualization capabilities, word retrieval functions, and word cloud generation.
    
    Args:
        topic_model (Optional[TopicModeler]): The initialized topic model object
            that provides topic extraction functionality, or None if no model
            is available.
        
    Returns:
        Dict[str, Any]: Dictionary with explorer configuration containing:
            - has_model: Boolean indicating if a topic model is available
            - num_topics: Number of topics in the model (if available)
            - topics_df: DataFrame of topic information
            - visualization_html: Interactive visualization HTML (if available)
            - get_topic_words: Function for retrieving top words for a topic
            - get_wordcloud: Function for generating topic word clouds
            
    Example:
        >>> explorer = create_topic_explorer(topic_model)
        >>> if explorer['has_model']:
        ...     topic_words = explorer['get_topic_words'](topic_id=3, n=20)
        ...     wordcloud = explorer['get_wordcloud'](topic_id=3)
            
    Notes:
        If the topic model is None or doesn't provide required functionality,
        the explorer will have limited features. The dashboard should check
        the 'has_model' flag before attempting to use topic exploration features.
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


