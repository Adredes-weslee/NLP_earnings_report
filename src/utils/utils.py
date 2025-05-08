"""
Utility functions for the NLP earnings report project.
"""

import os
import pickle
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('utils')

def setup_logging(name, log_file=None, level=logging.INFO):
    """
    Set up logging configuration for a module.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level (int, optional): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler if not already present
    has_console_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            has_console_handler = True
            break
    
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def plot_wordcloud(topic_words, title=None, max_words=50, figsize=(10, 6), save_path=None, colormap='viridis'):
    """
    Generate and plot a word cloud from topic words.
    
    Args:
        topic_words (dict): Dictionary mapping words to weights
        title (str, optional): Title for the word cloud
        max_words (int, optional): Maximum number of words to include
        figsize (tuple, optional): Figure size
        save_path (str, optional): Path to save the word cloud image
        colormap (str, optional): Colormap to use
        
    Returns:
        matplotlib.figure.Figure: Figure containing the word cloud
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.error("WordCloud package not found. Install with: pip install wordcloud")
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=max_words,
        colormap=colormap,
        contour_width=1,
        contour_color='steelblue',
        prefer_horizontal=0.9,
        random_state=42
    ).generate_from_frequencies(topic_words)
    
    # Create figure and plot word cloud
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    if title:
        plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Word cloud saved to {save_path}")
    
    return fig

def plot_feature_importance(feature_names, feature_importances, top_n=20, figsize=(12, 8), title='Feature Importance', save_path=None):
    """
    Plot feature importance from a model.
    
    Args:
        feature_names (list): Names of features
        feature_importances (array): Importance scores for each feature
        top_n (int, optional): Number of top features to show
        figsize (tuple, optional): Figure size
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object containing the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Create DataFrame for easier handling
    if len(feature_names) != len(feature_importances):
        raise ValueError(f"Length mismatch: feature_names ({len(feature_names)}) and feature_importances ({len(feature_importances)})")
    
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Sort by importance and select top N
    feature_df = feature_df.sort_values('importance', ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(feature_df))
    ax.barh(y_pos, feature_df['importance'], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_df['feature'])
    ax.invert_yaxis()  # Labels read top-to-bottom
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig

def create_figure_base64(fig, close_fig=True):
    """
    Convert a matplotlib figure to base64 encoded string.
    Useful for embedding figures in HTML or Streamlit.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to convert
        close_fig (bool): Whether to close the figure after conversion
        
    Returns:
        str: Base64 encoded string of the figure
    """
    import io
    import base64
    
    # Create a bytes buffer for the image to save to
    buf = io.BytesIO()
    
    # Save the figure to the buffer
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    
    # Close the figure if requested
    if close_fig:
        plt.close(fig)
    
    # Encode the bytes buffer to base64 string
    img_str = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return img_str

def create_dirs():
    """Create necessary directories for the project"""
    # Define required directories
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root_dir, 'models')
    output_dir = os.path.join(root_dir, 'results')
    
    # Create main directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Create subdirectories for models
    os.makedirs(os.path.join(model_dir, 'embeddings', 'tfidf_5000'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'sentiment', 'loughran_mcdonald'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'topics', 'lda_model'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'features', 'combined_features'), exist_ok=True)
    
    return model_dir, output_dir

def create_placeholder_models():
    """Create minimal placeholder models for the Streamlit dashboard"""
    model_dir, output_dir = create_dirs()
    
    # Create placeholder TF-IDF model
    config_path = os.path.join(model_dir, 'embeddings', 'tfidf_5000', 'config.joblib')
    vectorizer_path = os.path.join(model_dir, 'embeddings', 'tfidf_5000', 'vectorizer.joblib')
    
    if not os.path.exists(vectorizer_path):
        logger.info("Creating placeholder TF-IDF vectorizer")
        vectorizer = TfidfVectorizer(max_features=100)
        vectorizer.fit(['placeholder earnings report text for model initialization'])
        
        config = {
            'method': 'tfidf',
            'max_features': 100,
            'vocab_size': len(vectorizer.vocabulary_),
        }
        
        joblib.dump(config, config_path)
        joblib.dump(vectorizer, vectorizer_path)
        logger.info("Created placeholder embedding model")

    # Create placeholder sentiment analyzer
    sentiment_path = os.path.join(model_dir, 'sentiment', 'loughran_mcdonald', 'sentiment_config.joblib')
    if not os.path.exists(sentiment_path):
        logger.info("Creating placeholder sentiment analyzer")
        sentiment_config = {
            'method': 'loughran_mcdonald',
            'positive_words': ['increase', 'growth', 'profit'],
            'negative_words': ['decrease', 'loss', 'decline'],
            'uncertainty_words': ['may', 'approximately', 'risk'],
            'litigious_words': ['lawsuit', 'litigation', 'claim']
        }
        
        joblib.dump(sentiment_config, sentiment_path)
        logger.info("Created placeholder sentiment model")

    # Create placeholder topic model
    topic_model_path = os.path.join(model_dir, 'topics', 'lda_model', 'lda_model.pkl')
    if not os.path.exists(topic_model_path):
        logger.info("Creating placeholder topic model")
        # Create a simple structure that mimics what the real model would return
        topic_model = {
            'num_topics': 10,
            'topics': {i: [('word'+str(j), 0.1) for j in range(10)] for i in range(10)},
            'coherence': 0.5,
            'perplexity': -8.5,
            'method': 'lda',
            'model_params': {'num_topics': 10, 'random_state': 42}
        }
        
        with open(topic_model_path, 'wb') as f:
            pickle.dump(topic_model, f)
        logger.info("Created placeholder topic model")

    # Create placeholder feature extractor
    feature_path = os.path.join(model_dir, 'features', 'combined_features', 'feature_extractor.pkl')
    if not os.path.exists(feature_path):
        logger.info("Creating placeholder feature extractor")
        feature_names = [f'topic_{i}' for i in range(10)] + \
                        ['positive', 'negative', 'uncertainty', 'litigious'] + \
                        [f'word_{i}' for i in range(20)]
        
        feature_extractor = {
            'feature_groups': {
                'topic': [f'topic_{i}' for i in range(10)],
                'sentiment': ['positive', 'negative', 'uncertainty', 'litigious'],
                'embedding': [f'word_{i}' for i in range(20)]
            },
            'feature_names': feature_names,
            'feature_importances': np.random.random(len(feature_names)),
            'components': ['embedding_processor', 'sentiment_analyzer', 'topic_modeler']
        }
        
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_extractor, f)
        logger.info("Created placeholder feature extractor")

    # Create a sample figure for feature importance
    fig_path = os.path.join(output_dir, 'figures', 'feature_importances.png')
    if not os.path.exists(fig_path):
        logger.info("Creating placeholder feature importance figure")
        plt.figure(figsize=(10, 6))
        
        feature_sample = feature_names[:15]
        importance_sample = np.random.random(15)
        importance_sample = sorted(importance_sample, reverse=True)
        
        plt.barh(range(len(feature_sample)), importance_sample, align='center')
        plt.yticks(range(len(feature_sample)), feature_sample)
        plt.xlabel('Importance')
        plt.title('Feature Importance (Placeholder)')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        logger.info("Created placeholder feature importance figure")

    logger.info("All placeholder models created successfully!")
    return True