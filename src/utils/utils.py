"""
Consolidated utility functions for the NLP earnings report project.
"""

import os
import pickle
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import logging
import sys
import json
import base64
from io import BytesIO
from typing import Dict, List, Any, Tuple, Optional, Union
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('utils')

def setup_logging(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration for a module.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
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

def load_pickle(file_path: str) -> Any:
    """
    Load a pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Any: Loaded object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save pickle file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {file_path}")

def load_joblib(file_path: str) -> Any:
    """
    Load a joblib file.
    
    Args:
        file_path: Path to joblib file
        
    Returns:
        Any: Loaded object
    """
    return joblib.load(file_path)

def save_joblib(obj: Any, file_path: str) -> None:
    """
    Save an object to a joblib file.
    
    Args:
        obj: Object to save
        file_path: Path to save joblib file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
    logger.info(f"Object saved to {file_path}")

def load_json(file_path: str) -> Dict:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        dict: Loaded JSON object
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(obj: Dict, file_path: str, indent: int = 2) -> None:
    """
    Save an object to a JSON file.
    
    Args:
        obj: Object to save
        file_path: Path to save JSON file
        indent: Indentation level
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=indent)
    logger.info(f"JSON object saved to {file_path}")

def plot_feature_importance(feature_importance: Dict[str, float], n: int = 20, 
                          figsize: Tuple[int, int] = (12, 10), 
                          color_scheme: str = 'pos_neg') -> plt.Figure:
    """
    Plot feature importances.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance values
        n: Number of top features to show
        figsize: Figure size (width, height)
        color_scheme: Color scheme to use ('pos_neg' or 'single')
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create DataFrame from importance dictionary
    importance_df = pd.DataFrame({
        "feature": list(feature_importance.keys()),
        "importance": list(feature_importance.values())
    })
    
    # Sort by absolute importance to handle negative values
    importance_df["abs_importance"] = importance_df["importance"].abs()
    importance_df = importance_df.sort_values("abs_importance", ascending=False).head(n)
    importance_df = importance_df.drop("abs_importance", axis=1)
    
    # Sort for display (largest to smallest, regardless of sign)
    importance_df = importance_df.sort_values("importance")
    
    plt.figure(figsize=figsize)
    
    # Color bars based on positive/negative values
    if color_scheme == 'pos_neg':
        colors = ['red' if x < 0 else 'blue' for x in importance_df['importance']]
    else:
        colors = 'steelblue'
    
    plt.barh(y=importance_df['feature'], width=importance_df['importance'], color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {n} Feature Importances')
    plt.tight_layout()
    
    return plt.gcf()

def plot_wordcloud(word_weights: Dict[str, float], title: str = 'Word Cloud',
                 figsize: Tuple[int, int] = (10, 6), 
                 background_color: str = 'white') -> plt.Figure:
    """
    Generate a word cloud visualization from word weights.
    
    Args:
        word_weights: Dictionary mapping words to their weights
        title: Title for the plot
        figsize: Figure size (width, height)
        background_color: Background color of the word cloud
        
    Returns:
        matplotlib.figure.Figure: Word cloud figure
    """
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color=background_color,
        colormap='viridis',
        prefer_horizontal=1.0
    ).generate_from_frequencies(word_weights)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def format_topics(topic_words: Dict[int, List[str]], n_words: int = 5) -> str:
    """
    Format topics for display.
    
    Args:
        topic_words: Dictionary mapping topic indices to lists of top words
        n_words: Number of words to include per topic
        
    Returns:
        str: Formatted topic string
    """
    formatted = []
    for topic_idx, words in sorted(topic_words.items()):
        word_list = words[:n_words]
        formatted.append(f"Topic {topic_idx}: {', '.join(word_list)}")
    
    return "\n".join(formatted)

def plot_topic_coherence(topic_counts: List[int], coherence_scores: List[float], 
                       optimal_topic_count: int = None,
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot topic coherence scores.
    
    Args:
        topic_counts: List of topic counts evaluated
        coherence_scores: List of coherence scores for each topic count
        optimal_topic_count: Optimal topic count to highlight
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    plt.plot(topic_counts, coherence_scores, marker='o')
    
    if optimal_topic_count:
        plt.axvline(x=optimal_topic_count, color='r', linestyle='--', 
                    label=f'Optimal: {optimal_topic_count} topics')
    
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Topic Coherence vs. Number of Topics')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def plot_sentiment_distribution(sentiment_data: pd.DataFrame, 
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot sentiment distribution.
    
    Args:
        sentiment_data: DataFrame with sentiment scores
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    for i, column in enumerate(sentiment_data.columns):
        plt.subplot(1, len(sentiment_data.columns), i+1)
        sentiment_data[column].hist(bins=20)
        plt.title(column)
    
    plt.tight_layout()
    return plt.gcf()

def get_feature_group_importances(feature_importance: Dict[str, float], 
                                feature_groups: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate importance of feature groups.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance values
        feature_groups: Dictionary mapping group names to lists of feature names
        
    Returns:
        Dict[str, float]: Dictionary mapping group names to total importance
    """
    group_importances = {}
    
    for group_name, features in feature_groups.items():
        group_importance = sum(abs(feature_importance.get(feature, 0.0)) for feature in features)
        group_importances[group_name] = group_importance
    
    return group_importances

def plot_feature_group_importance(group_importances: Dict[str, float],
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot feature group importances.
    
    Args:
        group_importances: Dictionary mapping group names to importance values
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create DataFrame and sort by importance
    df = pd.DataFrame({
        'group': list(group_importances.keys()),
        'importance': list(group_importances.values())
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=figsize)
    plt.bar(df['group'], df['importance'], color='steelblue')
    plt.xlabel('Feature Group')
    plt.ylabel('Total Absolute Importance')
    plt.title('Feature Group Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to base64 string for embedding in HTML.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        str: Base64-encoded string
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def classify_sentiment(sentiment_scores: Dict[str, float]) -> str:
    """
    Classify text sentiment based on sentiment scores.
    
    Args:
        sentiment_scores: Dictionary of sentiment scores
        
    Returns:
        str: Sentiment classification
    """
    # Get key sentiment metrics
    positive = sentiment_scores.get('positive', 0)
    negative = sentiment_scores.get('negative', 0)
    polarity = sentiment_scores.get('polarity', 0)
    
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def generate_wordcloud_for_class(texts: List[str], labels: List[int], 
                              class_label: int, background_color: str = 'white') -> plt.Figure:
    """
    Generate a word cloud for documents of a specific class.
    
    Args:
        texts: List of text documents
        labels: List of class labels
        class_label: The class label to filter by
        background_color: Background color for the wordcloud
        
    Returns:
        matplotlib.figure.Figure: Word cloud figure
    """
    # Filter texts by class label
    class_texts = [text for text, label in zip(texts, labels) if label == class_label]
    
    if not class_texts:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"No texts found for class {class_label}", 
                ha='center', va='center')
        ax.axis('off')
        return fig
    
    # Combine all texts
    text = ' '.join(class_texts)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        colormap='viridis',
        max_words=100,
        prefer_horizontal=1.0
    ).generate(text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Word Cloud for Class {class_label}')
    ax.axis('off')
    
    return fig
