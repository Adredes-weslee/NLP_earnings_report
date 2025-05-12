"""Consolidated utility functions for the NLP earnings report project.

This module provides common utility functions used throughout the project,
including:

- File handling utilities (save/load models, create directories)
- Visualization helpers (plot generation, wordclouds)
- Data processing tools (text normalization, financial calculations)
- Evaluation metrics for NLP and ML models
- Logging configuration

These utilities support the core functionality of data processing,
NLP analysis, and results presentation.
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
    """Set up logging configuration for a module.
    
    This function configures a logger with consistent formatting and optional
    file output. It creates both console and file handlers if a log file is specified.
    
    Args:
        name (str): Logger name to identify the module in log messages.
        log_file (str, optional): Path to the log file. If None, logging is 
            only sent to the console. Defaults to None.
        level (int, optional): Logging level threshold. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: Configured logger instance ready for use.
        
    Example:
        >>> logger = setup_logging('data_processor', 'logs/processing.log')
        >>> logger.info('Data processing started')
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
    """Load serialized object from a pickle file.
    
    Deserialize a Python object from a binary file using the pickle protocol.
    
    Args:
        file_path (str): Path to the pickle file to be loaded.
        
    Returns:
        Any: The deserialized Python object.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
        
    Example:
        >>> model = load_pickle('models/sentiment_classifier.pkl')
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj: Any, file_path: str) -> None:
    """Serialize a Python object to a pickle file.
    
    Creates necessary directories if they don't exist and serializes
    the given object to the specified file path using the pickle protocol.
    
    Args:
        obj (Any): The Python object to serialize.
        file_path (str): Destination path where the pickle file will be saved.
    
    Example:
        >>> save_pickle(trained_model, 'models/sentiment_classifier.pkl')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {file_path}")

def load_joblib(file_path: str) -> Any:
    """Load a serialized object from a joblib file.
    
    Deserialize a Python object using joblib, which is optimized for 
    efficiently serializing large numpy arrays and scikit-learn models.
    
    Args:
        file_path (str): Path to the joblib file to be loaded.
        
    Returns:
        Any: The deserialized Python object.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        
    Example:
        >>> vectorizer = load_joblib('models/tfidf_vectorizer.joblib')
    """
    return joblib.load(file_path)

def save_joblib(obj: Any, file_path: str) -> None:
    """Serialize a Python object to a joblib file.
    
    Creates necessary directories if they don't exist and serializes
    the given object to the specified file path using joblib, which is 
    optimized for efficiently serializing large numpy arrays and scikit-learn models.
    
    Args:
        obj (Any): The Python object to serialize.
        file_path (str): Destination path where the joblib file will be saved.
    
    Example:
        >>> save_joblib(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
    logger.info(f"Object saved to {file_path}")

def load_json(file_path: str) -> Dict:
    """Load data from a JSON file.
    
    Parse JSON file and return the resulting Python dictionary.
    
    Args:
        file_path (str): Path to the JSON file to be loaded.
        
    Returns:
        Dict: The parsed JSON data as a Python dictionary.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        
    Example:
        >>> config = load_json('config/parameters.json')
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(obj: Dict, file_path: str, indent: int = 2) -> None:
    """Serialize a dictionary to a JSON file.
    
    Creates necessary directories if they don't exist and writes
    the given dictionary to the specified file path in JSON format.
    
    Args:
        obj (Dict): The dictionary to serialize to JSON.
        file_path (str): Destination path where the JSON file will be saved.
        indent (int, optional): Number of spaces for indentation in the output file.
            Makes the file more human-readable. Defaults to 2.
    
    Example:
        >>> save_json({'threshold': 0.75, 'max_features': 1000}, 'config/parameters.json')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=indent)
    logger.info(f"JSON object saved to {file_path}")

def plot_feature_importance(feature_importance: Dict[str, float], n: int = 20, 
                          figsize: Tuple[int, int] = (12, 10), 
                          color_scheme: str = 'pos_neg') -> plt.Figure:
    """Create a horizontal bar chart of feature importances.
    
    Visualizes the most important features based on their importance values,
    with options to display different colors for positive and negative values.
    
    Args:
        feature_importance (Dict[str, float]): Dictionary mapping feature names to 
            their importance values.
        n (int, optional): Number of top features to display. Defaults to 20.
        figsize (Tuple[int, int], optional): Figure size as (width, height). 
            Defaults to (12, 10).
        color_scheme (str, optional): Color scheme to use, either 'pos_neg' to color
            positive and negative values differently, or 'single' for a uniform color.
            Defaults to 'pos_neg'.
            
    Returns:
        plt.Figure: Matplotlib figure object containing the feature importance plot.
        
    Example:
        >>> importances = {'age': 0.25, 'income': 0.15, 'education': -0.10}
        >>> fig = plot_feature_importance(importances, n=3)
        >>> fig.savefig('feature_importance.png')
        
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
    """Generate a word cloud visualization from word weights.
    
    Creates a visual representation where word size reflects their weight,
    useful for visualizing term importance in documents.
    
    Args:
        word_weights (Dict[str, float]): Dictionary mapping words to their weights.
            Words with higher weights appear larger in the visualization.
        title (str, optional): Title to display above the word cloud. 
            Defaults to 'Word Cloud'.
        figsize (Tuple[int, int], optional): Figure size as (width, height).
            Defaults to (10, 6).
        background_color (str, optional): Background color of the word cloud.
            Defaults to 'white'.
            
    Returns:
        plt.Figure: Matplotlib figure object containing the word cloud visualization.
        
    Example:
        >>> word_freq = {'finance': 10, 'revenue': 8, 'growth': 6, 'earnings': 5}
        >>> fig = plot_wordcloud(word_freq, title='Financial Terms')
        >>> fig.savefig('financial_wordcloud.png')
        
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
    """Format topic model results for human-readable display.
    
    Converts a dictionary of topics and their associated words into a 
    formatted string for presentation in reports or dashboards.
    
    Args:
        topic_words (Dict[int, List[str]]): Dictionary mapping topic indices to 
            lists of the most representative words for each topic.
        n_words (int, optional): Number of words to include per topic. 
            Defaults to 5.
        
    Returns:
        str: Newline-separated string of formatted topics, where each line contains
            a topic number followed by its most representative words.
            
    Example:
        >>> topics = {0: ['finance', 'revenue', 'growth'], 1: ['product', 'launch', 'market']}
        >>> print(format_topics(topics, n_words=2))
        Topic 0: finance, revenue
        Topic 1: product, launch
    """
    formatted = []
    for topic_idx, words in sorted(topic_words.items()):
        word_list = words[:n_words]
        formatted.append(f"Topic {topic_idx}: {', '.join(word_list)}")
    
    return "\n".join(formatted)

def plot_topic_coherence(topic_counts: List[int], coherence_scores: List[float], 
                       optimal_topic_count: int = None,
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot topic model coherence scores across different numbers of topics.
    
    Creates a line plot showing how coherence scores change with different
    numbers of topics, helping to identify the optimal topic count.
    
    Args:
        topic_counts (List[int]): List of the number of topics evaluated.
        coherence_scores (List[float]): Corresponding coherence scores for 
            each number of topics.
        optimal_topic_count (int, optional): The optimal number of topics to 
            highlight with a vertical line. If None, no optimal value is highlighted.
            Defaults to None.
        figsize (Tuple[int, int], optional): Figure size as (width, height).
            Defaults to (10, 6).
        
    Returns:
        plt.Figure: Matplotlib figure object containing the coherence plot.
        
    Example:
        >>> topics = [5, 10, 15, 20]
        >>> scores = [0.42, 0.45, 0.38, 0.36]
        >>> fig = plot_topic_coherence(topics, scores, optimal_topic_count=10)
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
    """Plot histograms of sentiment scores from a DataFrame.
    
    Creates a series of histograms showing the distribution of sentiment scores
    across different sentiment dimensions (e.g., polarity, subjectivity).
    
    Args:
        sentiment_data (pd.DataFrame): DataFrame containing columns of sentiment scores.
            Each column is expected to represent a different sentiment metric.
        figsize (Tuple[int, int], optional): Figure size as (width, height).
            Defaults to (12, 6).
        
    Returns:
        plt.Figure: Matplotlib figure object containing the sentiment histograms.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'polarity': [0.2, 0.5, -0.1, 0.3],
        ...     'subjectivity': [0.4, 0.8, 0.3, 0.6]
        ... })
        >>> fig = plot_sentiment_distribution(df)
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
    """Calculate aggregated importance scores for groups of features.
    
    Combines individual feature importance values into group-level importance
    scores by summing the absolute importance values for all features in each group.
    
    Args:
        feature_importance (Dict[str, float]): Dictionary mapping individual feature 
            names to their importance values.
        feature_groups (Dict[str, List[str]]): Dictionary mapping group names to 
            lists of feature names that belong to each group.
        
    Returns:
        Dict[str, float]: Dictionary mapping group names to their aggregate 
            importance scores.
            
    Example:
        >>> feature_imp = {'age': 0.2, 'income': 0.3, 'education': 0.1}
        >>> groups = {'demographics': ['age', 'education'], 'financial': ['income']}
        >>> get_feature_group_importances(feature_imp, groups)
        {'demographics': 0.3, 'financial': 0.3}
    """
    group_importances = {}
    
    for group_name, features in feature_groups.items():
        group_importance = sum(abs(feature_importance.get(feature, 0.0)) for feature in features)
        group_importances[group_name] = group_importance
    
    return group_importances

def plot_feature_group_importance(group_importances: Dict[str, float],
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Create a bar chart of feature group importance values.
    
    Visualizes the importance of different feature groups in a model,
    helping to identify which types of features contribute most to predictions.
    
    Args:
        group_importances (Dict[str, float]): Dictionary mapping feature group names 
            to their aggregated importance values.
        figsize (Tuple[int, int], optional): Figure size as (width, height).
            Defaults to (10, 6).
        
    Returns:
        plt.Figure: Matplotlib figure object containing the feature group importance plot.
        
    Example:
        >>> groups = {'demographics': 0.3, 'financial': 0.5, 'behavioral': 0.2}
        >>> fig = plot_feature_group_importance(groups)
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
    """Convert a matplotlib figure to a base64 encoded string.
    
    Transforms a matplotlib figure into a base64 string representation that
    can be embedded directly in HTML documents, such as in dashboard visualizations.
    
    Args:
        fig (plt.Figure): The matplotlib figure object to convert.
        
    Returns:
        str: Base64 encoded string representation of the figure.
        
    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> img_str = fig_to_base64(fig)
        >>> html_img = f'<img src="data:image/png;base64,{img_str}"/>'
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def classify_sentiment(sentiment_scores: Dict[str, float]) -> str:
    """Classify text sentiment as positive, negative, or neutral.
    
    Analyzes sentiment scores to determine the overall sentiment classification
    based on polarity thresholds.
    
    Args:
        sentiment_scores (Dict[str, float]): Dictionary containing sentiment 
            metrics like 'positive', 'negative', and 'polarity' scores.
        
    Returns:
        str: Sentiment classification as either "Positive", "Negative", or "Neutral".
        
    Example:
        >>> scores = {'positive': 0.25, 'negative': 0.05, 'polarity': 0.2}
        >>> classify_sentiment(scores)
        'Positive'
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
    """Generate a word cloud visualization for documents of a specific class.
    
    Creates a visual representation of the most common words in documents
    belonging to a specified class, with word size reflecting frequency.
    
    Args:
        texts (List[str]): List of text documents to analyze.
        labels (List[int]): List of class labels corresponding to each text document.
        class_label (int): The specific class label to filter for.
        background_color (str, optional): Background color for the word cloud.
            Defaults to 'white'.
        
    Returns:
        plt.Figure: Matplotlib figure containing the word cloud visualization,
            or a placeholder figure if no documents match the class label.
            
    Example:
        >>> docs = ["growth in revenue", "profit declined", "stable outlook"]
        >>> class_ids = [1, 0, 1]
        >>> fig = generate_wordcloud_for_class(docs, class_ids, class_label=1)
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
