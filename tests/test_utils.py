"""Shared utility functions for test scripts.

This module centralizes common testing functions used across different test scripts
to avoid duplication and ensure consistent testing procedures. It provides functions for:

- Setting up standardized logging for test scripts
- Loading processed data with optional sampling for faster testing
- Testing the embedding processor functionality
- Testing the sentiment analyzer
- Testing topic modeling capabilities
- Testing feature extraction pipelines
- Testing model training workflows

Each function is designed to be reusable and configurable to support both
comprehensive testing and quick debugging with smaller data samples.
"""

import os
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any

# Import configuration values
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (NUM_TOPICS, MAX_FEATURES, RANDOM_STATE, TEST_SIZE, VAL_SIZE,
                   MODEL_DIR, OUTPUT_DIR)

# Ensure imports work from any directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
from src.data.data_versioner import DataVersioner
from src.nlp.embedding import EmbeddingProcessor
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.topic_modeling import TopicModeler
from src.nlp.feature_extraction import FeatureExtractor

logger = logging.getLogger('test_utils')

def setup_logging(log_file: str, logger_name: str):
    """Set up logging configuration for test scripts.
    
    Configures a logger with both file and console handlers to provide
    comprehensive logging during test execution.
    
    Args:
        log_file (str): Path to the log file where messages will be saved.
        logger_name (str): Name for the logger instance, typically matching the test module.
    
    Returns:
        logging.Logger: Configured logger ready for use.
    
    Examples:
        >>> logger = setup_logging('test_run.log', 'pipeline_test')
        >>> logger.info("Starting test execution")
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_processed_data(data_version: str = None, sample_size: int = None):
    """Load processed data from a specific version or the latest version.
    
    Retrieves preprocessed data splits from versioned datasets. If no version is specified,
    uses the latest available version. If no processed data is found, falls back to
    loading and minimally processing raw data.
    
    Args:
        data_version (str, optional): The specific data version ID to load. 
            If None, loads the latest version.
        sample_size (int, optional): If provided, limits the data to this number of samples
            for quick testing. Test and validation sets will be reduced proportionally.
    
    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - train: Training data split
            - val: Validation data split
            - test: Testing data split
    
    Examples:
        >>> # Load latest version with full data
        >>> train, val, test = load_processed_data()
        >>> print(f"Loaded {len(train)} training samples")
        >>> 
        >>> # Load specific version with reduced sample size for quick testing
        >>> train, val, test = load_processed_data('v1.0', sample_size=1000)
    
    Notes:
        If sample_size is provided, the validation and test sets are reduced to
        approximately 1/5th of the sample_size to maintain proportions.
    """
    versioner = DataVersioner()
    
    if data_version is None:
        data_version = versioner.get_latest_version()
        
        if data_version is None:
            # No processed data found, use direct data loading
            logger.warning("No processed data version found. Using raw data.")
            from src.data.pipeline import DataPipeline
            pipeline = DataPipeline()
            
            # Load raw data
            df = pipeline.load_data()
            train, val, test = pipeline.split_data(df)
            
            # Apply minimal processing
            for split in [train, val, test]:
                split['clean_sent'] = split['synopsis'].fillna('')
        else:
            # Load processed data from specified version
            paths = versioner.get_version_data_paths(data_version)
            train = pd.read_csv(paths['train'])
            val = pd.read_csv(paths['val'])
            test = pd.read_csv(paths['test'])
            logger.info(f"Loaded processed data version: {data_version}")
    else:
        # Load processed data from specified version
        paths = versioner.get_version_data_paths(data_version)
        train = pd.read_csv(paths['train'])
        val = pd.read_csv(paths['val'])
        test = pd.read_csv(paths['test'])
        logger.info(f"Loaded processed data version: {data_version}")
    
    # Sample data if requested
    if sample_size is not None:
        train = train.sample(min(sample_size, len(train)), random_state=42)
        val = val.sample(min(sample_size // 5, len(val)), random_state=42)
        test = test.sample(min(sample_size // 5, len(test)), random_state=42)
        logger.info(f"Using sampled data: {len(train)} train, {len(val)} val, {len(test)} test")
    
    return train, val, test

def test_embedding_processor(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           max_features: int = 5000, model_dir: str = None):
    """Test the embedding processor functionality with TF-IDF vectorization.
    
    Creates and evaluates an embedding processor using the TF-IDF method.
    Fits the processor on training data and transforms both training and validation
    data to generate document embeddings.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame containing text column.
        val_df (pd.DataFrame): Validation data DataFrame containing text column.
        max_features (int, optional): Maximum number of features (vocabulary size) for TF-IDF.
            Default is 5000.
        model_dir (str, optional): Directory to save the trained processor model.
            If None, the model won't be saved.
        
    Returns:
        tuple: A tuple containing three elements:
            - processor (EmbeddingProcessor): The fitted embedding processor
            - train_vectors (scipy.sparse.csr_matrix or numpy.ndarray): Vectorized training data
            - val_vectors (scipy.sparse.csr_matrix or numpy.ndarray): Vectorized validation data
    
    Examples:
        >>> processor, train_vecs, val_vecs = test_embedding_processor(
        ...     train_df, val_df, max_features=10000, model_dir='models/embeddings')
        >>> print(f"Embedding dimensionality: {train_vecs.shape[1]}")
    
    Notes:
        Automatically detects whether to use 'ea_text' or 'clean_sent' as the text column
        based on what's available in the DataFrame.
    """
    logger.info("Testing embedding processor...")
    
    # Check if clean_sent or ea_text column is available
    text_column = 'ea_text' if 'ea_text' in train_df.columns else 'clean_sent'
    logger.info(f"Using '{text_column}' column for text data")
    
    # Initialize embedding processor with correct parameters
    # Check the actual expected parameters for the class
    processor = EmbeddingProcessor(method='tfidf')
    
    # Set max_features if applicable (might be done through other methods)
    if hasattr(processor, 'set_params'):
        processor.set_params(max_features=max_features)
    
    # Fit and transform training data
    train_vectors = processor.fit_transform(train_df[text_column])
    logger.info(f"Train vectors shape: {train_vectors.shape}")
    
    # Transform validation data
    val_vectors = processor.transform(val_df[text_column])
    logger.info(f"Validation vectors shape: {val_vectors.shape}")
    
    # Save model if directory provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        processor.save(os.path.join(model_dir, 'tfidf_processor'))
        logger.info(f"Saved embedding processor to {model_dir}")
        
    return processor, train_vectors, val_vectors

def test_sentiment_analyzer(train_df: pd.DataFrame, lexicon: str = 'loughran_mcdonald', 
                          model_dir: str = None):
    """Test the sentiment analyzer functionality.
    
    Creates a sentiment analyzer using the specified lexicon and evaluates it on a
    sample of documents from the training data. The Loughran-McDonald lexicon is
    particularly suitable for financial text analysis.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame containing a 'clean_sent' column
            with preprocessed text.
        lexicon (str, optional): Lexicon to use for sentiment analysis.
            Default is 'loughran_mcdonald', which is specialized for financial texts.
        model_dir (str, optional): Directory to save the sentiment analyzer.
            If None, the analyzer won't be saved.
        
    Returns:
        tuple: A tuple containing two elements:
            - analyzer (SentimentAnalyzer): The configured sentiment analyzer
            - sentiment_df (pd.DataFrame): DataFrame containing sentiment scores for the sample
    
    Examples:
        >>> analyzer, results = test_sentiment_analyzer(train_df, model_dir='models/sentiment')
        >>> print(f"Average positive sentiment: {results['positive'].mean():.4f}")
        >>> print(f"Average negative sentiment: {results['negative'].mean():.4f}")
    
    Notes:
        - Analysis is performed on a sample of up to 100 documents to save time
        - Uses the 'clean_sent' column for text data
    """
    logger.info("Testing sentiment analyzer...")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(lexicon=lexicon)
    
    # Analyze a sample of documents
    sample_size = min(100, len(train_df))
    sample = train_df.sample(sample_size, random_state=42)
    
    # Calculate sentiment scores
    sentiments = []
    for text in sample['clean_sent']:
        sentiment = analyzer.analyze(text)
        sentiments.append(sentiment)
    
    sentiment_df = pd.DataFrame(sentiments)
    logger.info(f"Sentiment analysis results:\n{sentiment_df.describe()}")
    
    # Save model if directory provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        analyzer.save(os.path.join(model_dir, 'sentiment_analyzer'))
        logger.info(f"Saved sentiment analyzer to {model_dir}")
    
    return analyzer, sentiment_df

def test_topic_modeling(train_df: pd.DataFrame, num_topics: int = 40, 
                      method: str = 'lda', model_dir: str = None):
    """Test topic modeling functionality.
    
    Creates a topic model using the specified method and fits it on the training data.
    Generates the document-term matrix, fits the topic model, and extracts the top words
    for each discovered topic.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame containing a 'clean_sent' column
            with preprocessed text.
        num_topics (int, optional): Number of topics to extract. Default is 40.
            Higher values provide more granular topics but may lead to redundancy.
        method (str, optional): Topic modeling method to use. Default is 'lda'.
            Options include:
            - 'lda': Latent Dirichlet Allocation (probabilistic)
            - 'nmf': Non-negative Matrix Factorization (algebraic)
            - 'bertopic': BERTopic (transformer-based)
        model_dir (str, optional): Directory to save the topic model and related artifacts.
            If None, nothing will be saved.
        
    Returns:
        tuple: A tuple containing four elements:
            - modeler (TopicModeler): The fitted topic modeler
            - dtm (scipy.sparse.csr_matrix): Document-term matrix
            - topic_distributions (numpy.ndarray): Topic distributions for each document
            - vocab (list): Vocabulary list
    
    Examples:
        >>> modeler, dtm, distributions, vocab = test_topic_modeling(
        ...     train_df, num_topics=20, method='lda', model_dir='models/topics')
        >>> print(f"Number of documents: {distributions.shape[0]}")
        >>> print(f"Number of topics: {distributions.shape[1]}")
    
    Notes:
        - LDA works best with a sufficient number of topics (typically 10-100)
        - Topic model training can be computationally intensive for large datasets
        - Only the top 3 topics will be logged for quick inspection
    """
    logger.info(f"Testing topic modeling with {method}...")
    
    # Initialize topic modeler
    modeler = TopicModeler(method=method, num_topics=num_topics)
    
    # Create document-term matrix
    dtm, vectorizer, vocab = modeler.create_document_term_matrix(
        train_df['clean_sent'].tolist(),
        save_path=os.path.join(model_dir, 'vectorizer.pkl') if model_dir else None
    )
    
    logger.info(f"Document-term matrix shape: {dtm.shape}")
    
    # Fit topic model
    model, topic_distributions = modeler.fit(dtm, save_model=bool(model_dir), model_dir=model_dir)
    
    # Get top words for each topic
    top_words = modeler.get_top_words(
        n_words=10, 
        save_results=bool(model_dir), 
        output_dir=model_dir
    )
    
    # Log sample of top words
    for topic_idx in list(top_words.keys())[:3]:
        logger.info(f"Topic {topic_idx}: {', '.join(top_words[topic_idx])}")
    
    return modeler, dtm, topic_distributions, vocab

def test_feature_extraction(train_df: pd.DataFrame, topic_modeler: TopicModeler = None, 
                          sentiment_analyzer: SentimentAnalyzer = None,
                          embedding_processor: EmbeddingProcessor = None,
                          model_dir: str = None):
    """Test the feature extraction pipeline.
    
    Creates and evaluates a feature extractor that combines multiple NLP components
    (topic modeling, sentiment analysis, embeddings, and text metrics) into a unified
    feature matrix suitable for machine learning.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame with 'clean_sent' column.
        topic_modeler (TopicModeler, optional): Fitted topic modeler. If None,
            topic-based features won't be included.
        sentiment_analyzer (SentimentAnalyzer, optional): Configured sentiment analyzer.
            If None, sentiment features won't be included.
        embedding_processor (EmbeddingProcessor, optional): Fitted embedding processor.
            If None, embedding-based features won't be included.
        model_dir (str, optional): Directory to save the feature extractor.
            If None, the extractor won't be saved.
        
    Returns:
        tuple: A tuple containing three elements:
            - extractor (FeatureExtractor): The configured feature extractor
            - features (numpy.ndarray): Combined feature matrix for training data
            - feature_names (list): Names of all features in the matrix
    
    Examples:
        >>> extractor, features, feature_names = test_feature_extraction(
        ...     train_df, topic_modeler, sentiment_analyzer, embedding_processor,
        ...     model_dir='models/features')
        >>> print(f"Feature matrix shape: {features.shape}")
        >>> print(f"Feature groups: {extractor.get_feature_groups().keys()}")
    
    Notes:
        - Includes text metrics features by default
        - Other feature types are only included if the corresponding component is provided
        - Reports the number and types of features extracted
    """
    logger.info("Testing feature extraction...")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        use_topics=topic_modeler is not None,
        use_sentiment=sentiment_analyzer is not None,
        use_embeddings=embedding_processor is not None,
        use_metrics=True
    )
    
    # Set components
    if topic_modeler is not None:
        extractor.set_topic_model(topic_modeler.model)
        extractor.vectorizer = topic_modeler.vectorizer
    
    if sentiment_analyzer is not None:
        extractor.set_sentiment_analyzer(sentiment_analyzer)
    
    if embedding_processor is not None:
        extractor.set_embedding_model(embedding_processor)
    
    # Extract features
    features, feature_names = extractor.extract_features(train_df, text_column='clean_sent')
    
    logger.info(f"Extracted {features.shape[1]} features")
    logger.info(f"Feature types: {extractor.get_feature_groups().keys()}")
    
    # Save model if directory provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        extractor.save(os.path.join(model_dir, 'feature_extractor.pkl'))
        logger.info(f"Saved feature extractor to {model_dir}")
    
    return extractor, features, feature_names

def test_model_training(features: np.ndarray, targets: np.ndarray,
                      model_type: str = 'classifier', model_dir: str = None):
    """Test model training functionality.
    
    Trains and evaluates machine learning models using the specified features and targets.
    Supports both regression (Lasso) and classification models, with options to train
    either type or both.
    
    Args:
        features (np.ndarray): Feature matrix derived from NLP processing.
        targets (np.ndarray): Target values for prediction:
            - For regression: Continuous return values
            - For classification: Binary labels (typically 0/1)
        model_type (str, optional): Type of model to train. Default is 'classifier'.
            Options:
            - 'lasso': Train Lasso regression model for sparse feature selection
            - 'classifier': Train various classifier models (LogisticRegression, SVM, etc.)
            - 'all': Train both regression and classification models
        model_dir (str, optional): Directory to save trained models.
            If None, models won't be saved.
        
    Returns:
        dict: Dictionary containing trained models and results with structure:
            {
                'lasso': {  # (only if model_type is 'lasso' or 'all')
                    'model': trained_lasso_model,
                    'results': lasso_training_metrics,
                    'nonzero_topics': nonzero_feature_indices
                },
                'classifier': {  # (only if model_type is 'classifier' or 'all')
                    'model': best_classifier_model,
                    'results': classifier_training_metrics
                }
            }
    
    Examples:
        >>> # Train a classifier model
        >>> results = test_model_training(features, binary_targets, model_type='classifier')
        >>> classifier = results['classifier']['model']
        >>> metrics = results['classifier']['results']
        >>> 
        >>> # Train both model types
        >>> results = test_model_training(features, continuous_targets, model_type='all')
    
    Notes:
        - Uses the model_trainer module for the actual training process
        - For classification, trains multiple models and returns the best performer
        - For Lasso, performs hyperparameter tuning to find the optimal alpha
    """
    logger.info(f"Testing model training with {model_type}...")
    
    # Import model trainer
    from src.models.model_trainer import train_model
    
    # Train model
    results = train_model(features, targets, model_type=model_type)
    
    # Log results
    if 'lasso' in results:
        lasso_results = results['lasso']['results']
        logger.info(f"Lasso regression results: {lasso_results}")
        
    if 'classifier' in results:
        classifier_results = results['classifier']['results']
        logger.info(f"Classifier results: {classifier_results}")
    
    # Save models if directory provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        
        import pickle
        with open(os.path.join(model_dir, 'model_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved model results to {model_dir}")
    
    return results
