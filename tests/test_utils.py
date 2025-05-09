"""
Shared utility functions for test scripts.
This module centralizes common functions used across different test scripts to avoid duplication.
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
    """
    Set up logging configuration for test scripts
    
    Args:
        log_file: Path to log file
        logger_name: Name for the logger
    
    Returns:
        logging.Logger: Configured logger
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
    """
    Load processed data from a specific version or the latest version
    
    Args:
        data_version: The specific data version to load
        sample_size: If provided, limits the data to this number of samples for quick testing
    
    Returns:
        tuple: (train, val, test) DataFrames
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
    """
    Test the embedding processor with TF-IDF
    
    Args:
        train_df: Training data
        val_df: Validation data
        max_features: Maximum number of features for TF-IDF
        model_dir: Directory to save the model
        
    Returns:
        tuple: (embedding processor, train_vectors, val_vectors)
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
    """
    Test the sentiment analyzer
    
    Args:
        train_df: Training data
        lexicon: Lexicon to use for sentiment analysis
        model_dir: Directory to save the model
        
    Returns:
        tuple: (sentiment analyzer, sentiment scores)
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
    """
    Test topic modeling
    
    Args:
        train_df: Training data
        num_topics: Number of topics to extract
        method: Topic modeling method ('lda', 'nmf', or 'bertopic')
        model_dir: Directory to save the model
        
    Returns:
        tuple: (topic modeler, document-term matrix, topic distributions, vocabulary)
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
    """
    Test feature extraction
    
    Args:
        train_df: Training data
        topic_modeler: Fitted topic modeler
        sentiment_analyzer: Sentiment analyzer
        embedding_processor: Embedding processor
        model_dir: Directory to save the model
        
    Returns:
        tuple: (feature extractor, feature matrix, feature names)
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
    """
    Test model training
    
    Args:
        features: Feature matrix
        targets: Target values
        model_type: Type of model to train ('lasso', 'classifier', or 'all')
        model_dir: Directory to save the model
        
    Returns:
        dict: Dictionary containing trained models and results
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
