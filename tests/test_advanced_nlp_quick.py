"""Test script for the enhanced NLP module with configurable sample size.

This script provides a lightweight version of the NLP testing suite that can
run on a reduced dataset size for faster debugging and development iterations.
It tests all key NLP components including:

- Embedding generation with TF-IDF vectorization
- Sentiment analysis using financial lexicons
- Topic modeling with LDA (reduced number of topics)
- Combined feature extraction from all NLP components

The script supports configuration via command-line arguments to adjust sample size,
features, and topic count for testing efficiency.
"""

import os
import pandas as pd
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from pathlib import Path
import argparse

# Add the project root directory to Python path first
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Then import configuration
from src.config import (NUM_TOPICS, MAX_FEATURES, NGRAM_RANGE, MAX_DOC_FREQ, 
                   SAMPLE_SIZE, CV_FOLDS, TEST_SIZE, VAL_SIZE, RANDOM_STATE,
                   MODEL_DIR, OUTPUT_DIR, FIGURE_DPI)

# Import data pipeline modules
from src.data.pipeline import DataPipeline
from src.data.text_processor import TextProcessor

# Import advanced NLP modules
from src.nlp.embedding import EmbeddingProcessor
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.topic_modeling import TopicModeler
from src.nlp.feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_nlp_quick_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('advanced_nlp_quick_test')

def load_processed_data(data_version=None, sample_size=None):
    """Load processed data from a specific version or the latest version.
    
    Retrieves versioned data splits for NLP processing and optionally limits
    the dataset size for quicker testing iterations.
    
    Args:
        data_version (str, optional): The specific data version ID to load.
            If None, attempts to load the latest version.
        sample_size (int, optional): If provided, limits the data to this number
            of samples for quick testing. Validation and test sets are sized
            proportionally smaller.
    
    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - train_df: Training dataset with features and labels
            - val_df: Validation dataset
            - test_df: Test dataset
    
    Notes:
        - If no data version exists, an appropriate error will be logged
        - When sample_size is specified, random sampling is used to select records
        - Ensures all data splits have the required columns for NLP processing
    """
    from src.data.data_versioner import DataVersioner
    
    versioner = DataVersioner()
    if data_version is None:
        data_version = versioner.get_latest_version()
        
    if data_version is None:
        logger.error("No data versions found. Run data pipeline first.")
        return None, None, None
    
    logger.info(f"Loading data version: {data_version}")
    paths = versioner.get_version_data_paths(data_version)
    
    train_df = pd.read_csv(paths['train'])
    val_df = pd.read_csv(paths['val'])
    test_df = pd.read_csv(paths['test'])
    
    # Apply sample size limit if provided
    if sample_size is not None:
        logger.info(f"Using limited sample size: {sample_size}")
        train_size = min(sample_size, len(train_df))
        val_size = min(sample_size // 5, len(val_df))  # validation set is smaller
        test_size = min(sample_size // 5, len(test_df))  # test set is smaller
        
        train_df = train_df.sample(train_size, random_state=42)
        val_df = val_df.sample(val_size, random_state=42)
        test_df = test_df.sample(test_size, random_state=42)
    
    logger.info(f"Loaded {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
    
    return train_df, val_df, test_df

def test_embedding_processor(train_df, val_df, max_features=5000):
    """Test the enhanced embedding processor with reduced feature count.
    
    Creates and evaluates a TF-IDF embedding processor with configurable
    feature count for faster testing. The processor is fitted on training data
    and then used to transform both training and validation texts.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame with 'ea_text' column
        val_df (pd.DataFrame): Validation data DataFrame with 'ea_text' column
        max_features (int, optional): Maximum number of features for TF-IDF
            vectorization. Defaults to 5000 but can be reduced for quicker testing.
    
    Returns:
        EmbeddingProcessor: The fitted embedding processor ready for text vectorization
    
    Notes:
        - Handles missing text values by filling with empty strings
        - Saves the model to 'models/embeddings/tfidf_5000' for reuse
        - Reports embedding dimensions for both training and validation sets
    """
    logger.info("Testing embedding processor...")
    
    # Create embedding processor with TF-IDF
    embedding_processor = EmbeddingProcessor(method='tfidf')
    
    # Fit on training data
    texts = train_df['ea_text'].fillna('').tolist()
    embedding_processor.fit(texts, max_features=max_features)
    
    # Transform training and validation data
    train_embeddings = embedding_processor.transform(texts)
    val_texts = val_df['ea_text'].fillna('').tolist()
    val_embeddings = embedding_processor.transform(val_texts)
    
    logger.info(f"Training embeddings shape: {train_embeddings.shape}")
    logger.info(f"Validation embeddings shape: {val_embeddings.shape}")
    
    # Save the embedding processor
    os.makedirs('models/embeddings', exist_ok=True)
    embedding_processor.save('models/embeddings/tfidf_5000')
    
    return embedding_processor

def test_sentiment_analyzer(train_df):
    """Test the sentiment analyzer with financial lexicon.
    
    Creates a sentiment analyzer using the Loughran-McDonald financial lexicon
    and analyzes a sample of earnings report texts to extract sentiment metrics.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame with 'ea_text' column
            containing earnings announcement text
    
    Returns:
        SentimentAnalyzer: The configured sentiment analyzer with financial lexicon
    
    Notes:
        - Analyzes only the first 100 samples (or fewer) for efficiency
        - Reports key sentiment metrics: positive, negative, and net sentiment
        - Saves the analyzer to 'models/sentiment/loughran_mcdonald' for reuse
    """
    logger.info("Testing sentiment analyzer...")
    
    # Create sentiment analyzer with Loughran-McDonald lexicon
    sentiment_analyzer = SentimentAnalyzer(method='loughran_mcdonald')
    
    # Analyze a sample of texts
    sample_size = min(100, len(train_df))
    sample_texts = train_df['ea_text'].head(sample_size).fillna('').tolist()
    
    # Get sentiment scores
    sentiment_df = sentiment_analyzer.batch_analyze(sample_texts)
    
    logger.info(f"Sentiment analysis results shape: {sentiment_df.shape}")
    logger.info(f"Sentiment features: {sentiment_df.columns.tolist()}")
    logger.info(f"Average positive sentiment: {sentiment_df['positive'].mean():.4f}")
    logger.info(f"Average negative sentiment: {sentiment_df['negative'].mean():.4f}")
    logger.info(f"Average net sentiment: {sentiment_df['net_sentiment'].mean():.4f}")
    
    # Save the sentiment analyzer
    os.makedirs('models/sentiment', exist_ok=True)
    sentiment_analyzer.save('models/sentiment/loughran_mcdonald')
    
    return sentiment_analyzer

def test_topic_modeling(train_df, embedding_processor, num_topics=10):
    """Test the topic modeling capabilities with reduced complexity for quick testing.
    
    Creates and fits a topic model with a significantly reduced number of topics
    for faster testing. Uses the document-term matrix from the embedding processor
    to avoid redundant vectorization.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame with 'ea_text' column
        embedding_processor (EmbeddingProcessor): Fitted embedding processor that
            contains the vectorizer for generating document-term matrices
        num_topics (int, optional): Number of topics to generate. Defaults to 10,
            which is much lower than production use but suitable for quick testing.
    
    Returns:
        TopicModeler: The fitted topic model with extracted topics
    
    Notes:
        - Reuses the vectorizer from the embedding processor for efficiency
        - Skips hyperparameter optimization to speed up testing
        - Prints top terms for each topic for quick inspection
        - Saves the model to 'models/topics/quick_test/lda_model' for reuse
    """
    logger.info("Testing topic modeling...")
    
    # Get document-term matrix from embedding processor
    texts = train_df['ea_text'].fillna('').tolist()
    dtm = embedding_processor.vectorizer.transform(texts)
    feature_names = embedding_processor.vocab
    
    # Create topic modeler with LDA and fixed number of topics for quick testing
    topic_modeler = TopicModeler(method='lda', num_topics=num_topics)
    
    # Important: Store the vectorizer
    topic_modeler.vectorizer = embedding_processor.vectorizer
    
    # Skip optimization for quick testing and directly fit the model
    logger.info(f"Using fixed number of topics ({num_topics}) for quick testing")
    topic_modeler.fit(dtm, feature_names)
    
    # Get topics
    topics = topic_modeler.get_top_words(n_words=10)
    
    # Print top terms for each topic
    logger.info("\nTop terms per topic:")
    for topic_id, terms in topics.items():
        # Check if terms contains tuples (term, weight) or just strings
        if terms and isinstance(terms[0], tuple):
            # If terms are (term, weight) tuples
            term_str = ", ".join([term for term, _ in terms])
        else:
            # If terms are just strings
            term_str = ", ".join(terms)
        
        logger.info(f"Topic {topic_id}: {term_str}")
    
    # Get document-topic distributions
    doc_topics = topic_modeler.transform(dtm)
    logger.info(f"Document-topic matrix shape: {doc_topics.shape}")
    
    # Save the topic model
    os.makedirs('models/topics/quick_test', exist_ok=True)
    topic_modeler.save('models/topics/quick_test/lda_model')
    
    return topic_modeler

def test_feature_extraction(train_df, val_df, embedding_processor, sentiment_analyzer, topic_modeler):
    """Test the combined feature extraction pipeline.
    
    Creates a feature extractor that integrates multiple NLP components (embeddings,
    topic modeling, and sentiment analysis) into a unified feature matrix for
    machine learning models.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame with 'ea_text' column and target variables
        val_df (pd.DataFrame): Validation data DataFrame with 'ea_text' column
        embedding_processor (EmbeddingProcessor): Fitted embedding processor for text vectorization
        sentiment_analyzer (SentimentAnalyzer): Configured sentiment analyzer
        topic_modeler (TopicModeler): Fitted topic model
    
    Returns:
        FeatureExtractor: The configured feature extractor that combines all NLP components
    
    Notes:
        - Configures the feature extractor to use all available NLP components
        - Extracts features from both training and validation data
        - Reports feature matrix dimensions and feature group information
        - Ensures feature alignment between training and validation sets
        - Prepares target variables for both regression and classification tasks
        - Attempts to save the feature extractor configuration for reproducibility
    """
    logger.info("Testing feature extraction...")
    
    # Create feature extractor with the desired settings upfront
    feature_extractor = FeatureExtractor(
        use_embeddings=True,
        use_topics=True,
        use_sentiment=True
    )
    
    # Set components
    feature_extractor.set_embedding_model(embedding_processor)
    feature_extractor.set_sentiment_analyzer(sentiment_analyzer)
    feature_extractor.set_topic_model(topic_modeler)
    
    # Extract features from training data
    X_train, feature_names = feature_extractor.extract_features(
        train_df, 
        text_column='ea_text'
    )
    
    logger.info(f"Combined feature matrix shape: {X_train.shape}")
    logger.info(f"Feature groups: {feature_extractor.get_feature_groups()}")
    
    # Extract features from validation data 
    X_val, val_feature_names = feature_extractor.extract_features(
        val_df,
        text_column='ea_text'
    )
    
    # CRITICAL FIX: Create DataFrames ONLY if the shapes match, otherwise use temporary names
    if X_val.shape[1] == len(val_feature_names):
        X_val_df = pd.DataFrame(X_val, columns=val_feature_names)
    else:
        # If shapes don't match, create DataFrame with dummy column names first
        logger.warning(f"Feature matrix shape mismatch: Matrix has {X_val.shape[1]} columns but feature_names has {len(val_feature_names)} entries")
        temp_cols = [f"col_{i}" for i in range(X_val.shape[1])]
        X_val_df = pd.DataFrame(X_val, columns=temp_cols)
        
        # Try to match column names where possible
        if len(temp_cols) <= len(val_feature_names):
            for i, col in enumerate(temp_cols):
                X_val_df = X_val_df.rename(columns={col: val_feature_names[i]})
    
    # Create training DataFrame for alignment
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    
    # Ensure validation features match training features
    if len(feature_names) != X_val_df.shape[1] or not all(col in X_val_df.columns for col in feature_names):
        logger.warning(f"Feature mismatch: Training has {len(feature_names)} features, validation has {X_val_df.shape[1]}")
        
        # Add missing columns from training to validation
        for col in feature_names:
            if col not in X_val_df.columns:
                X_val_df[col] = 0
                logger.info(f"Adding missing column '{col}' to validation features")
        
        # Remove extra columns in validation not in training (if any)
        extra_cols = [col for col in X_val_df.columns if col not in feature_names]
        if extra_cols:
            X_val_df = X_val_df.drop(columns=extra_cols)
            logger.info(f"Removed {len(extra_cols)} extra columns from validation features")
        
        # Ensure column order matches training data
        X_val_df = X_val_df[feature_names]
        
        # Convert back to numpy arrays
        X_val = X_val_df.values
        
        logger.info(f"Adjusted validation feature matrix shape: {X_val.shape}")
    
    logger.info(f"Validation feature matrix shape: {X_val.shape}")
    
    # Create target variables for regression and classification
    # For regression: predict BHAR0_2
    if 'BHAR0_2' in train_df.columns:
        y_reg_train = train_df['BHAR0_2'].values
        y_reg_val = val_df['BHAR0_2'].values
        
        # Train a Lasso regression model with fewer iterations for speed
        lasso = Lasso(alpha=0.001, max_iter=1000)
        lasso.fit(X_train, y_reg_train)
        
        # Get feature importances
        if hasattr(feature_extractor, 'set_feature_importances'):
            feature_extractor.set_feature_importances(lasso.coef_, feature_names)
        else:
            # Fallback if the method doesn't exist
            feature_extractor.feature_importance = dict(zip(feature_names, np.abs(lasso.coef_)))
        
        # Print top features
        if hasattr(feature_extractor, 'get_top_features'):
            top_features = feature_extractor.get_top_features(n=20)
            logger.info("\nTop features for predicting returns:")
            for _, row in top_features.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.6f}")
        
        # Print group importances if available
        if hasattr(feature_extractor, 'get_group_importances'):
            group_imp = feature_extractor.get_group_importances()
            logger.info("\nFeature group importances:")
            for _, row in group_imp.iterrows():
                logger.info(f"{row['feature_group']}: {row['importance']:.6f}")
        
        # Save feature importance plot if available
        if hasattr(feature_extractor, 'plot_feature_importances'):
            os.makedirs('results/figures/quick_test', exist_ok=True)
            fig = feature_extractor.plot_feature_importances(n=20)
            if fig:
                fig.savefig('results/figures/quick_test/feature_importances.png')
                plt.close(fig)
    
    # For classification: predict if BHAR0_2 > 0.05 (5% return)
    if 'label' in train_df.columns:
        y_cls_train = train_df['label'].values
        y_cls_val = val_df['label'].values
        
        # Train a Random Forest classifier with fewer trees for speed
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_cls_train)
        
        # Evaluate on validation set
        val_preds = rf.predict(X_val)
        
        # Print classification report
        logger.info("\nClassification Report (Validation Set):")
        logger.info(classification_report(y_cls_val, val_preds))
    
    # Save the feature extractor configuration
    try:
        os.makedirs('models/features/quick_test', exist_ok=True)
        
        # Try an alternative path if the first one fails
        try:
            feature_extractor.save('models/features/quick_test/feature_extractor')
        except PermissionError:
            # Try with a timestamped filename to avoid conflicts
            import time
            timestamp = int(time.time())
            alt_path = f'models/features/quick_test/feature_extractor_{timestamp}'
            logger.warning(f"Permission denied on original path, trying alternative: {alt_path}")
            feature_extractor.save(alt_path)
            
    except Exception as e:
        logger.warning(f"Failed to save feature extractor: {str(e)}")
    
    return feature_extractor

def run_advanced_nlp_test(sample_size=100, max_features=100, num_topics=2):
    """Run the full advanced NLP test pipeline with configurable parameters for quick testing.
    
    Executes the entire NLP pipeline with reduced parameters suitable for quick testing
    and development iterations. The function creates minimal versions of all components
    (embeddings, sentiment analysis, topic modeling, feature extraction) to verify
    the overall workflow without the computational cost of full-scale processing.
    
    Args:
        sample_size (int, optional): Number of data samples to use. Defaults to 100,
            which is sufficient to test pipeline functionality while remaining fast.
        max_features (int, optional): Maximum number of features for embeddings.
            Defaults to 100, significantly reduced from production settings.
        num_topics (int, optional): Number of topics for topic modeling.
            Defaults to 2, which is minimal but sufficient for testing.
    
    Returns:
        bool: True if all tests completed successfully, False otherwise
    
    Examples:
        >>> # Run test with default quick settings
        >>> success = run_advanced_nlp_test()
        >>> 
        >>> # Run with slightly larger parameters for more realistic testing
        >>> success = run_advanced_nlp_test(sample_size=500, max_features=1000, num_topics=10)
    
    Notes:
        - Creates necessary directory structure for models and results
        - All parameters are dramatically reduced from production settings
        - The quick test focuses on validating pipeline functionality, not model quality
        - Models are saved to allow inspection after testing
    """
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Load processed data with limited sample size
    logger.info(f"Running quick test with sample_size={sample_size}, max_features={max_features}, num_topics={num_topics}")
    train_df, val_df, test_df = load_processed_data(sample_size=sample_size)
    
    if train_df is None:
        logger.error("Failed to load data. Exiting.")
        return False
    
    # 2. Test embedding processor with reduced features
    embedding_processor = test_embedding_processor(train_df, val_df, max_features=max_features)
    
    # 3. Test sentiment analyzer
    sentiment_analyzer = test_sentiment_analyzer(train_df)
    
    # 4. Test topic modeling with fewer topics
    topic_modeler = test_topic_modeling(train_df, embedding_processor, num_topics=num_topics)
    
    # 5. Test combined feature extraction
    feature_extractor = test_feature_extraction(
        train_df, val_df, embedding_processor, sentiment_analyzer, topic_modeler
    )
    
    logger.info("Quick Advanced NLP test completed successfully!")
    return True

if __name__ == "__main__":
    # Set up command line arguments for configuring test size
    parser = argparse.ArgumentParser(description='Run advanced NLP tests with configurable sample size')
    parser.add_argument('--sample-size', type=int, default=100, 
                        help='Number of samples to use for quick testing (default: 100)')
    parser.add_argument('--max-features', type=int, default=100,
                        help='Maximum number of features for embeddings (default: 100)')
    parser.add_argument('--num-topics', type=int, default=2,
                        help='Number of topics for topic modeling (default: 2)')
    
    args = parser.parse_args()
    
    success = run_advanced_nlp_test(
        sample_size=args.sample_size,
        max_features=args.max_features,
        num_topics=args.num_topics
    )
    
    if not success:
        logger.error("Advanced NLP quick test failed.")
        sys.exit(1)
