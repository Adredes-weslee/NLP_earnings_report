"""
Test script for the enhanced NLP module.
Demonstrates how to use the advanced NLP features.
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

# Import data pipeline modules
from data.pipeline import DataPipeline
from data.text_processor import TextProcessor

# Import advanced NLP modules
from nlp.embedding import EmbeddingProcessor
from nlp.sentiment import SentimentAnalyzer
from nlp.topic_modeling import TopicModeler
from nlp.feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_nlp_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('advanced_nlp_test')

def load_processed_data(data_version=None):
    """Load processed data from a specific version or the latest version"""
    from data.data_versioner import DataVersioner
    
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
    
    logger.info(f"Loaded {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
    
    return train_df, val_df, test_df

def test_embedding_processor(train_df, val_df):
    """Test the enhanced embedding processor"""
    logger.info("Testing embedding processor...")
    
    # Create embedding processor with TF-IDF
    embedding_processor = EmbeddingProcessor(method='tfidf')
    
    # Fit on training data
    texts = train_df['processed_text'].fillna('').tolist()
    embedding_processor.fit(texts, max_features=5000)
    
    # Transform training and validation data
    train_embeddings = embedding_processor.transform(texts)
    val_texts = val_df['processed_text'].fillna('').tolist()
    val_embeddings = embedding_processor.transform(val_texts)
    
    logger.info(f"Training embeddings shape: {train_embeddings.shape}")
    logger.info(f"Validation embeddings shape: {val_embeddings.shape}")
    
    # Save the embedding processor
    os.makedirs('models/embeddings', exist_ok=True)
    embedding_processor.save('models/embeddings/tfidf_5000')
    
    return embedding_processor

def test_sentiment_analyzer(train_df):
    """Test the sentiment analyzer"""
    logger.info("Testing sentiment analyzer...")
    
    # Create sentiment analyzer with Loughran-McDonald lexicon
    sentiment_analyzer = SentimentAnalyzer(method='loughran_mcdonald')
    
    # Analyze a sample of texts
    sample_size = min(100, len(train_df))
    sample_texts = train_df['processed_text'].head(sample_size).fillna('').tolist()
    
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

def test_topic_modeling(train_df, embedding_processor):
    """Test the topic modeling capabilities"""
    logger.info("Testing topic modeling...")
    
    # Get document-term matrix from embedding processor
    texts = train_df['processed_text'].fillna('').tolist()
    dtm = embedding_processor.vectorizer.transform(texts)
    feature_names = embedding_processor.vocab
    
    # Create topic modeler with LDA
    topic_modeler = TopicModeler(method='lda', num_topics=40)
    
    # Find optimal number of topics
    try:
        topic_range = range(10, 51, 10)  # Try 10, 20, 30, 40, 50 topics
        logger.info(f"Optimizing number of topics in range {min(topic_range)}-{max(topic_range)}...")
        coherence_values, perplexity_values, optimal_num_topics = topic_modeler.optimize_num_topics(
            dtm, feature_names, topic_range)
        logger.info(f"Optimal number of topics: {optimal_num_topics}")
    except Exception as e:
        logger.warning(f"Topic optimization failed: {str(e)}. Using default 40 topics.")
        # Just fit the model with default number of topics
        topic_modeler.fit(dtm, feature_names)
    
    # Get topics
    topics = topic_modeler.get_topics(num_words=10)
    
    # Print top terms for each topic
    logger.info("\nTop terms per topic:")
    for topic_id, terms in topics.items():
        term_str = ", ".join([term for term, _ in terms])
        logger.info(f"Topic {topic_id}: {term_str}")
    
    # Get document-topic distributions
    doc_topics = topic_modeler.get_document_topics(dtm)
    logger.info(f"Document-topic matrix shape: {doc_topics.shape}")
    
    # Save the topic model
    os.makedirs('models/topics', exist_ok=True)
    topic_modeler.save('models/topics/lda_model')
    
    return topic_modeler

def test_feature_extraction(train_df, val_df, embedding_processor, sentiment_analyzer, topic_modeler):
    """Test the combined feature extraction"""
    logger.info("Testing feature extraction...")
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Set components
    feature_extractor.set_embedding_processor(embedding_processor)
    feature_extractor.set_sentiment_analyzer(sentiment_analyzer)
    feature_extractor.set_topic_modeler(topic_modeler)
    
    # Extract features from training data
    X_train, feature_names = feature_extractor.extract_features(
        train_df, 
        text_column='processed_text',
        include_embeddings=True,
        include_topics=True,
        include_sentiment=True
    )
    
    logger.info(f"Combined feature matrix shape: {X_train.shape}")
    logger.info(f"Feature groups: {feature_extractor.get_feature_groups()}")
    
    # Extract features from validation data
    X_val, _ = feature_extractor.extract_features(
        val_df,
        text_column='processed_text'
    )
    
    logger.info(f"Validation feature matrix shape: {X_val.shape}")
    
    # Create target variables for regression and classification
    # For regression: predict BHAR0_2
    if 'BHAR0_2' in train_df.columns:
        y_reg_train = train_df['BHAR0_2'].values
        y_reg_val = val_df['BHAR0_2'].values
        
        # Train a Lasso regression model
        lasso = Lasso(alpha=0.001, max_iter=10000)
        lasso.fit(X_train, y_reg_train)
        
        # Get feature importances
        feature_extractor.set_feature_importances(lasso.coef_, feature_names)
        
        # Print top features
        top_features = feature_extractor.get_top_features(n=20)
        logger.info("\nTop features for predicting returns:")
        for _, row in top_features.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.6f}")
        
        # Print group importances
        group_imp = feature_extractor.get_group_importances()
        logger.info("\nFeature group importances:")
        for _, row in group_imp.iterrows():
            logger.info(f"{row['feature_group']}: {row['importance']:.6f}")
        
        # Save feature importance plot
        os.makedirs('results/figures', exist_ok=True)
        fig = feature_extractor.plot_feature_importances(n=20)
        if fig:
            fig.savefig('results/figures/feature_importances.png')
            plt.close(fig)
    
    # For classification: predict if BHAR0_2 > 0.05 (5% return)
    if 'label' in train_df.columns:
        y_cls_train = train_df['label'].values
        y_cls_val = val_df['label'].values
        
        # Train a Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_cls_train)
        
        # Evaluate on validation set
        val_preds = rf.predict(X_val)
        
        # Print classification report
        logger.info("\nClassification Report (Validation Set):")
        logger.info(classification_report(y_cls_val, val_preds))
    
    # Save the feature extractor configuration
    os.makedirs('models/features', exist_ok=True)
    feature_extractor.save('models/features/combined_features')
    
    return feature_extractor

def run_advanced_nlp_test():
    """Run the full advanced NLP test pipeline"""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Load processed data
    logger.info("Loading processed data...")
    train_df, val_df, test_df = load_processed_data()
    
    if train_df is None:
        logger.error("Failed to load data. Exiting.")
        return False
    
    # 2. Test embedding processor
    embedding_processor = test_embedding_processor(train_df, val_df)
    
    # 3. Test sentiment analyzer
    sentiment_analyzer = test_sentiment_analyzer(train_df)
    
    # 4. Test topic modeling
    topic_modeler = test_topic_modeling(train_df, embedding_processor)
    
    # 5. Test combined feature extraction
    feature_extractor = test_feature_extraction(
        train_df, val_df, embedding_processor, sentiment_analyzer, topic_modeler
    )
    
    logger.info("Advanced NLP test completed successfully!")
    return True

if __name__ == "__main__":
    success = run_advanced_nlp_test()
    if not success:
        logger.error("Advanced NLP test failed.")
        sys.exit(1)