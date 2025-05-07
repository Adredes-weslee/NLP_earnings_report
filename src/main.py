"""
Main entry point for the Enhanced NLP Earnings Report Analysis.
This script provides the main command-line interface for running the pipeline.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import subprocess

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nlp_earnings_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('nlp_earnings_main')

# Import pipeline components
from data.pipeline import DataPipeline
from nlp.embedding import EmbeddingProcessor
from nlp.sentiment import SentimentAnalyzer
from nlp.topic_modeling import TopicModeler
from nlp.feature_extraction import FeatureExtractor

def setup_directories():
    """Create necessary directories for output files."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/embeddings", exist_ok=True)
    os.makedirs("models/sentiment", exist_ok=True)
    os.makedirs("models/topics", exist_ok=True)
    os.makedirs("models/features", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    logger.info("Created output directories")

def run_data_pipeline(args):
    """Run the data preprocessing pipeline."""
    logger.info("Starting data pipeline")
    
    try:
        # Create and run pipeline
        pipeline = DataPipeline(
            input_file=args.input_file,
            text_column=args.text_column,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        
        # Run pipeline stages
        pipeline.load_data()
        pipeline.clean_text()
        pipeline.compute_text_statistics()
        pipeline.process_financial_data()
        pipeline.generate_labels()
        pipeline.split_data()
        version_id = pipeline.save_splits()
        
        logger.info(f"Data pipeline completed successfully. Version: {version_id}")
        return version_id
    
    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        raise

def run_nlp_analysis(args, version_id=None):
    """Run the advanced NLP analysis."""
    logger.info("Starting NLP analysis")
    
    try:
        # Load the data
        pipeline = DataPipeline()
        if version_id is None:
            # Get the latest version
            from data.data_versioner import DataVersioner
            versioner = DataVersioner()
            version_id = versioner.get_latest_version()
            
            if version_id is None:
                logger.error("No data versions found. Run data pipeline first.")
                return False
        
        logger.info(f"Using data version: {version_id}")
        data_paths = pipeline.get_data_paths(version_id)
        
        # Load the split datasets
        import pandas as pd
        train_df = pd.read_csv(data_paths['train'])
        val_df = pd.read_csv(data_paths['val'])
        test_df = pd.read_csv(data_paths['test'])
        
        logger.info(f"Loaded {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
        
        # 1. Create and train embedding processor
        logger.info("Training embedding processor")
        embedding_processor = EmbeddingProcessor(method=args.embedding_method)
        texts = train_df['processed_text'].fillna('').tolist()
        embedding_processor.fit(texts, max_features=args.max_features)
        
        # Save the embedding processor
        embedding_processor.save(f'models/embeddings/{args.embedding_method}_{args.max_features}')
        
        # 2. Initialize sentiment analyzer
        logger.info("Initializing sentiment analyzer")
        sentiment_analyzer = SentimentAnalyzer(method=args.sentiment_method)
        
        # Save the sentiment analyzer
        sentiment_analyzer.save(f'models/sentiment/{args.sentiment_method}')
        
        # 3. Create and train topic model
        logger.info(f"Training topic model with {args.num_topics} topics")
        # Get document-term matrix from embedding processor
        dtm = embedding_processor.vectorizer.transform(texts)
        feature_names = embedding_processor.vocab
        
        topic_modeler = TopicModeler(method=args.topic_method, num_topics=args.num_topics)
        topic_modeler.fit(dtm, feature_names)
        
        # Save the topic model
        topic_modeler.save(f'models/topics/{args.topic_method}_model')
        
        # 4. Feature extraction and model training
        logger.info("Extracting features and training models")
        feature_extractor = FeatureExtractor()
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
        
        # Train regression model if target exists
        if 'BHAR0_2' in train_df.columns:
            y_reg_train = train_df['BHAR0_2'].values
            
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=0.001, max_iter=10000)
            lasso.fit(X_train, y_reg_train)
            
            # Set feature importances
            feature_extractor.set_feature_importances(lasso.coef_, feature_names)
            
            # Save feature extractor
            feature_extractor.save('models/features/combined_features')
            
            # Print top features
            logger.info("Top predictive features:")
            top_features = feature_extractor.get_top_features(n=10)
            for _, row in top_features.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.6f}")
        
        logger.info("NLP analysis completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in NLP analysis: {str(e)}")
        raise

def run_dashboard():
    """Launch the Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard")
    
    try:
        # Check if Streamlit is installed
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], 
                      check=True, stdout=subprocess.PIPE)
        
        # Launch the dashboard
        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", dashboard_path])
        
        logger.info(f"Dashboard launched: {dashboard_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")
        return False

def main():
    """Main entry point function."""
    parser = argparse.ArgumentParser(description='Enhanced NLP Earnings Report Analysis')
    
    # Add arguments
    parser.add_argument('--action', type=str, default='all',
                        choices=['data', 'nlp', 'dashboard', 'all'],
                        help='Action to perform: data processing, nlp analysis, launch dashboard, or all')
    
    # Data pipeline arguments
    parser.add_argument('--input_file', type=str, default='data/ExpTask2Data.csv.gz',
                        help='Path to input data file')
    parser.add_argument('--text_column', type=str, default='sent',
                        help='Name of text column in input data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # NLP analysis arguments
    parser.add_argument('--embedding_method', type=str, default='tfidf',
                        choices=['tfidf', 'count', 'word2vec', 'transformer'],
                        help='Method for text embedding')
    parser.add_argument('--max_features', type=int, default=5000,
                        help='Maximum number of features for text vectorization')
    parser.add_argument('--sentiment_method', type=str, default='loughran_mcdonald',
                        choices=['loughran_mcdonald', 'textblob', 'vader', 'transformer', 'combined'],
                        help='Method for sentiment analysis')
    parser.add_argument('--topic_method', type=str, default='lda',
                        choices=['lda', 'nmf', 'gensim_lda'],
                        help='Method for topic modeling')
    parser.add_argument('--num_topics', type=int, default=40,
                        help='Number of topics for topic modeling')
    
    args = parser.parse_args()
    
    # Create output directories
    setup_directories()
    
    # Run requested actions
    if args.action in ['data', 'all']:
        try:
            version_id = run_data_pipeline(args)
            logger.info(f"Data pipeline completed successfully with version: {version_id}")
        except Exception as e:
            logger.error(f"Data pipeline failed: {str(e)}")
            if args.action == 'data':
                return
    else:
        version_id = None
    
    if args.action in ['nlp', 'all']:
        try:
            success = run_nlp_analysis(args, version_id)
            if success:
                logger.info("NLP analysis completed successfully")
            else:
                logger.error("NLP analysis failed")
                if args.action == 'nlp':
                    return
        except Exception as e:
            logger.error(f"NLP analysis failed: {str(e)}")
            if args.action == 'nlp':
                return
    
    if args.action in ['dashboard', 'all']:
        try:
            success = run_dashboard()
            if success:
                logger.info("Dashboard launched successfully")
            else:
                logger.error("Failed to launch dashboard")
        except Exception as e:
            logger.error(f"Dashboard launch failed: {str(e)}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting Enhanced NLP Earnings Report Analysis at {start_time}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Analysis completed in {duration}")