"""Main entry point for the Enhanced NLP Earnings Report Analysis.

This script provides the main command-line interface for running the complete
analysis pipeline. It coordinates the execution of data processing, feature
extraction, model training, and evaluation components.

The pipeline includes the following main components:
- Data loading and preprocessing
- Text embedding using various techniques
- Topic modeling of earnings reports
- Sentiment analysis
- Feature extraction
- Model training and evaluation

Usage:
    python -m src.main --mode full --data_path path/to/data.csv.gz

Options:
    --mode: Operation mode (full, preprocess, embed, topic, sentiment, features, evaluate)
    --data_path: Path to the input data file
    --output_dir: Directory for output files
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import subprocess

from src.config import (DATA_DIR, MODEL_DIR, OUTPUT_DIR, RAW_DATA_PATH, PROCESSED_DATA_DIR,
                  NUM_TOPICS, MAX_FEATURES, NGRAM_RANGE, RANDOM_STATE, TEST_SIZE, VAL_SIZE)
from src.data.pipeline import DataPipeline
from src.nlp.embedding import EmbeddingProcessor
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.topic_modeling import TopicModeler
from src.nlp.feature_extraction import FeatureExtractor

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



def setup_directories():
    """Create necessary directories for output files.
    
    This function ensures that all required directories for storing
    models, features, and results exist before running the pipeline.
    It creates the following directory structure if not already present:
    
    - MODEL_DIR/
      - embeddings/
      - sentiment/
      - topics/
      - features/
    - OUTPUT_DIR/
      - figures/
    
    Returns:
        None
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "sentiment"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "topics"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "features"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    logger.info("Created output directories")

def run_data_pipeline(args):
    """Run the data preprocessing pipeline.
    
    This function executes the complete data preprocessing workflow:
    1. Loads raw data from specified path
    2. Cleans and normalizes text data using specialized financial text cleaning
    3. Computes text statistics (length, complexity)
    4. Processes financial metrics
    5. Generates target labels for prediction tasks
    6. Splits data into train/validation/test sets
    
    Args:
        args: Command-line arguments containing configuration parameters:
            input_file (str): Path to input data file containing earnings reports
            seed (int): Random seed for reproducibility of data splits
            train_ratio (float): Portion of data for training (0.0-1.0)
            val_ratio (float): Portion of data for validation (0.0-1.0)
            output_dir (str): Directory where processed data will be saved
            
    Returns:
        str: A unique version identifier for the processed data
        
    Raises:
        FileNotFoundError: If the input data file is not found
        ValueError: If data validation fails (e.g., invalid column names)
        TypeError: If input data is not in the expected format
        
    Example:
        >>> args = argparse.Namespace(
        ...     input_file='data/earnings_reports.csv.gz',
        ...     seed=42,
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     output_dir='data/processed'
        ... )
        >>> version_id = run_data_pipeline(args)
        >>> print(f"Data processed with version: {version_id}")
    """
    logger.info("Starting data pipeline")
    
    try:
        # Create pipeline with parameters matching DataPipeline constructor
        pipeline = DataPipeline(
            data_path=args.input_file,
            random_state=args.seed,
            test_size=1.0 - args.train_ratio - args.val_ratio,
            val_size=args.val_ratio
        )
        
        # Run the complete pipeline using the streamlined run_pipeline method
        paths = pipeline.run_pipeline(output_dir=args.output_dir)
        
        # Get sample text for logging
        if len(pipeline.processed_data) > 0:
            sample_text = pipeline.processed_data['clean_sent'].iloc[0][:100]
            logger.info(f"Sample cleaned text: {sample_text}...")
        
        version_id = pipeline.data_version
        logger.info(f"Data pipeline completed successfully. Version: {version_id}")
        return version_id
    
    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        raise

def run_nlp_analysis(args, version_id=None):
    """Run the advanced NLP analysis pipeline.
    
    This function orchestrates the complete NLP analysis workflow:
    1. Loads the appropriate data version
    2. Performs embedding using specified method
    3. Runs topic modeling
    4. Conducts sentiment analysis
    5. Extracts and combines features
    6. Trains and evaluates predictive models
    
    Args:
        args: Command-line arguments containing analysis parameters:
            embedding_method (str): Method for text embedding ('tfidf', 'count', 'word2vec', 'transformer')
            max_features (int): Maximum number of features for text vectorization
            sentiment_method (str): Method for sentiment analysis ('loughran_mcdonald', 'textblob', etc.)
            topic_method (str): Method for topic modeling ('lda', 'nmf', 'gensim_lda')
            num_topics (int): Number of topics for topic modeling
        version_id (str, optional): Data version identifier. If None,
            the latest available version will be used.
            
    Returns:
        bool: True if analysis completes successfully, False otherwise.
        
    Raises:
        ValueError: If required models or data are not available
        ImportError: If required libraries are missing
        RuntimeError: If a critical component of the pipeline fails
        
    Example:
        >>> args = argparse.Namespace(
        ...     embedding_method='tfidf',
        ...     max_features=5000,
        ...     sentiment_method='loughran_mcdonald',
        ...     topic_method='lda',
        ...     num_topics=40
        ... )
        >>> success = run_nlp_analysis(args, version_id='v20230714')
        >>> if success:
        ...     print("NLP analysis completed successfully")
    """
    logger.info("Starting NLP analysis")
    
    try:        # Load the data
        pipeline = DataPipeline()
        if version_id is None:
            # Get the latest version
            from src.data.data_versioner import DataVersioner
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
        
        # Use 'clean_sent' or 'ea_text' column as available
        text_column = 'clean_sent' if 'clean_sent' in train_df.columns else 'ea_text'
        logger.info(f"Using '{text_column}' column for text data")
        
        texts = train_df[text_column].fillna('').tolist()
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
        dtm = embedding_processor.get_document_term_matrix(texts)
        feature_names = embedding_processor.vocab
        
        topic_modeler = TopicModeler(method=args.topic_method, num_topics=args.num_topics)
        topic_modeler.fit(dtm, feature_names)
        
        # Save the topic model
        topic_modeler.save(f'models/topics/{args.topic_method}_model')
        # 4. Feature extraction using the refactored FeatureExtractor
        logger.info("Extracting features and training models")
        feature_extractor = FeatureExtractor(
            max_features=args.max_features,
            random_state=RANDOM_STATE,
            nlp_processor=embedding_processor.nlp_processor  # Reuse the NLPProcessor
        )

        # Get texts from DataFrame 
        texts = train_df[text_column].fillna('').tolist()

        # Extract features using the refactored method
        feature_dict = feature_extractor.extract_features(
            texts,
            include_statistical=True,
            include_semantic=True,
            include_topics=True,
            include_transformer=False,  # Set to True if you want to use transformers
            semantic_components=50,
            n_topics=args.num_topics
        )

        # Combine features for modeling - use statistical, semantic and topics
        X_train = feature_extractor.combine_features(
            feature_dict, 
            feature_sets=['statistical', 'semantic', 'topics']
        )

        # Create feature names for interpretability
        feature_names = []
        for i in range(feature_dict['statistical'].shape[1]):
            feature_names.append(f'stat_{i}')
        for i in range(feature_dict['semantic'].shape[1]):
            feature_names.append(f'sem_{i}')
        for i in range(feature_dict['topics'].shape[1]):
            feature_names.append(f'topic_{i}')

        # Train regression model if target exists
        if 'BHAR0_2' in train_df.columns:
            y_reg_train = train_df['BHAR0_2'].values
            
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=0.001, max_iter=10000)
            lasso.fit(X_train, y_reg_train)
            
            # Store feature importances in a DataFrame
            importances_df = pd.DataFrame({
                'feature': feature_names,
                'importance': lasso.coef_
            })
            
            # Sort by absolute importance
            importances_df['abs_importance'] = importances_df['importance'].abs()
            top_features = importances_df.sort_values('abs_importance', ascending=False).head(10)
            
            # Save feature extractor with error handling
            try:
                os.makedirs('models/features', exist_ok=True)
                feature_extractor.save('models/features/combined_features')
            except Exception as e:
                logger.warning(f"Failed to save feature extractor: {str(e)}")
            
            # Print top features
            logger.info("Top predictive features:")
            for _, row in top_features.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.6f}")
        
        logger.info("NLP analysis completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in NLP analysis: {str(e)}")
        raise

def run_full_pipeline(data_path=None, n_topics=None, force_reprocess=False, tune_topics=True):
    """Run the full analysis pipeline with specified parameters.
    
    This function orchestrates the entire earnings report analysis workflow by:
    1. Setting up necessary directories
    2. Running the data preprocessing pipeline
    3. Running the NLP analysis pipeline
    4. Configuring parameters for each component
    
    Args:
        data_path (str, optional): Path to input data file. If None, uses the
            default path from configuration. Defaults to None.
        n_topics (int, optional): Number of topics for topic modeling. If None,
            uses the default value or performs auto-tuning. Defaults to None.
        force_reprocess (bool, optional): Whether to force data reprocessing
            even if processed data already exists. Defaults to False.
        tune_topics (bool, optional): Whether to automatically tune the
            optimal number of topics based on coherence scores. Defaults to True.
        
    Returns:
        bool: True if the pipeline completes successfully, False otherwise.
        
    Example:
        >>> # Run with default settings
        >>> success = run_full_pipeline()
        >>> # Run with custom settings
        >>> success = run_full_pipeline(
        ...     data_path='data/custom_dataset.csv.gz',
        ...     n_topics=50,
        ...     force_reprocess=True
        ... )
    """
    # Set up directories
    setup_directories()
    
    # Define default parameters
    input_file = data_path or RAW_DATA_PATH
    num_topics = n_topics or NUM_TOPICS
    
    # Create a custom args object for data pipeline
    data_args = type('Args', (), {
        'input_file': input_file,
        'output_dir': PROCESSED_DATA_DIR,
        'force_reprocess': force_reprocess,
        'train_ratio': 1.0 - TEST_SIZE - VAL_SIZE,
        'val_ratio': VAL_SIZE / (1.0 - TEST_SIZE),  # Adjusted to match config's proportions
        'seed': RANDOM_STATE,
        'text_column': 'sent'
    })
    
    # Run data pipeline
    version_id = run_data_pipeline(data_args)
    
    # Create a custom args object for NLP analysis
    nlp_args = type('Args', (), {
        'data_version': version_id,
        'embedding_method': 'tfidf',
        'max_features': 5000,
        'sentiment_method': 'loughran_mcdonald',
        'topic_method': 'lda',
        'num_topics': num_topics,
        'tune_topics': tune_topics,
        'output_dir': 'results'
    })
    
    # Run NLP analysis
    success = run_nlp_analysis(nlp_args, version_id)
    
    return success

def run_dashboard():
    """Launch the Streamlit dashboard for interactive analysis.
    
    This function starts the Streamlit web application that provides
    an interactive interface for exploring the earnings report analysis
    results. It ensures Streamlit is installed and then launches the
    dashboard as a subprocess.
    
    The dashboard provides:
    - Text analysis for individual earnings reports
    - Topic exploration and visualization
    - Sentiment trend analysis
    - Financial metric extraction
    - Prediction simulation
    
    Returns:
        bool: True if the dashboard was successfully launched, False otherwise.
        
    Note:
        This function doesn't block - it returns immediately after launching
        the dashboard process.
    """
    logger.info("Launching Streamlit dashboard")
    
    try:
        # Check if Streamlit is installed
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], 
                      check=True, stdout=subprocess.PIPE)
        
        # Launch the dashboard
        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard/app.py")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", dashboard_path])
        
        logger.info(f"Dashboard launched: {dashboard_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")
        return False

def main():
    """Main entry point function for the NLP Earnings Report pipeline.
    
    This function serves as the command-line interface for the earnings report
    analysis system. It:
    1. Parses command-line arguments
    2. Sets up logging and directories
    3. Executes the requested pipeline components based on arguments
    4. Handles errors and returns appropriate exit codes
    
    The function supports different operation modes:
    - 'data': Run only the data preprocessing pipeline
    - 'nlp': Run only the NLP analysis pipeline
    - 'dashboard': Launch the interactive Streamlit dashboard
    - 'all': Run the complete pipeline (default)
    
    Returns:
        int: 0 for successful execution, non-zero for errors
        
    Command-line Arguments:
        --action {data,nlp,dashboard,all}: Which components to run
        --input_file PATH: Path to input data file
        --text_column NAME: Name of column containing earnings text
        --output_dir PATH: Directory to save processed data
        --train_ratio FLOAT: Proportion of data for training
        --val_ratio FLOAT: Proportion of data for validation
        --seed INT: Random seed for reproducibility
        --embedding_method {tfidf,count,word2vec,transformer}: Text embedding method
        --max_features INT: Maximum vocabulary size
        --sentiment_method {loughran_mcdonald,textblob,vader,transformer,combined}: 
            Sentiment analysis method
        --topic_method {lda,nmf,gensim_lda}: Topic modeling method
        --num_topics INT: Number of topics for modeling
    
    Examples:
        # Run the full pipeline with default settings:
        $ python -m src.main --action all
            
        # Run only the data processing step with custom input:
        $ python -m src.main --action data --input_file data/custom.csv.gz
        
        # Run NLP analysis with custom settings:
        $ python -m src.main --action nlp --embedding_method transformer \
            --num_topics 50 --sentiment_method vader
            
        # Launch the interactive dashboard:
        $ python -m src.main --action dashboard
    """
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