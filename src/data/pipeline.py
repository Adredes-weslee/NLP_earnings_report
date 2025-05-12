"""Data pipeline module for NLP Earnings Report project.

This module provides classes and functions for handling the complete data pipeline:
loading raw financial earnings report data, preprocessing text content,
splitting datasets for training/validation/testing, and handling data versioning.

The main class, DataPipeline, orchestrates the entire process and tracks configuration
settings for reproducibility. It also computes unique data hashes to track different
data versions throughout the preprocessing steps.

Example:
    Basic usage of the data pipeline:
    
    >>> pipeline = DataPipeline()
    >>> pipeline.load_data()
    >>> pipeline.preprocess()
    >>> pipeline.create_splits()
    >>> pipeline.save_splits()
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import hashlib
import logging
import sys
from datetime import datetime

# Import configuration
from src.config import (RAW_DATA_PATH, PROCESSED_DATA_DIR, TEST_SIZE, 
                  VAL_SIZE, RANDOM_STATE)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_pipeline')

class DataPipeline:
    """Handles the complete process of data preparation for NLP analysis.
    
    This class manages the entire data pipeline from loading raw earnings report data
    through preprocessing, splitting, and versioning. It maintains configuration
    settings to ensure reproducibility and tracks data versions using hash signatures.
    
    Attributes:
        data_path (str): Path to the raw data file.
        random_state (int): Random seed for reproducible splits.
        test_size (float): Proportion of data to allocate to test set.
        val_size (float): Proportion of remaining data to allocate to validation.
        data_version (str): Unique hash identifying the dataset version.
        config (dict): Configuration parameters for tracking and reproducibility.
        raw_data (pd.DataFrame): Original unprocessed data.
        processed_data (pd.DataFrame): Data after preprocessing steps.
        train_data (pd.DataFrame): Training split of processed data.
        val_data (pd.DataFrame): Validation split of processed data.
        test_data (pd.DataFrame): Test split of processed data.
    """
    
    def __init__(self, data_path=None, random_state=RANDOM_STATE, test_size=TEST_SIZE, val_size=VAL_SIZE):
        """Initialize the data pipeline with configuration settings.
        
        Args:
            data_path (str, optional): Path to raw data file. If None, uses
                RAW_DATA_PATH from config. Defaults to None.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to RANDOM_STATE from config.
            test_size (float, optional): Proportion of data for test set (0-1).
                Defaults to TEST_SIZE from config.
            val_size (float, optional): Proportion of remaining data for validation (0-1).
                Defaults to VAL_SIZE from config.
                
        Example:
            >>> # Create pipeline with custom settings
            >>> pipeline = DataPipeline(
            ...     data_path='data/earnings_reports_2022.csv', 
            ...     test_size=0.2,
            ...     val_size=0.15
            ... )
        """
        self.data_path = data_path if data_path is not None else RAW_DATA_PATH
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.data_version = None
        self.config = {
            "data_path": data_path,
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _compute_data_hash(self, df):
        """Generate a unique hash fingerprint for the dataset to track versions.
        
        This method creates a reproducible hash signature based on the dataframe's
        content that can be used to identify specific versions of the dataset.
        Useful for tracking data lineage and ensuring reproducibility.
        
        Args:
            df (pd.DataFrame): Dataframe to compute hash for.
            
        Returns:
            str: First 10 characters of MD5 hash representing the data.
            
        Note:
            The hash is computed from pandas' internal hash_pandas_object method,
            which creates a hash based on the dataframe's content, not its memory
            location or object ID.
        """
        data_str = pd.util.hash_pandas_object(df).sum()
        return hashlib.md5(str(data_str).encode()).hexdigest()[:10]
    
    def load_data(self):
        """Load raw earnings report data from file.
        
        Loads data from the configured data_path and computes a version hash.
        Automatically detects and handles gzipped CSV files.
        
        Returns:
            pd.DataFrame: The loaded raw data.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
            pandas.errors.EmptyDataError: If the file exists but is empty.
            pandas.errors.ParserError: If the file cannot be parsed as CSV.
            Exception: Any other errors during file loading.
            
        Example:
            >>> pipeline = DataPipeline()
            >>> raw_data = pipeline.load_data()
            >>> print(f"Loaded {len(raw_data)} records")
        """
        logger.info(f"Loading data from {self.data_path}")
        try:
            if self.data_path.endswith('.gz'):
                self.raw_data = pd.read_csv(self.data_path, compression='gzip')
            else:
                self.raw_data = pd.read_csv(self.data_path)
                
            self.data_version = self._compute_data_hash(self.raw_data)
            self.config["data_version"] = self.data_version
            self.config["num_samples"] = len(self.raw_data)
            logger.info(f"Loaded {len(self.raw_data)} samples, data version: {self.data_version}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess(self, data=None, replace_numbers=True, filter_bad_sentences=True):
        """Preprocess the loaded financial text data for analysis.
        
        Applies text normalization, cleaning, and optional transformations to
        prepare the raw text data for feature extraction and modeling.
        
        Args:
            data (pd.DataFrame, optional): Data to preprocess. If None, uses the
                previously loaded raw_data. Defaults to None.
            replace_numbers (bool, optional): Whether to replace numbers with 
                token placeholders. Defaults to True.
            filter_bad_sentences (bool, optional): Whether to remove malformed
                or problematic sentences. Defaults to True.
                
        Returns:
            pd.DataFrame: The preprocessed data with added 'processed_text' column.
            
        Raises:
            ValueError: If no data is available to process.
            
        Example:
            >>> pipeline = DataPipeline()
            >>> pipeline.load_data()
            >>> processed = pipeline.preprocess(replace_numbers=False)
            >>> processed['processed_text'].iloc[0][:100]  # View first 100 chars
        """
        if data is None:
            data = self.raw_data.copy()
        
        logger.info("Starting preprocessing")
        
        # Record preprocessing config
        self.config["preprocessing"] = {
            "replace_numbers": replace_numbers,
            "filter_bad_sentences": filter_bad_sentences
        }
        
        # Import text processor here to avoid circular imports
        from src.data.text_processor import TextProcessor
        text_processor = TextProcessor()
        
        # Process the text column(s) - assumes 'text' column exists
        if 'text' in data.columns:
            logger.info(f"Processing {len(data)} text records")
            data['processed_text'] = data['text'].apply(
                lambda x: text_processor.process_text(
                    x, 
                    replace_numbers=replace_numbers,
                    filter_bad=filter_bad_sentences
                )
            )
        
        self.processed_data = data
        logger.info("Preprocessing completed")
        return self.processed_data
    
    def clean_text(self, data=None):
        """Clean financial text by removing noise and normalizing content.
        
        Applies text cleaning operations like removing HTML tags, URLs,
        handling special characters, and normalizing whitespace.
        
        Args:
            data (pd.DataFrame, optional): Data containing text to clean. If None,
                uses the previously loaded raw_data. Defaults to None.
                
        Returns:
            pd.DataFrame: DataFrame with added 'clean_sent' column containing
                the cleaned text.
                
        Note:
            This performs simpler cleaning than the full preprocess method and is
            useful for exploratory analysis or when full preprocessing might
            remove too much information.
        """
        if data is None:
            data = self.raw_data.copy()
        
        from src.data.text_processor import TextProcessor
        processor = TextProcessor()
        
        # Determine text column - use 'ea_text' as default if available
        text_col = 'ea_text' if 'ea_text' in data.columns else 'text'
        
        data['clean_sent'] = data[text_col].apply(processor.clean_text)
        self.processed_data = data
        return data

    def compute_text_statistics(self, text_column='clean_sent'):
        """Compute statistical metrics about the processed text data.
        
        Calculates metrics such as character count, word count, sentence count,
        average sentence length, and readability scores for the text corpus.
        
        Args:
            text_column (str, optional): Column name containing the text to analyze.
                Defaults to 'clean_sent'.
                
        Returns:
            dict: Dictionary of text statistics with metrics like:
                - total_chars: Total character count
                - total_words: Total word count
                - mean_sentence_length: Average words per sentence
                - readability_scores: Flesch reading ease and other metrics
                
        Example:
            >>> pipeline = DataPipeline()
            >>> pipeline.load_data()
            >>> pipeline.clean_text()
            >>> stats = pipeline.compute_text_statistics()
            >>> print(f"Average sentence length: {stats['mean_sentence_length']:.1f} words")
        """
        from src.data.data_processor import compute_text_statistics
        return compute_text_statistics(self.processed_data, text_column)

    def process_financial_data(self, data=None):
        """Process financial metrics and indicators in the earnings report data.
        
        Extracts, transforms, or normalizes financial metrics found in the data,
        such as revenue, earnings per share, or growth rates.
        
        Args:
            data (pd.DataFrame, optional): Data to process. If None, uses
                previously processed data. Defaults to None.
                
        Returns:
            pd.DataFrame: Data with processed financial metrics.
            
        Note:
            Implementation is currently minimal and will be expanded in future
            versions to extract more financial indicators automatically.
        """
        if data is None:
            data = self.processed_data
        
        # Add financial processing logic or delegate to data_processor functions
        # For now, just return the data
        self.processed_data = data
        return data

    def generate_labels(self, threshold=0.05, column='BHAR0_2'):
        """Generate binary classification labels based on financial outcomes.
        
        Creates a binary label column based on whether a financial metric
        (typically a return measure) exceeds a specified threshold.
        
        Args:
            threshold (float, optional): The cutoff value for creating the binary
                label. Defaults to 0.05 (5% return).
            column (str, optional): The column name containing the financial metric
                to threshold. Defaults to 'BHAR0_2' (buy-and-hold abnormal return).
                
        Returns:
            pd.DataFrame: The data with an added 'label' column containing
                binary values (0 or 1).
                
        Example:
            >>> pipeline = DataPipeline()
            >>> pipeline.load_data()
            >>> labeled_data = pipeline.generate_labels(threshold=0.03)
            >>> positive_rate = labeled_data['label'].mean() * 100
            >>> print(f"Positive class rate: {positive_rate:.1f}%")
        """
        if column in self.processed_data.columns:
            self.processed_data['label'] = (self.processed_data[column] > threshold).astype(int)
        return self.processed_data
    
    
    def split_data(self, data=None, target_column='BHAR0_2'):
        """Split data into train, validation and test sets.
        
        Divides the dataset into training, validation, and test subsets using
        the configured proportions. Maintains class balance if a target column
        is available by using stratified splitting.
        
        Args:
            data (pd.DataFrame, optional): Data to split. If None, uses
                the processed data. Defaults to None.
            target_column (str, optional): Column name to use for stratification
                in classification tasks. Defaults to 'BHAR0_2'.
                
        Returns:
            tuple: (train_df, val_df, test_df) - The three data splits.
            
        Example:
            >>> pipeline = DataPipeline()
            >>> pipeline.load_data()
            >>> pipeline.preprocess()
            >>> train, val, test = pipeline.split_data()
            >>> print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
        """
        if data is None:
            data = self.processed_data
        
        logger.info(f"Splitting data with test_size={self.test_size}, val_size={self.val_size}")
        
        # Create binary label for classification (if target exists)
        if target_column in data.columns:
            # Define positive as returns > 5%
            data['label'] = (data[target_column] > 0.05).astype(int)
            stratify_col = 'label'
        else:
            stratify_col = None
        
        # First split: training+validation and test
        train_val, test = train_test_split(
            data, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[stratify_col] if stratify_col in data.columns else None
        )
        
        # Second split: training and validation
        train, val = train_test_split(
            train_val,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_val[stratify_col] if stratify_col in train_val.columns else None
        )
        
        self.train_data = train
        self.val_data = val
        self.test_data = test
        
        # Log split information
        logger.info(f"Data split complete: train={len(train)}, val={len(val)}, test={len(test)}")
        self.config["data_splits"] = {
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "target_column": target_column,
            "stratify_column": stratify_col
        }
        
        if stratify_col:
            # Log class distribution
            train_pos = train[stratify_col].mean() * 100
            val_pos = val[stratify_col].mean() * 100
            test_pos = test[stratify_col].mean() * 100
            logger.info(f"Class distribution (% positive): train={train_pos:.1f}%, val={val_pos:.1f}%, test={test_pos:.1f}%")
            
        return train, val, test
    
    def save_splits(self, output_dir="data/processed"):
        """Save the train, validation, and test splits to disk.
        
        Exports the data splits to CSV files and saves the configuration
        metadata to a JSON file for reproducibility.
        
        Args:
            output_dir (str, optional): Directory path to save the files.
                Defaults to "data/processed".
                
        Returns:
            dict: Dictionary with paths to the saved files.
            
        Example:
            >>> pipeline = DataPipeline()
            >>> pipeline.run_pipeline()
            >>> paths = pipeline.save_splits("data/2023_analysis")
            >>> print(f"Data saved to: {paths['train_path']}")
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data splits
        self.train_data.to_csv(f"{output_dir}/train_{self.data_version}.csv", index=False)
        self.val_data.to_csv(f"{output_dir}/val_{self.data_version}.csv", index=False)
        self.test_data.to_csv(f"{output_dir}/test_{self.data_version}.csv", index=False)
        
        # Save configuration
        with open(f"{output_dir}/config_{self.data_version}.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Data splits and configuration saved to {output_dir}")
        
        return {
            "train_path": f"{output_dir}/train_{self.data_version}.csv",
            "val_path": f"{output_dir}/val_{self.data_version}.csv",
            "test_path": f"{output_dir}/test_{self.data_version}.csv",
            "config_path": f"{output_dir}/config_{self.data_version}.json"
        }
    
    def get_data_paths(self, version_id):
        """Get file paths to dataset files for a specific version.
        
        Retrieves the standardized file paths for a given dataset version,
        useful for loading previously processed data.
        
        Args:
            version_id (str): The data version identifier (hash).
            
        Returns:
            dict: Dictionary containing paths to:
                - 'train': Training data CSV file
                - 'val': Validation data CSV file
                - 'test': Test data CSV file
                - 'config': Configuration JSON file
                
        Example:
            >>> pipeline = DataPipeline()
            >>> paths = pipeline.get_data_paths('a7b3c9d1e5')
            >>> train_df = pd.read_csv(paths['train'])
        """
        return {
            "train": f"{PROCESSED_DATA_DIR}/train_{version_id}.csv",
            "val": f"{PROCESSED_DATA_DIR}/val_{version_id}.csv",
            "test": f"{PROCESSED_DATA_DIR}/test_{version_id}.csv",
            "config": f"{PROCESSED_DATA_DIR}/config_{version_id}.json"
        }
    
    def run_pipeline(self, output_dir="data/processed"):
        """Run the full data preprocessing pipeline from loading to saving.
        
        Executes all steps of the data pipeline in sequence: loading raw data,
        preprocessing text, splitting into train/val/test sets, and saving
        the results to disk.
        
        Args:
            output_dir (str, optional): Directory to save the processed data.
                Defaults to "data/processed".
                
        Returns:
            dict: Dictionary with paths to all saved files.
            
        Example:
            >>> pipeline = DataPipeline(data_path='data/new_reports.csv')
            >>> paths = pipeline.run_pipeline()
            >>> print(f"Pipeline complete. Data version: {pipeline.data_version}")
        """
        self.load_data()
        self.preprocess()
        self.split_data()
        return self.save_splits(output_dir)
