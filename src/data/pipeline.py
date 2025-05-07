"""
Enhanced data pipeline for NLP Earnings Report project.
Handles data loading, preprocessing, splitting, and versioning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import hashlib
import logging
from datetime import datetime

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
    def __init__(self, data_path, random_state=42, test_size=0.2, val_size=0.2):
        """
        Initialize the data pipeline with configuration
        
        Args:
            data_path (str): Path to raw data file
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of training data for validation
        """
        self.data_path = data_path
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
        """Generate a unique hash for the dataset to track versions"""
        data_str = pd.util.hash_pandas_object(df).sum()
        return hashlib.md5(str(data_str).encode()).hexdigest()[:10]
    
    def load_data(self):
        """Load raw data from file"""
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
        """Preprocess the loaded data"""
        if data is None:
            data = self.raw_data.copy()
        
        logger.info("Starting preprocessing")
        
        # Record preprocessing config
        self.config["preprocessing"] = {
            "replace_numbers": replace_numbers,
            "filter_bad_sentences": filter_bad_sentences
        }
        
        # Import text processor here to avoid circular imports
        from .text_processor import TextProcessor
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
    
    def split_data(self, data=None, target_column='BHAR0_2'):
        """Split data into train, validation and test sets"""
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
        """Save the train, validation, and test splits to disk"""
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
    
    def run_pipeline(self, output_dir="data/processed"):
        """Run the full data pipeline"""
        self.load_data()
        self.preprocess()
        self.split_data()
        return self.save_splits(output_dir)