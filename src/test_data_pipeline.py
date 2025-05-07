"""
Test script for the enhanced data pipeline.
"""

import os
import pandas as pd
import logging
import sys
from data.pipeline import DataPipeline
from data.text_processor import TextProcessor
from data.data_versioner import DataVersioner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('pipeline_test')

def test_data_pipeline():
    """Test the data pipeline functionality"""
    
    # Test input file path
    data_path = "data/ExpTask2Data.csv.gz"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return False
    
    # Initialize pipeline
    logger.info("Initializing data pipeline")
    pipeline = DataPipeline(
        data_path=data_path, 
        random_state=42, 
        test_size=0.2, 
        val_size=0.15
    )
    
    # Run pipeline
    logger.info("Running data pipeline...")
    try:
        output_paths = pipeline.run_pipeline(output_dir="data/processed")
        
        # Log results
        logger.info(f"Pipeline completed with data version: {pipeline.data_version}")
        logger.info(f"Train set: {pipeline.config['data_splits']['train_size']} samples")
        logger.info(f"Validation set: {pipeline.config['data_splits']['val_size']} samples")
        logger.info(f"Test set: {pipeline.config['data_splits']['test_size']} samples")
        
        # Register version
        versioner = DataVersioner()
        versioner.register_version(
            version_id=pipeline.data_version,
            config=pipeline.config,
            description="Test run of enhanced data pipeline"
        )
        
        # Verify outputs
        success = True
        for path_name, file_path in output_paths.items():
            if os.path.exists(file_path):
                logger.info(f"{path_name}: ✓ ({file_path})")
            else:
                logger.error(f"{path_name}: ✗ (File not created)")
                success = False
        
        return success, pipeline
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None

if __name__ == "__main__":
    success, pipeline = test_data_pipeline()
    if success:
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")