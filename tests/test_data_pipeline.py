"""
Test script for the enhanced data pipeline.
"""

import os
import pandas as pd
import logging
import sys
from pathlib import Path

# Debug print statements for paths
print("="*50)
print("TEST FILE DEBUG INFO:")
print("Current working directory:", os.getcwd())
print("__file__ value:", __file__)
print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
print("="*50)

# Add the project root directory to Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Project root path:", project_root)
sys.path.insert(0, project_root)
print("sys.path after insertion:", sys.path)
print("="*50)

# Try direct import first
try:
    print("Attempting direct import...")
    # Now import from src directly
    from src.data.pipeline import DataPipeline
    from src.data.text_processor import TextProcessor
    from src.data.data_versioner import DataVersioner
    from src.config import DATA_DIR, RAW_DATA_PATH, PROCESSED_DATA_DIR
    print("Imports successful")
except Exception as e:
    print(f"Direct import failed with error: {e}")
    print("Trying alternative import method...")
    
    # Add specific paths
    sys.path.append(os.path.join(project_root, 'src'))
    print("Added src directory to sys.path:", os.path.join(project_root, 'src'))
    
    try:
        # Import again with updated paths
        from data.pipeline import DataPipeline
        from data.text_processor import TextProcessor
        from data.data_versioner import DataVersioner
        from config import DATA_DIR, RAW_DATA_PATH, PROCESSED_DATA_DIR
        print("Alternative imports successful")
    except Exception as e:
        print(f"Alternative import also failed: {e}")
        raise

print("="*50)
print("After imports - checking paths:")
print("DATA_DIR:", DATA_DIR)
print("RAW_DATA_PATH:", RAW_DATA_PATH) 
print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)
print("="*50)

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
    data_path = RAW_DATA_PATH
    print("Testing data path:", data_path)
    
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
        output_paths = pipeline.run_pipeline(output_dir=PROCESSED_DATA_DIR)
        
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
            print(f"Checking output path for {path_name}: {file_path}")
            if os.path.exists(file_path):
                # Use "[OK]" instead of checkmark symbol to avoid Unicode issues
                logger.info(f"{path_name}: [OK] ({file_path})")
            else:
                logger.error(f"{path_name}: [FAIL] (File not created)")
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