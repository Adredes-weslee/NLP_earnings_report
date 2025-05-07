"""
Data versioning module for NLP Earnings Report project.
Manages versioning information for different data states.
"""

import os
import json
import pandas as pd
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger('data_versioner')

class DataVersioner:
    def __init__(self, base_dir="data"):
        """
        Initialize the data versioner
        
        Args:
            base_dir (str): Base directory for data storage
        """
        self.base_dir = base_dir
        self.versions_file = os.path.join(base_dir, "versions.json")
        self.versions = self._load_versions()
        
    def _load_versions(self):
        """Load existing versions information"""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    versions = json.load(f)
                logger.info(f"Loaded {len(versions)} existing data versions")
                return versions
            except Exception as e:
                logger.error(f"Error loading versions file: {str(e)}")
                return {}
        else:
            logger.info("No existing versions file found, creating new")
            return {}
    
    def _save_versions(self):
        """Save versions information to disk"""
        os.makedirs(os.path.dirname(self.versions_file), exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
        logger.info(f"Saved versions info with {len(self.versions)} entries")
    
    def register_version(self, version_id, config, description=""):
        """
        Register a new data version
        
        Args:
            version_id (str): Unique identifier for this version
            config (dict): Configuration details for this version
            description (str): Human-readable description
        
        Returns:
            str: The version ID that was registered
        """
        if version_id in self.versions:
            logger.warning(f"Version {version_id} already exists, overwriting")
            
        self.versions[version_id] = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "config": config
        }
        
        self._save_versions()
        return version_id
    
    def get_version_info(self, version_id):
        """
        Get information about a specific version
        
        Args:
            version_id (str): The version to retrieve
            
        Returns:
            dict: Version information or None if not found
        """
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return None
        return self.versions[version_id]
    
    def list_versions(self):
        """
        List all available data versions
        
        Returns:
            list: List of version IDs
        """
        return list(self.versions.keys())
    
    def get_latest_version(self):
        """
        Get the most recent data version
        
        Returns:
            str: Latest version ID or None if no versions exist
        """
        if not self.versions:
            return None
            
        # Find the latest version by creation timestamp
        latest = max(self.versions.keys(), 
                    key=lambda k: self.versions[k]['created_at'])
        return latest
        
    def get_version_data_paths(self, version_id, processed_dir="data/processed"):
        """
        Get file paths for a specific version's data files
        
        Args:
            version_id (str): The data version
            processed_dir (str): Directory containing processed data
            
        Returns:
            dict: Paths to train, val, and test data files
        """
        return {
            "train": f"{processed_dir}/train_{version_id}.csv",
            "val": f"{processed_dir}/val_{version_id}.csv",
            "test": f"{processed_dir}/test_{version_id}.csv",
            "config": f"{processed_dir}/config_{version_id}.json"
        }