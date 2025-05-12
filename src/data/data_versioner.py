"""Data versioning module for NLP Earnings Report project.

This module implements a lightweight data versioning system for tracking different
versions of processed datasets. It maintains metadata about dataset versions,
including configuration parameters, creation timestamps, and descriptions.

The DataVersioner class provides functionality to:
  - Register new data versions with associated configurations
  - Retrieve information about existing versions
  - List all available versions
  - Get the latest version
  - Retrieve file paths for specific versions' data files

This versioning system helps maintain reproducibility by tracking the exact
configuration used to generate each dataset version.
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
        """Initialize the data versioner.
        
        Sets up the data versioner and loads existing version information if available.
        Creates a new version registry if none exists.
        
        Args:
            base_dir (str): Base directory for data storage. The versions metadata file
                will be stored as "{base_dir}/versions.json".
        
        Examples:
            >>> versioner = DataVersioner()  # Uses default "data" directory
            >>> versioner = DataVersioner("path/to/data_dir")
        """
        self.base_dir = base_dir
        self.versions_file = os.path.join(base_dir, "versions.json")
        self.versions = self._load_versions()
        
    def _load_versions(self):
        """Load existing versions information from the versions file.
        
        Reads the versions.json file if it exists, otherwise initializes an empty dictionary.
        Handles file read errors gracefully by returning an empty dictionary.
        
        Returns:
            dict: Dictionary containing version information or empty dict if file doesn't exist
                  or cannot be read.
        """
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
        """Save versions information to disk.
        
        Creates the necessary directories if they don't exist and
        writes the versions dictionary to the versions.json file with proper
        formatting. This is an internal method called whenever version data
        is modified.
        
        The method ensures that:
        1. The directory structure exists
        2. The versions data is serialized with readable indentation
        3. The save operation is logged
        
        Notes:
            This method is called automatically by other methods that modify
            version information, such as `register_version()`. There is typically
            no need to call it directly.
            
        Raises:
            IOError: If the file cannot be written due to permission issues or
                disk space limitations.
        """
        os.makedirs(os.path.dirname(self.versions_file), exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
        logger.info(f"Saved versions info with {len(self.versions)} entries")
    
    def register_version(self, version_id, config, description=""):
        """Register a new data version.
        
        Creates a new entry in the versions registry with the given ID,
        configuration, and description. Overwrites existing versions with
        the same ID if they exist.
        
        Args:
            version_id (str): Unique identifier for this version. Should be
                meaningful and represent the dataset version (e.g., a hash
                or timestamp).
            config (dict): Configuration details for this version. Should contain
                all parameters used to generate this dataset version for
                reproducibility.
            description (str): Human-readable description of this version.
                Helpful for understanding the purpose of this version.
        
        Returns:
            str: The version ID that was registered.
        
        Examples:
            >>> config = {'preprocessing': 'standard', 'min_token_count': 5}
            >>> versioner.register_version('v1.0', config, 'Initial clean dataset')
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
        """Get information about a specific version.
        
        Retrieves the metadata for a given version ID, including
        creation timestamp, description, and configuration.
        
        Args:
            version_id (str): The version identifier to retrieve.
            
        Returns:
            dict: Version information with keys 'created_at', 'description',
                and 'config', or None if the version was not found.
        
        Examples:
            >>> info = versioner.get_version_info('v1.0')
            >>> if info:
            ...     print(f"Created: {info['created_at']}")
            ...     print(f"Description: {info['description']}")
        """
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return None
        return self.versions[version_id]
    
    def list_versions(self):
        """List all available data versions.
        
        Retrieves the IDs of all registered dataset versions from the internal
        version registry. This method is useful for discovering what dataset 
        versions are available for analysis or comparing versions.
        
        Returns:
            list: List of version IDs in no particular order. Each ID is a string
                that uniquely identifies a registered dataset version. Returns an
                empty list if no versions have been registered yet.
        
        Examples:
            >>> versions = versioner.list_versions()
            >>> print(f"Available versions: {', '.join(versions)}")
            >>> 
            >>> # Check if any versions exist
            >>> if not versioner.list_versions():
            ...     print("No dataset versions found. Run the pipeline first.")
            >>> 
            >>> # Iterate through versions to find one with specific properties
            >>> for version_id in versioner.list_versions():
            ...     info = versioner.get_version_info(version_id)
            ...     if info['config'].get('include_sentiment', False):
            ...         print(f"Found version with sentiment: {version_id}")
        """
        return list(self.versions.keys())
    
    def get_latest_version(self):
        """Get the most recent data version.
        
        Identifies the latest version based on creation timestamp. This is a 
        convenience method that's particularly useful for workflows that should 
        always use the most recent dataset version by default, while still allowing 
        specific versions to be accessed when needed.
        
        The method compares the 'created_at' timestamp of all registered versions
        and returns the ID of the version with the most recent timestamp.
        
        Returns:
            str: Latest version ID or None if no versions exist. The returned ID
                can be used with other methods like `get_version_info()` or
                `get_version_data_paths()` to access the corresponding version data.
        
        Examples:
            >>> latest = versioner.get_latest_version()
            >>> if latest:
            ...     info = versioner.get_version_info(latest)
            ...     print(f"Latest version: {latest}, created {info['created_at']}")
            >>>
            >>> # Use the latest version for analysis by default
            >>> default_version = versioner.get_latest_version() or "fallback_version"
            >>> paths = versioner.get_version_data_paths(default_version)
            >>> train_df = pd.read_csv(paths['train'])
        
        Notes:
            If multiple versions have the exact same timestamp (unlikely but possible),
            the behavior is determined by the Python `max()` function when comparing strings.
        """
        if not self.versions:
            return None
            
        # Find the latest version by creation timestamp
        latest = max(self.versions.keys(), 
                    key=lambda k: self.versions[k]['created_at'])
        return latest
        
    def get_version_data_paths(self, version_id, processed_dir="data/processed"):
        """Get file paths for a specific version's data files.
        
        Constructs standardized paths to train, validation, test, and config
        files for a given dataset version.
        
        Args:
            version_id (str): The data version identifier.
            processed_dir (str): Directory containing processed data files.
                Default is "data/processed".
            
        Returns:
            dict: Dictionary containing paths to data files with keys:
                - 'train': Path to training data file
                - 'val': Path to validation data file
                - 'test': Path to test data file
                - 'config': Path to configuration file
        
        Examples:
            >>> paths = versioner.get_version_data_paths('v1.0')
            >>> train_df = pd.read_csv(paths['train'])
            >>> test_df = pd.read_csv(paths['test'])
        """
        return {
            "train": f"{processed_dir}/train_{version_id}.csv",
            "val": f"{processed_dir}/val_{version_id}.csv",
            "test": f"{processed_dir}/test_{version_id}.csv",
            "config": f"{processed_dir}/config_{version_id}.json"
        }
