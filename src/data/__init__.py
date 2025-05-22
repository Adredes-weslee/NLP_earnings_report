"""
Data processing module for earnings report text analysis.
Handles loading, preprocessing, and versioning of earnings report data.
"""

# Import data processing components
from .pipeline import DataPipeline
from .text_processor import TextProcessor
from .data_versioner import DataVersioner

# Remove references to non-existent functions
__all__ = [
    'DataPipeline',
    'TextProcessor',
    'DataVersioner'
]