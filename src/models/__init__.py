"""
Model training and evaluation for the NLP Earnings Report Analysis project.
This package handles predictive models and feature engineering for financial text analysis.
"""

# Import model components
from .model_trainer import (
    train_model,
    evaluate_model,
    cross_validate,
    get_feature_importance
)

from .feature_extractor import (
    FeatureExtractorPipeline,
    combine_features,
    extract_financial_metrics
)

__all__ = [
    'train_model',
    'evaluate_model',
    'cross_validate',
    'get_feature_importance',
    'FeatureExtractorPipeline',
    'combine_features',
    'extract_financial_metrics'
]