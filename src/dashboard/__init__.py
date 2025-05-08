"""
Dashboard module for NLP Earnings Report Analysis.
Provides interactive visualization and exploration of earnings report analysis.
"""

# Import dashboard components
from .app import EarningsReportDashboard, main as run_dashboard
from .utils import (
    load_models,
    get_available_models,
    format_topics,
    classify_sentiment,
    format_sentiment_result,
    extract_topic_visualization,
    get_feature_importance_plot,
    get_wordcloud_for_topic,
    create_prediction_simulator,
    create_topic_explorer
)

__all__ = [
    'EarningsReportDashboard',
    'run_dashboard',
    'load_models',
    'get_available_models',
    'format_topics',
    'classify_sentiment',
    'format_sentiment_result',
    'extract_topic_visualization',
    'get_feature_importance_plot',
    'get_wordcloud_for_topic',
    'create_prediction_simulator',
    'create_topic_explorer'
]