"""
Script to create placeholder models for the Streamlit dashboard.
This allows the dashboard to run without running the full NLP pipeline.
"""

import os
import pickle
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('placeholder_models')

# Import configuration
from config import MODEL_DIR, OUTPUT_DIR

def create_placeholder_models():
    """Create minimal placeholder models for the Streamlit dashboard"""
    
    # Create necessary directories
    os.makedirs(os.path.join(MODEL_DIR, 'embeddings', 'tfidf_5000'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, 'sentiment', 'loughran_mcdonald'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, 'topics', 'lda_model'), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, 'features', 'combined_features'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

    # Create placeholder TF-IDF model
    config_path = os.path.join(MODEL_DIR, 'embeddings', 'tfidf_5000', 'config.joblib')
    vectorizer_path = os.path.join(MODEL_DIR, 'embeddings', 'tfidf_5000', 'vectorizer.joblib')
    
    if not os.path.exists(vectorizer_path):
        logger.info("Creating placeholder TF-IDF vectorizer")
        vectorizer = TfidfVectorizer(max_features=100)
        vectorizer.fit(['placeholder earnings report text for model initialization'])
        
        config = {
            'method': 'tfidf',
            'max_features': 100,
            'vocab_size': len(vectorizer.vocabulary_),
        }
        
        joblib.dump(config, config_path)
        joblib.dump(vectorizer, vectorizer_path)
        logger.info("Created placeholder embedding model")

    # Create placeholder sentiment analyzer
    sentiment_path = os.path.join(MODEL_DIR, 'sentiment', 'loughran_mcdonald', 'sentiment_config.joblib')
    if not os.path.exists(sentiment_path):
        logger.info("Creating placeholder sentiment analyzer")
        sentiment_config = {
            'method': 'loughran_mcdonald',
            'positive_words': ['increase', 'growth', 'profit'],
            'negative_words': ['decrease', 'loss', 'decline'],
            'uncertainty_words': ['may', 'approximately', 'risk'],
            'litigious_words': ['lawsuit', 'litigation', 'claim']
        }
        
        joblib.dump(sentiment_config, sentiment_path)
        logger.info("Created placeholder sentiment model")

    # Create placeholder topic model
    topic_model_path = os.path.join(MODEL_DIR, 'topics', 'lda_model', 'lda_model.pkl')
    if not os.path.exists(topic_model_path):
        logger.info("Creating placeholder topic model")
        # Create a simple structure that mimics what the real model would return
        topic_model = {
            'num_topics': 10,
            'topics': {i: [('word'+str(j), 0.1) for j in range(10)] for i in range(10)},
            'coherence': 0.5,
            'perplexity': -8.5,
            'method': 'lda',
            'model_params': {'num_topics': 10, 'random_state': 42}
        }
        
        with open(topic_model_path, 'wb') as f:
            pickle.dump(topic_model, f)
        logger.info("Created placeholder topic model")

    # Create placeholder feature extractor
    feature_path = os.path.join(MODEL_DIR, 'features', 'combined_features', 'feature_extractor.pkl')
    if not os.path.exists(feature_path):
        logger.info("Creating placeholder feature extractor")
        feature_names = [f'topic_{i}' for i in range(10)] + \
                        ['positive', 'negative', 'uncertainty', 'litigious'] + \
                        [f'word_{i}' for i in range(20)]
        
        feature_extractor = {
            'feature_groups': {
                'topic': [f'topic_{i}' for i in range(10)],
                'sentiment': ['positive', 'negative', 'uncertainty', 'litigious'],
                'embedding': [f'word_{i}' for i in range(20)]
            },
            'feature_names': feature_names,
            'feature_importances': np.random.random(len(feature_names)),
            'components': ['embedding_processor', 'sentiment_analyzer', 'topic_modeler']
        }
        
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_extractor, f)
        logger.info("Created placeholder feature extractor")

    # Create a sample figure for feature importance
    import matplotlib.pyplot as plt
    fig_path = os.path.join(OUTPUT_DIR, 'figures', 'feature_importances.png')
    if not os.path.exists(fig_path):
        logger.info("Creating placeholder feature importance figure")
        plt.figure(figsize=(10, 6))
        
        feature_sample = feature_names[:15]
        importance_sample = np.random.random(15)
        importance_sample = sorted(importance_sample, reverse=True)
        
        plt.barh(range(len(feature_sample)), importance_sample, align='center')
        plt.yticks(range(len(feature_sample)), feature_sample)
        plt.xlabel('Importance')
        plt.title('Feature Importance (Placeholder)')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        logger.info("Created placeholder feature importance figure")

    logger.info("All placeholder models created successfully!")
    return True

if __name__ == "__main__":
    create_placeholder_models()