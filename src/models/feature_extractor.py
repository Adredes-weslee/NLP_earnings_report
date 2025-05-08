# feature_extractor.py
# Functions for feature extraction and topic modeling

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import matplotlib.pyplot as plt
import os
import sys
import pickle

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (MAX_FEATURES, MAX_DOC_FREQ, NGRAM_RANGE, TOPIC_RANGE_MIN, 
                   TOPIC_RANGE_MAX, TOPIC_RANGE_STEP, TOPIC_WORD_PRIOR, 
                   DOC_TOPIC_PRIOR_FACTOR, SAMPLE_SIZE, OPTIMAL_TOPICS,
                   RANDOM_STATE, OUTPUT_DIR, MODEL_DIR)

def create_document_term_matrix(texts, save_path=None):
    """
    Create a document-term matrix from cleaned texts
    
    Args:
        texts (list or pandas.Series): Cleaned text data
        save_path (str, optional): Path to save the vectorizer
        
    Returns:
        tuple: (document-term matrix, vectorizer object, feature names)
    """
    from nltk.corpus import stopwords
    stops = stopwords.words('english')
    
    vec = CountVectorizer(
        token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b',
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        stop_words=stops,
        max_df=MAX_DOC_FREQ
    )
    
    dtm = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    
    print(f"DTM shape (documents x features): {dtm.shape}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vec, f)
        print(f"Vectorizer saved to {save_path}")
    
    return dtm, vec, vocab

def tune_lda_topics(dtm, vocab, sample_size=SAMPLE_SIZE, save_results=True):
    """
    Tune LDA model by testing a range of topic counts
    
    Args:
        dtm (scipy.sparse.csr.csr_matrix): Document-term matrix
        vocab (list): List of vocabulary terms
        sample_size (int): Number of documents to sample for tuning
        save_results (bool): Whether to save the tuning results
        
    Returns:
        dict: Records of coherence scores by topic count
    """
    if dtm.shape[0] <= sample_size:
        sample = pd.DataFrame(dtm.todense())
    else:
        sample = pd.DataFrame(dtm.todense()).sample(sample_size, random_state=RANDOM_STATE)
    
    records = []
    
    for top in range(TOPIC_RANGE_MIN, TOPIC_RANGE_MAX, TOPIC_RANGE_STEP):
        print(f"Fitting LDA with {top} topics...")
        record = {'topics': top}
        
        lda = LDA(
            n_components=top,
            topic_word_prior=TOPIC_WORD_PRIOR,
            doc_topic_prior=DOC_TOPIC_PRIOR_FACTOR/top,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        
        lda.fit(sample)
        
        umass = metric_coherence_gensim(
            'u_mass',
            topic_word_distrib=lda.components_,
            vocab=vocab,
            dtm=sample.values
        )
        
        record['mean_umass'] = np.mean(umass)
        records.append(record)
    
    if save_results and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df = pd.DataFrame(records)
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'lda_tuning_results.csv'), index=False)
    
    return records

def plot_topic_coherence(records, save_plot=True):
    """
    Plot the topic coherence scores from LDA tuning
    
    Args:
        records (list): List of dictionaries with tuning results
        save_plot (bool): Whether to save the plot
        
    Returns:
        tuple: (optimal topic count, plot)
    """
    topic_counts = [rec['topics'] for rec in records]
    umass_means = [rec['mean_umass'] for rec in records]
    
    # Find optimal topic count (least negative coherence score)
    optimal_idx = np.argmax(umass_means)
    optimal_topics = topic_counts[optimal_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(topic_counts, umass_means, marker='o')
    plt.axvline(x=optimal_topics, color='r', linestyle='--', 
                label=f'Optimal: {optimal_topics} topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Mean UMass Coherence')
    plt.title('Topic Coherence vs. Number of Topics')
    plt.legend()
    plt.grid(True)
    
    if save_plot and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'topic_coherence_plot.png'))
    
    return optimal_topics, plt

def fit_lda_model(dtm, n_topics=None, save_model=True):
    """
    Fit the final LDA model with the optimal number of topics
    
    Args:
        dtm (scipy.sparse.csr.csr_matrix): Document-term matrix
        n_topics (int, optional): Number of topics. If None, uses OPTIMAL_TOPICS from config
        save_model (bool): Whether to save the model
        
    Returns:
        tuple: (LDA model, topic distribution matrix)
    """
    if n_topics is None:
        n_topics = OPTIMAL_TOPICS
    
    print(f"Fitting final LDA model with {n_topics} topics...")
    
    final_lda = LDA(
        n_components=n_topics,
        topic_word_prior=TOPIC_WORD_PRIOR,
        doc_topic_prior=DOC_TOPIC_PRIOR_FACTOR/n_topics,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    final_lda.fit(dtm)
    topics = final_lda.transform(dtm)
    
    print(f"Topic distribution matrix shape: {topics.shape}")
    
    if save_model and MODEL_DIR:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, 'lda_model.pkl'), 'wb') as f:
            pickle.dump(final_lda, f)
        print(f"LDA model saved to {os.path.join(MODEL_DIR, 'lda_model.pkl')}")
    
    return final_lda, topics

def get_top_words(lda_model, vocab, n_words=5, save_results=True):
    """
    Get the top words for each topic in the LDA model
    
    Args:
        lda_model: Fitted LDA model
        vocab (list): Vocabulary list
        n_words (int): Number of top words to extract per topic
        save_results (bool): Whether to save the results
        
    Returns:
        dict: Dictionary mapping topic indices to lists of top words
    """
    topics_words = {}
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-(n_words):][::-1]
        top_words = [vocab[i] for i in top_indices]
        topics_words[topic_idx] = top_words
    
    if save_results and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, 'topic_top_words.txt'), 'w') as f:
            for topic_idx, words in topics_words.items():
                f.write(f"Topic {topic_idx}: {', '.join(words)}\n")
    
    return topics_words

def combine_features(topic_features=None, sentiment_features=None, financial_features=None, embeddings=None):
    """
    Combine different types of features into a single feature matrix.
    
    Args:
        topic_features (numpy.ndarray, optional): Topic distribution features
        sentiment_features (numpy.ndarray, optional): Sentiment analysis features
        financial_features (numpy.ndarray, optional): Extracted financial metrics
        embeddings (numpy.ndarray, optional): Text embedding features
        
    Returns:
        tuple: (Combined feature matrix, List of feature names)
    """
    features_to_combine = []
    feature_names = []
    
    # Add topic features if provided
    if topic_features is not None:
        features_to_combine.append(topic_features)
        feature_names.extend([f"topic_{i}" for i in range(topic_features.shape[1])])
        print(f"Added {topic_features.shape[1]} topic features")
    
    # Add sentiment features if provided
    if sentiment_features is not None:
        features_to_combine.append(sentiment_features)
        if isinstance(sentiment_features, pd.DataFrame):
            feature_names.extend(sentiment_features.columns)
            sentiment_features = sentiment_features.values
        else:
            feature_names.extend([f"sentiment_{i}" for i in range(sentiment_features.shape[1])])
        print(f"Added {sentiment_features.shape[1]} sentiment features")
    
    # Add financial features if provided
    if financial_features is not None:
        features_to_combine.append(financial_features)
        if isinstance(financial_features, pd.DataFrame):
            feature_names.extend(financial_features.columns)
            financial_features = financial_features.values
        else:
            feature_names.extend([f"financial_{i}" for i in range(financial_features.shape[1])])
        print(f"Added {financial_features.shape[1]} financial features")
    
    # Add embedding features if provided
    if embeddings is not None:
        # Optionally reduce dimensionality if embeddings are too large
        if embeddings.shape[1] > 50:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
            embeddings = svd.fit_transform(embeddings)
            print(f"Reduced embedding dimensions from {embeddings.shape[1]} to 50")
        
        features_to_combine.append(embeddings)
        feature_names.extend([f"embedding_{i}" for i in range(embeddings.shape[1])])
        print(f"Added {embeddings.shape[1]} embedding features")
    
    if not features_to_combine:
        raise ValueError("No features provided to combine")
    
    # Handle case of only one feature set
    if len(features_to_combine) == 1:
        return features_to_combine[0], feature_names
    
    # Otherwise combine all features horizontally
    combined_features = np.hstack(features_to_combine)
    print(f"Combined feature matrix shape: {combined_features.shape}")
    
    return combined_features, feature_names

def extract_financial_metrics(self, text):
    """
    Extract financial metrics and ratios from text using regex patterns.
    
    Args:
        text (str): Input financial text
        
    Returns:
        dict: Dictionary of extracted financial metrics
    """
    import re
    
    if not isinstance(text, str):
        return {}
    
    metrics = {}
    
    # Patterns for common financial metrics with units
    patterns = {
        'revenue_million': r'revenue (?:of )?\$?(\d+(?:\.\d+)?)\s*million',
        'revenue_billion': r'revenue (?:of )?\$?(\d+(?:\.\d+)?)\s*billion',
        'gross_margin': r'gross margin (?:of )?(\d+(?:\.\d+)?)%',
        'operating_margin': r'operating margin (?:of )?(\d+(?:\.\d+)?)%',
        'profit_margin': r'(?:profit|net) margin (?:of )?(\d+(?:\.\d+)?)%',
        'eps': r'(?:EPS|earnings per share) (?:of )?\$?(\d+(?:\.\d+)?)',
        'diluted_eps': r'diluted (?:EPS|earnings per share) (?:of )?\$?(\d+(?:\.\d+)?)',
        'yoy_growth': r'(?:year[- ]over[- ]year|y-o-y|yoy) growth (?:of )?(\d+(?:\.\d+)?)%',
        'qoq_growth': r'(?:quarter[- ]over[- ]quarter|q-o-q|qoq) growth (?:of )?(\d+(?:\.\d+)?)%',
        'cash': r'cash (?:and cash equivalents )?(?:of )?\$?(\d+(?:\.\d+)?)\s*(?:million|billion)',
        'debt': r'(?:debt|loans) (?:of )?\$?(\d+(?:\.\d+)?)\s*(?:million|billion)',
    }
    
    # Extract metrics
    for name, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for i, match in enumerate(matches):
            # If multiple matches, use suffix to distinguish
            key = f"{name}_{i+1}" if i > 0 else name
            value = float(match.group(1))
            metrics[key] = value
    
    return metrics

class FeatureExtractorPipeline:
    """
    Pipeline for extracting features from text data using multiple feature extractors.
    Combines topic modeling, sentiment analysis, and financial metric extraction.
    """
    
    def __init__(self, use_topics=True, use_sentiment=True, use_metrics=True, use_embeddings=False):
        """
        Initialize the feature extractor pipeline.
        
        Args:
            use_topics (bool): Whether to include topic features
            use_sentiment (bool): Whether to include sentiment features
            use_metrics (bool): Whether to include extracted financial metrics
            use_embeddings (bool): Whether to include text embeddings
        """
        self.use_topics = use_topics
        self.use_sentiment = use_sentiment
        self.use_metrics = use_metrics
        self.use_embeddings = use_embeddings
        
        self.topic_model = None
        self.sentiment_analyzer = None
        self.metric_extractor = None
        self.embedding_model = None
        self.vectorizer = None
        self.feature_names = None
        self.feature_importance = None
        
        print(f"Initialized FeatureExtractorPipeline with: topics={use_topics}, "
              f"sentiment={use_sentiment}, metrics={use_metrics}, embeddings={use_embeddings}")
    
    def set_topic_model(self, model):
        """Set the topic model to use for feature extraction"""
        self.topic_model = model
        return self
    
    def set_sentiment_analyzer(self, analyzer):
        """Set the sentiment analyzer to use for feature extraction"""
        self.sentiment_analyzer = analyzer
        return self
    
    def set_metric_extractor(self, extractor):
        """Set the financial metric extractor to use"""
        self.metric_extractor = extractor
        return self
    
    def set_embedding_model(self, model, vectorizer=None):
        """Set the embedding model to use for text embeddings"""
        self.embedding_model = model
        self.vectorizer = vectorizer
        return self
    
    def extract_topic_features(self, texts):
        """Extract topic distribution features from texts"""
        if not self.topic_model:
            print("Warning: Topic model not set, skipping topic feature extraction")
            return None
        
        if self.vectorizer:
            dtm = self.vectorizer.transform(texts)
        else:
            # Create DTM if vectorizer not provided
            dtm, self.vectorizer, _ = create_document_term_matrix(texts)
        
        topic_distributions = self.topic_model.transform(dtm)
        
        # Create feature names for topics
        topic_feature_names = [f"topic_{i}" for i in range(topic_distributions.shape[1])]
        
        return topic_distributions, topic_feature_names
    
    def extract_sentiment_features(self, texts):
        """Extract sentiment features from texts"""
        if not self.sentiment_analyzer:
            print("Warning: Sentiment analyzer not set, skipping sentiment feature extraction")
            return None
        
        sentiment_features = []
        feature_names = []
        
        for text in texts:
            sentiment = self.sentiment_analyzer.analyze(text)
            sentiment_features.append(list(sentiment.values()))
            
            # Set feature names on first iteration
            if not feature_names:
                feature_names = [f"sentiment_{k}" for k in sentiment.keys()]
        
        return np.array(sentiment_features), feature_names
    
    def extract_financial_metrics(self, texts):
        """Extract financial metrics from texts"""
        if not self.metric_extractor:
            print("Warning: Metric extractor not set, skipping financial metric extraction")
            return None
        
        metric_features = []
        feature_names = []
        
        for text in texts:
            metrics = self.metric_extractor.extract_features(text)
            
            # Set feature names on first iteration
            if not feature_names:
                feature_names = list(metrics.keys())
                
            # Get values in consistent order
            values = [metrics.get(name, 0.0) for name in feature_names]
            metric_features.append(values)
        
        return np.array(metric_features), feature_names
    
    def extract_embedding_features(self, texts, reduce_dim=None):
        """Extract text embedding features"""
        if not self.embedding_model:
            print("Warning: Embedding model not set, skipping embedding feature extraction")
            return None
        
        # Generate embeddings
        embeddings = self.embedding_model.transform(texts)
        
        # Reduce dimensionality if requested
        if reduce_dim and reduce_dim < embeddings.shape[1]:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=reduce_dim, random_state=RANDOM_STATE)
            embeddings = svd.fit_transform(embeddings)
            feature_names = [f"embedding_{i}" for i in range(reduce_dim)]
        else:
            feature_names = [f"embedding_{i}" for i in range(embeddings.shape[1])]
        
        return embeddings, feature_names
    
    def extract_features(self, df, text_column='text'):
        """
        Extract all features from text data.
        
        Args:
            df (pandas.DataFrame): DataFrame containing text data
            text_column (str): Name of column containing text
            
        Returns:
            tuple: (feature matrix, feature names)
        """
        texts = df[text_column].fillna('').tolist()
        features = []
        all_feature_names = []
        
        # Extract topic features if enabled
        if self.use_topics and self.topic_model:
            topic_features, topic_names = self.extract_topic_features(texts)
            features.append(topic_features)
            all_feature_names.extend(topic_names)
            print(f"Added {len(topic_names)} topic features")
        
        # Extract sentiment features if enabled
        if self.use_sentiment and self.sentiment_analyzer:
            sentiment_features, sentiment_names = self.extract_sentiment_features(texts)
            features.append(sentiment_features)
            all_feature_names.extend(sentiment_names)
            print(f"Added {len(sentiment_names)} sentiment features")
        
        # Extract financial metric features if enabled
        if self.use_metrics and self.metric_extractor:
            metric_features, metric_names = self.extract_financial_metrics(texts)
            features.append(metric_features)
            all_feature_names.extend(metric_names)
            print(f"Added {len(metric_names)} financial metric features")
        
        # Extract embedding features if enabled
        if self.use_embeddings and self.embedding_model:
            embedding_features, embedding_names = self.extract_embedding_features(texts, reduce_dim=50)
            features.append(embedding_features)
            all_feature_names.extend(embedding_names)
            print(f"Added {len(embedding_names)} embedding features")
        
        if not features:
            raise ValueError("No features extracted. Check feature extraction settings and models.")
        
        # Combine all feature sets
        combined_features = np.hstack(features)
        self.feature_names = all_feature_names
        
        print(f"Total features extracted: {combined_features.shape[1]}")
        return combined_features, all_feature_names
    
    def set_feature_importances(self, importances, feature_names=None):
        """
        Set feature importance values.
        
        Args:
            importances (array): Feature importance values
            feature_names (list, optional): Feature names
        """
        names = feature_names if feature_names is not None else self.feature_names
        
        if names is None or len(importances) != len(names):
            print("Warning: Feature names don't match importance values")
            self.feature_importance = {i: importance for i, importance in enumerate(importances)}
        else:
            self.feature_importance = {name: importance for name, importance in zip(names, importances)}
    
    def get_top_features(self, n=20):
        """
        Get the top n most important features.
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            pandas.DataFrame: DataFrame with feature names and importance values
        """
        if not self.feature_importance:
            raise ValueError("Feature importances not set. Use set_feature_importances first.")
        
        # Create sorted DataFrame of feature importances
        importance_df = pd.DataFrame({
            "feature": list(self.feature_importance.keys()),
            "importance": list(self.feature_importance.values())
        })
        
        # Sort by absolute importance since negative values can be important too
        importance_df["abs_importance"] = importance_df["importance"].abs()
        importance_df = importance_df.sort_values("abs_importance", ascending=False).head(n)
        importance_df = importance_df.drop("abs_importance", axis=1)
        
        return importance_df
    
    def plot_feature_importance(self, n=20, figsize=(12, 10)):
        """
        Plot feature importances.
        
        Args:
            n (int): Number of top features to show
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        top_features = self.get_top_features(n)
        
        plt.figure(figsize=figsize)
        colors = ['red' if x < 0 else 'blue' for x in top_features['importance']]
        plt.barh(y=top_features['feature'], width=top_features['importance'], color=colors)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {n} Feature Importances')
        plt.tight_layout()
        
        # Save the plot if output directory is specified
        if OUTPUT_DIR:
            os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
            plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'feature_importances.png'))
            print(f"Feature importance plot saved to {os.path.join(OUTPUT_DIR, 'figures', 'feature_importances.png')}")
        
        return plt.gcf()
    
    def save(self, path):
        """
        Save the feature extractor pipeline.
        
        Args:
            path (str): Path to save the pipeline
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create a dictionary with the state to save
        state = {
            'use_topics': self.use_topics,
            'use_sentiment': self.use_sentiment,
            'use_metrics': self.use_metrics,
            'use_embeddings': self.use_embeddings,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Feature extractor pipeline saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a feature extractor pipeline.
        
        Args:
            path (str): Path to load the pipeline from
            
        Returns:
            FeatureExtractorPipeline: Loaded pipeline
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance with the saved settings
        instance = cls(
            use_topics=state.get('use_topics', True),
            use_sentiment=state.get('use_sentiment', True),
            use_metrics=state.get('use_metrics', True),
            use_embeddings=state.get('use_embeddings', False)
        )
        
        # Restore state
        instance.feature_names = state.get('feature_names')
        instance.feature_importance = state.get('feature_importance')
        
        return instance

if __name__ == "__main__":
    import src.data.data_processor as dp
    
    print("Loading data...")
    try:
        df = pd.read_csv("./task2_data_clean.csv.gz")
        print("Using pre-processed data.")
    except:
        print("Pre-processed data not found. Processing raw data...")
        df = dp.load_data()
        df = dp.process_data(df, save_path='./task2_data_clean.csv.gz')
    
    print("Creating document-term matrix...")
    dtm, vec, vocab = create_document_term_matrix(df['clean_sent'], 
                                                save_path=os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    
    print("Tuning LDA topics...")
    records = tune_lda_topics(dtm, vocab)
    
    print("Plotting topic coherence...")
    optimal_topics, _ = plot_topic_coherence(records)
    
    print(f"Fitting final LDA model with {optimal_topics} topics...")
    lda_model, topics = fit_lda_model(dtm, n_topics=optimal_topics)
    
    print("Extracting top words per topic...")
    topics_words = get_top_words(lda_model, vocab)
    
    print("Feature extraction complete!")
    
    # Save topic distributions
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, 'topic_distributions.npy'), topics)
        print(f"Topic distributions saved to {os.path.join(OUTPUT_DIR, 'topic_distributions.npy')}")