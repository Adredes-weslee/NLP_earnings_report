"""
Consolidated topic modeling module for financial text analysis.
Provides LDA and transformer-based topic modeling for earnings reports.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
import sys
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer

# Import configuration values
from ..config import (NUM_TOPICS, RANDOM_STATE, TOPIC_WORD_PRIOR, DOC_TOPIC_PRIOR_FACTOR,
                  MAX_FEATURES, NGRAM_RANGE, MAX_DOC_FREQ, TOPIC_MODEL_PATH, MODEL_DIR,
                  OPTIMAL_TOPICS, LDA_MAX_ITER, LDA_LEARNING_DECAY, LDA_LEARNING_OFFSET)
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
from wordcloud import WordCloud

# Optional imports for BERTopic
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

logger = logging.getLogger('topic_modeler')

class TopicModeler:
    """
    Class for topic modeling on financial texts.
    Supports LDA, NMF, and BERTopic approaches.
    """
    def __init__(self, method: str = 'lda', num_topics: int = NUM_TOPICS, random_state: int = RANDOM_STATE,
                 topic_word_prior: float = TOPIC_WORD_PRIOR, doc_topic_prior_factor: float = DOC_TOPIC_PRIOR_FACTOR):
        """
        Initialize the topic modeler.
        
        Args:
            method: Topic modeling method ('lda', 'nmf', 'bertopic')
            num_topics: Number of topics to extract (not used for BERTopic)
            random_state: Random seed for reproducibility
            topic_word_prior: Topic-word prior for LDA (alpha)
            doc_topic_prior_factor: Document-topic prior factor for LDA (beta)
        """
        self.method = method
        self.num_topics = num_topics
        self.random_state = random_state
        self.topic_word_prior = topic_word_prior
        self.doc_topic_prior_factor = doc_topic_prior_factor
        
        self.model = None
        self.vectorizer = None
        self.topic_words = None
        self.feature_names = None
        self.topic_word_distributions = None
        
        logger.info(f"Initializing TopicModeler with method={method}, num_topics={num_topics}")
        
        # Check if BERTopic is available when needed
        if method == 'bertopic' and not BERTOPIC_AVAILABLE:
            logger.warning("BERTopic method requested but not available. Falling back to LDA.")
            self.method = 'lda'
    
    def create_document_term_matrix(self, texts: List[str], save_path: str = None) -> Tuple:
        """
        Create a document-term matrix from cleaned texts
        
        Args:
            texts: Cleaned text data
            save_path: Path to save the vectorizer
            
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
        
        logger.info(f"DTM shape (documents x features): {dtm.shape}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(vec, f)
            logger.info(f"Vectorizer saved to {save_path}")
        
        self.vectorizer = vec
        self.feature_names = vocab
        return dtm, vec, vocab
    
    def optimize_num_topics(self, dtm, vocab=None, 
                           min_topics: int = 10, max_topics: int = 100, step: int = 5, 
                           sample_size: int = 2000, save_results: bool = True,
                           output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Tune LDA model by testing a range of topic counts
        
        Args:
            dtm: Document-term matrix
            vocab: List of vocabulary terms (if None, uses self.feature_names)
            min_topics: Minimum number of topics to try
            max_topics: Maximum number of topics to try
            step: Step size for topic count
            sample_size: Number of documents to sample for tuning
            save_results: Whether to save the tuning results
            output_dir: Directory to save results
            
        Returns:
            List of dictionaries with coherence scores by topic count
        """
        vocab_to_use = vocab if vocab is not None else self.feature_names
        
        if dtm.shape[0] <= sample_size:
            sample = pd.DataFrame(dtm.todense())
        else:
            sample = pd.DataFrame(dtm.todense()).sample(sample_size, random_state=self.random_state)
        
        records = []
        for top in range(min_topics, max_topics + 1, step):
            logger.info(f"Fitting LDA with {top} topics...")
            record = {'topics': top}
            
            doc_topic_prior = min(1.0, self.doc_topic_prior_factor/top)
            lda = LatentDirichletAllocation(
                n_components=top,
                topic_word_prior=self.topic_word_prior,
                doc_topic_prior=doc_topic_prior,  # Use scaled value
                n_jobs=-1,
                max_iter=LDA_MAX_ITER,
                learning_decay=LDA_LEARNING_DECAY,
                learning_offset=LDA_LEARNING_OFFSET,
                random_state=self.random_state
            )
            
            lda.fit(sample)
            
            umass = metric_coherence_gensim(
                'u_mass',
                topic_word_distrib=lda.components_,
                vocab=vocab_to_use,
                dtm=sample.values
            )
            
            record['mean_umass'] = np.mean(umass)
            records.append(record)
        
        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_df = pd.DataFrame(records)
            results_df.to_csv(os.path.join(output_dir, 'lda_tuning_results.csv'), index=False)
        
        return records
    
    def plot_topic_coherence(self, records: List[Dict[str, Any]], 
                            save_plot: bool = True, 
                            output_dir: str = None) -> Tuple[int, plt.Figure]:
        """
        Plot the topic coherence scores from LDA tuning
        
        Args:
            records: List of dictionaries with tuning results
            save_plot: Whether to save the plot
            output_dir: Directory to save the plot
            
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
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'topic_coherence_plot.png'))
        
        # Update optimal number of topics
        self.num_topics = optimal_topics
        
        return optimal_topics, plt.gcf()
    
    def fit(self, dtm, feature_names=None, vectorizer=None, n_topics=None, 
        save_model=True, model_dir=None):
        """
        Fit the topic model with the specified number of topics
        
        Args:
            dtm: Document-term matrix
            feature_names: Optional list of feature names (vocabulary)
            n_topics: Number of topics (if None, uses self.num_topics)
            save_model: Whether to save the model
            model_dir: Directory to save the model. If None, uses TOPIC_MODEL_PATH from config
            
        Returns:
            tuple: (topic model, topic distribution matrix)
        """
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Update num_topics if provided and ensure it's an integer
        if n_topics is not None:
            self.num_topics = int(n_topics)
        
        # Store the vectorizer if provided
        if vectorizer is not None:
            self.vectorizer = vectorizer
        
        # Ensure num_topics is an integer regardless of how it was set
        if not isinstance(self.num_topics, int):
            self.num_topics = int(self.num_topics)
        
        logger.info(f"Fitting {self.method} model with {self.num_topics} topics...")
        
        if self.method == 'lda':            
            # Calculate doc_topic_prior and ensure it's in valid range [0, 1]
            doc_topic_prior = min(1.0, self.doc_topic_prior_factor/self.num_topics)
            model = LatentDirichletAllocation(
                n_components=self.num_topics,
                topic_word_prior=self.topic_word_prior,
                doc_topic_prior=doc_topic_prior,  # Use scaled value
                n_jobs=-1,
                max_iter=LDA_MAX_ITER,
                learning_decay=LDA_LEARNING_DECAY,
                learning_offset=LDA_LEARNING_OFFSET,
                random_state=self.random_state
            )
        elif self.method == 'nmf':
            model = NMF(
                n_components=self.num_topics,
                random_state=self.random_state
            )
        elif self.method == 'bertopic' and BERTOPIC_AVAILABLE:
            model = BERTopic(
                nr_topics=self.num_topics,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported topic modeling method: {self.method}")
        
        if self.method != 'bertopic':
            model.fit(dtm)
            topics = model.transform(dtm)
            self.topic_word_distributions = model.components_
        else:
            # BERTopic requires documents, not DTM
            if hasattr(dtm, 'todense'):
                dtm = dtm.todense()
            topics, _ = model.fit_transform(dtm)
        
        self.model = model
        
        logger.info(f"Topic distribution matrix shape: {topics.shape}")
        
        # Use config path if model_dir is not specified
        save_dir = model_dir if model_dir is not None else TOPIC_MODEL_PATH
        
        if save_model:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the model
            model_path = os.path.join(save_dir, f'{self.method}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Topic model saved to {model_path}")
            
            # Save topic distributions if not too large
            if topics.shape[0] * topics.shape[1] < 10000000:  # ~10M elements
                # Fix: Use save_dir instead of model_dir
                np.save(os.path.join(save_dir, 'topic_distributions.npy'), topics)
                logger.info(f"Topic distributions saved to {os.path.join(save_dir, 'topic_distributions.npy')}")
        
        return model, topics
    
    def get_top_words(self, n_words: int = 10, 
                     save_results: bool = True, 
                     output_dir: Optional[str] = None) -> Dict[int, List[str]]:
        """
        Get the top words for each topic in the model
        
        Args:
            n_words: Number of top words to extract per topic
            save_results: Whether to save the results
            output_dir: Directory to save the results
            
        Returns:
            dict: Dictionary mapping topic indices to lists of top words
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() first.")
            return {}
        
        topics_words = {}
        
        if self.method in ['lda', 'nmf']:
            for topic_idx, topic in enumerate(self.model.components_):
                top_indices = topic.argsort()[-(n_words):][::-1]
                top_words = [self.feature_names[i] for i in top_indices]
                topics_words[topic_idx] = top_words
        elif self.method == 'bertopic':
            topics = self.model.get_topics()
            for topic_idx in topics:
                if topic_idx != -1:  # -1 is reserved for outliers in BERTopic
                    topics_words[topic_idx] = [word for word, _ in self.model.get_topic(topic_idx)[:n_words]]
        
        self.topic_words = topics_words
        
        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as text file
            with open(os.path.join(output_dir, 'topic_top_words.txt'), 'w') as f:
                for topic_idx, words in topics_words.items():
                    f.write(f"Topic {topic_idx}: {', '.join(words)}\n")
            
            # Save as JSON
            with open(os.path.join(output_dir, 'topic_words.json'), 'w') as f:
                json.dump(topics_words, f, indent=2)
        
        return topics_words
    
    def plot_wordcloud(self, topic_idx: int, n_words: int = 30, 
                      figsize: Tuple[int, int] = (10, 6), 
                      background_color: str = 'white') -> plt.Figure:
        """
        Generate a word cloud for a specific topic
        
        Args:
            topic_idx: Index of the topic to visualize
            n_words: Number of words to include
            figsize: Figure size
            background_color: Background color of the word cloud
            
        Returns:
            matplotlib.figure.Figure: Word cloud figure
        """
        if self.model is None or self.feature_names is None:
            logger.error("Model not fitted or feature names not available.")
            return None
        
        # Get word weights for the topic
        if self.method in ['lda', 'nmf']:
            topic = self.model.components_[topic_idx]
            word_weights = {self.feature_names[i]: topic[i] for i in topic.argsort()[:-n_words-1:-1]}
        elif self.method == 'bertopic':
            word_weights = {word: weight for word, weight in self.model.get_topic(topic_idx)[:n_words]}
        else:
            logger.error(f"Word cloud not supported for method {self.method}")
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color=background_color,
            colormap='viridis',
            prefer_horizontal=1.0
        ).generate_from_frequencies(word_weights)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'Word Cloud for Topic {topic_idx}')
        ax.axis('off')
        
        return fig
    
    def transform(self, dtm):
        """
        Transform document-term matrix to topic distributions
        
        Args:
            dtm: Document-term matrix
            
        Returns:
            array: Topic distributions for input documents
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() first.")
            return None
        
        if self.method in ['lda', 'nmf']:
            return self.model.transform(dtm)
        elif self.method == 'bertopic':
            # BERTopic requires documents, not DTM
            if hasattr(dtm, 'todense'):
                dtm = dtm.todense()
            topics, probs = self.model.transform(dtm)
            return probs
        else:
            logger.error(f"Transform not supported for method {self.method}")
            return None
    
    def save(self, path: str):
        """
        Save the topic modeler and model
        
        Args:
            path: Path to save the topic modeler
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'method': self.method,
            'num_topics': self.num_topics,
            'random_state': self.random_state,
            'topic_word_prior': self.topic_word_prior,
            'doc_topic_prior_factor': self.doc_topic_prior_factor,
            'feature_names': self.feature_names,
            'topic_words': self.topic_words
        }
        
        # Save the state
        with open(f"{path}_state.pkl", 'wb') as f:
            pickle.dump(state, f)
        
        # Save the model if available
        if self.model is not None:
            with open(f"{path}_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
        
        logger.info(f"Topic modeler saved to {path}")
    @classmethod
    def load(cls, path: str) -> 'TopicModeler':
        """
        Load a topic modeler
        
        Args:
            path: Path to load the topic modeler from
            
        Returns:
            TopicModeler: Loaded topic modeler
        """
        # Load state
        with open(f"{path}_state.pkl", 'rb') as f:
            state = pickle.load(f)
        
        # Create instance
        instance = cls(
            method=state['method'],
            num_topics=state['num_topics'],
            random_state=state['random_state'],
            topic_word_prior=state.get('topic_word_prior', 0.01),
            doc_topic_prior_factor=state.get('doc_topic_prior_factor', 50.0)
        )
        
        # Restore state
        instance.feature_names = state['feature_names']
        instance.topic_words = state['topic_words']
        
        # Load model if available
        try:
            with open(f"{path}_model.pkl", 'rb') as f:
                instance.model = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Model file not found at {path}_model.pkl")
        
        return instance


# # Standalone functions for backward compatibility

# def create_document_term_matrix(texts, save_path=None):
#     """
#     Create a document-term matrix from cleaned texts.
#     Wrapper around TopicModeler's method for backward compatibility.
    
#     Args:
#         texts: Cleaned text data
#         save_path: Path to save the vectorizer
        
#     Returns:
#         tuple: (document-term matrix, vectorizer object, feature names)
#     """
#     modeler = TopicModeler()
#     return modeler.create_document_term_matrix(texts, save_path)

# def tune_lda_topics(dtm, vocab, sample_size=2000, save_results=True, output_dir=None):
#     """
#     Tune LDA model by testing a range of topic counts.
#     Wrapper around TopicModeler's method for backward compatibility.
    
#     Args:
#         dtm: Document-term matrix
#         vocab: List of vocabulary terms
#         sample_size: Number of documents to sample for tuning
#         save_results: Whether to save the tuning results
#         output_dir: Directory to save results
        
#     Returns:
#         dict: Records of coherence scores by topic count
#     """
#     modeler = TopicModeler()
#     modeler.feature_names = vocab
#     return modeler.optimize_num_topics(
#         dtm, vocab, sample_size=sample_size, save_results=save_results, output_dir=output_dir
#     )

# def plot_topic_coherence(records, save_plot=True, output_dir=None):
#     """
#     Plot the topic coherence scores from LDA tuning.
#     Wrapper around TopicModeler's method for backward compatibility.
    
#     Args:
#         records: List of dictionaries with tuning results
#         save_plot: Whether to save the plot
#         output_dir: Directory to save the plot
        
#     Returns:
#         tuple: (optimal topic count, plot)
#     """
#     modeler = TopicModeler()
#     return modeler.plot_topic_coherence(records, save_plot=save_plot, output_dir=output_dir)

# def fit_lda_model(dtm, n_topics=None, save_model=True, model_dir=None):
#     """
#     Fit the final LDA model with the optimal number of topics.
#     Wrapper around TopicModeler's method for backward compatibility.
    
#     Args:
#         dtm: Document-term matrix
#         n_topics: Number of topics
#         save_model: Whether to save the model
#         model_dir: Directory to save the model
        
#     Returns:
#         tuple: (LDA model, topic distribution matrix)
#     """
#     modeler = TopicModeler(num_topics=n_topics if n_topics is not None else 40)
#     return modeler.fit(dtm, save_model=save_model, model_dir=model_dir)

# def get_top_words(lda_model, vocab, n_words=10, save_results=True, output_dir=None):
#     """
#     Get the top words for each topic in the LDA model.
#     Wrapper around TopicModeler's method for backward compatibility.
    
#     Args:
#         lda_model: Fitted LDA model
#         vocab: Vocabulary list
#         n_words: Number of top words to extract per topic
#         save_results: Whether to save the results
#         output_dir: Directory to save the results
        
#     Returns:
#         dict: Dictionary mapping topic indices to lists of top words
#     """
#     modeler = TopicModeler()
#     modeler.model = lda_model
#     modeler.feature_names = vocab
#     return modeler.get_top_words(n_words=n_words, save_results=save_results, output_dir=output_dir)
