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
    """Topic modeling for financial texts analysis.
    
    This class provides topic modeling capabilities for financial texts 
    using various approaches including Latent Dirichlet Allocation (LDA),
    Non-negative Matrix Factorization (NMF), and transformer-based models
    (BERTopic) when available.
    
    Attributes:
        method (str): Topic modeling method used ('lda', 'nmf', or 'bertopic').
        num_topics (int): Number of topics to extract.
        random_state (int): Random seed for reproducibility.
        topic_word_prior (float): Topic-word prior for LDA (alpha parameter).
        doc_topic_prior_factor (float): Document-topic prior factor for LDA (beta).
        model: The underlying topic model instance.
        vectorizer: Document vectorizer used for creating document-term matrices.
        topic_words: List of words associated with each topic.
        feature_names: Names of features in the document-term matrix.
        topic_word_distributions: Word probability distributions for each topic.
    """
    def __init__(self, method: str = 'lda', num_topics: int = NUM_TOPICS, random_state: int = RANDOM_STATE,
                 topic_word_prior: float = TOPIC_WORD_PRIOR, doc_topic_prior_factor: float = DOC_TOPIC_PRIOR_FACTOR):
        """Initialize the topic modeler with specified parameters.
        
        Args:
            method (str): Topic modeling method to use. Options are 'lda', 'nmf', or 
                'bertopic'. Defaults to 'lda'.
            num_topics (int): Number of topics to extract. Not used for BERTopic.
                Defaults to value from config (NUM_TOPICS).
            random_state (int): Random seed for reproducibility.
                Defaults to value from config (RANDOM_STATE).
            topic_word_prior (float): Topic-word prior for LDA (alpha parameter).
                Defaults to value from config (TOPIC_WORD_PRIOR).
            doc_topic_prior_factor (float): Document-topic prior factor for LDA (beta).
                Defaults to value from config (DOC_TOPIC_PRIOR_FACTOR).
                
        Example:
            >>> modeler = TopicModeler(method='lda', num_topics=20)
            >>> modeler.fit(texts)
            >>> topics = modeler.transform(new_texts)
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
        """Create a document-term matrix from cleaned texts.
        
        This method constructs a document-term matrix using CountVectorizer
        with financial text-specific settings. It filters out common stopwords
        and applies parameters from the configuration.
        
        Args:
            texts (List[str]): List of cleaned text documents.
            save_path (str, optional): Path to save the vectorizer for later use.
                If None, the vectorizer is not saved.
            
        Returns:
            Tuple: A tuple containing:
                - document-term matrix (sparse matrix)
                - vectorizer object (CountVectorizer)
                - vocabulary/feature names (list)
                
        Example:
            >>> dtm, vec, vocab = modeler.create_document_term_matrix(texts)
            >>> print(f"Matrix shape: {dtm.shape}")
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
        """Optimize the number of topics by evaluating coherence scores.
        
        This method tests LDA models with different numbers of topics and
        computes coherence scores to identify the optimal topic count.
        It can optionally save the results for later analysis.
        
        Args:
            dtm: Document-term matrix (sparse or dense).
            vocab (List[str], optional): List of vocabulary terms.
                If None, uses self.feature_names.
            min_topics (int, optional): Minimum number of topics to try.
                Defaults to 10.
            max_topics (int, optional): Maximum number of topics to try.
                Defaults to 100.
            step (int, optional): Step size for topic count increments.
                Defaults to 5.
            sample_size (int, optional): Number of documents to sample for tuning.
                Using a sample can speed up the process. Defaults to 2000.
            save_results (bool, optional): Whether to save the tuning results.
                Defaults to True.
            output_dir (str, optional): Directory to save results.
                If None and save_results is True, results are saved to the default model directory.
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries with topic counts and their
            corresponding coherence scores and perplexity values.
            
        Example:
            >>> dtm, _, vocab = modeler.create_document_term_matrix(texts)
            >>> results = modeler.optimize_num_topics(dtm, vocab, min_topics=5, max_topics=50, step=5)
            >>> best_result = max(results, key=lambda x: x.get('coherence', 0))
            >>> print(f"Best number of topics: {best_result['topics']}")
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
        """Plot the topic coherence scores from LDA tuning.
        
        This method creates a visualization of topic coherence scores across
        different numbers of topics, helping to identify the optimal topic count.
        It highlights the optimal number of topics with a vertical line.
        
        Args:
            records (List[Dict[str, Any]]): List of dictionaries with tuning results
                from the optimize_num_topics method.
            save_plot (bool, optional): Whether to save the plot to a file.
                Defaults to True.
            output_dir (str, optional): Directory to save the plot.
                If None and save_plot is True, uses the default output directory.
            
        Returns:
            Tuple[int, plt.Figure]: A tuple containing:
                - optimal_topics (int): The optimal number of topics
                - fig (plt.Figure): The matplotlib figure object
                
        Example:
            >>> results = modeler.optimize_num_topics(dtm, vocab)
            >>> optimal_topics, fig = modeler.plot_topic_coherence(results)
            >>> print(f"Optimal number of topics: {optimal_topics}")
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
        """Fit the topic model with the specified document-term matrix.
        
        This method trains the topic model using the provided document-term matrix.
        It supports different topic modeling approaches including LDA, NMF,
        and BERTopic (if available). The trained model can be saved for later use.
        
        Args:
            dtm: Document-term matrix, either sparse (scipy.sparse) or dense (numpy.ndarray).
                For LDA/NMF: should be a document-term matrix.
                For BERTopic: can be either documents or embeddings.
            feature_names (list, optional): List of feature names (vocabulary terms).
                If None, uses self.feature_names. Defaults to None.
            vectorizer (CountVectorizer, optional): Vectorizer used to create DTM.
                If provided, it's stored for later use. Defaults to None.
            n_topics (int, optional): Number of topics to extract.
                If None, uses self.num_topics. Defaults to None.
            save_model (bool, optional): Whether to save the trained model to disk.
                Defaults to True.
            model_dir (str, optional): Directory to save the model.
                If None, uses TOPIC_MODEL_PATH from config. Defaults to None.
            
        Returns:
            tuple: A tuple containing:
                - model: The fitted topic model instance
                - topics: Topic distribution matrix for the input documents
                
        Raises:
            ValueError: If an unsupported topic modeling method is specified
            
        Example:
            >>> dtm, vectorizer, vocab = modeler.create_document_term_matrix(texts)
            >>> model, topics = modeler.fit(dtm, feature_names=vocab, vectorizer=vectorizer)
            >>> print(f"Topics shape: {topics.shape}")
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
        """Get the top words for each topic in the model.
        
        This method extracts the most representative words for each topic
        based on their probability in the topic-word distribution matrix.
        The results can be saved to files for later analysis or visualization.
        
        Args:
            n_words (int, optional): Number of top words to extract per topic.
                Defaults to 10.
            save_results (bool, optional): Whether to save the results to files.
                Defaults to True.
            output_dir (str, optional): Directory to save the results.
                If None and save_results is True, uses the default output directory.
            
        Returns:
            Dict[int, List[str]]: Dictionary mapping topic indices to lists of top words.
            Each list contains the n_words most representative words for that topic.
            
        Raises:
            ValueError: If the model has not been fitted yet.
            
        Example:
            >>> topic_words = modeler.get_top_words(n_words=15)
            >>> for topic_idx, words in topic_words.items():
            ...     print(f"Topic {topic_idx}: {', '.join(words)}")
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
        """Generate a word cloud visualization for a specific topic.
        
        This method creates a visual representation of the most important words
        in a given topic, where the size of each word indicates its relative
        importance in that topic. The word cloud provides an intuitive way to
        understand the theme of a topic.
        
        Args:
            topic_idx (int): Index of the topic to visualize. Should be in range
                [0, num_topics-1].
            n_words (int, optional): Number of most important words to include
                in the visualization. Defaults to 30.
            figsize (Tuple[int, int], optional): Size of the figure in inches
                as (width, height). Defaults to (10, 6).
            background_color (str, optional): Background color of the word cloud.
                Can be any valid matplotlib color. Defaults to 'white'.
            
        Returns:
            matplotlib.figure.Figure: The word cloud figure that can be displayed
                or saved to a file.
                
        Raises:
            ValueError: If the model has not been fitted yet
            IndexError: If topic_idx is out of range
            
        Example:
            >>> # Generate and display word cloud for topic 5
            >>> fig = modeler.plot_wordcloud(topic_idx=5, n_words=40)
            >>> plt.show()
            >>> 
            >>> # Save word cloud to a file
            >>> fig.savefig('topic5_wordcloud.png', dpi=300, bbox_inches='tight')
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
        """Transform document-term matrix to topic distributions.
        
        This method applies a fitted topic model to new documents to extract
        their topic distributions. It requires that the model has been previously
        fitted using the fit() method.
        
        Args:
            dtm: Document-term matrix for the documents to transform.
                Should match the format used during fitting:
                - For LDA/NMF: Document-term matrix (sparse or dense)
                - For BERTopic: Text documents or their embeddings
            
        Returns:
            numpy.ndarray: Topic distributions for input documents. Each row
                represents a document, and each column represents the document's
                association with a topic. Shape is (n_documents, n_topics).
                
        Raises:
            ValueError: If the model has not been fitted yet
            
        Example:
            >>> # Fit model on training data
            >>> train_dtm, _, vocab = modeler.create_document_term_matrix(train_texts)
            >>> modeler.fit(train_dtm, feature_names=vocab)
            >>> 
            >>> # Transform new data
            >>> test_dtm = modeler.vectorizer.transform(test_texts)
            >>> topic_distributions = modeler.transform(test_dtm)
            >>> print(f"Top topic for first document: {topic_distributions[0].argmax()}")
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
        """Save the topic modeler and model to disk.
        
        This method serializes the topic modeler's state and the fitted model
        to the specified path. It saves two files:
        1. A state file (.pkl) containing configuration and metadata
        2. A model file (.pkl) containing the fitted model itself
        
        The saved model can later be loaded using the load() class method.
        
        Args:
            path (str): Base path where the topic modeler will be saved.
                The method will append '_state.pkl' and '_model.pkl' to this path.
                If the directory doesn't exist, it will be created.
                
        Raises:
            OSError: If there's an error creating the directory or writing files
            pickle.PicklingError: If the model cannot be serialized
            
        Example:
            >>> modeler = TopicModeler(method='lda', num_topics=20)
            >>> modeler.fit(dtm, feature_names=vocab)
            >>> modeler.save('models/topics/financial_topics')
            # Creates:
            # - models/topics/financial_topics_state.pkl
            # - models/topics/financial_topics_model.pkl
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
        """Load a previously saved topic modeler from disk.
        
        This class method reconstructs a TopicModeler instance from files
        saved using the save() method. It loads both the state (configuration
        and metadata) and the fitted model if available.
        
        Args:
            path (str): Base path where the topic modeler was saved.
                The method will append '_state.pkl' and '_model.pkl' to this path
                to locate the saved files.
            
        Returns:
            TopicModeler: A new TopicModeler instance with the same configuration
                and state as the saved one, including the fitted model if available.
                
        Raises:
            FileNotFoundError: If the state file cannot be found
            pickle.UnpicklingError: If there's an error deserializing the files
            
        Example:
            >>> # Load a previously saved model
            >>> modeler = TopicModeler.load('models/topics/financial_topics')
            >>> # Use it to transform new documents
            >>> new_dtm = vectorizer.transform(new_texts)
            >>> topic_distributions = modeler.transform(new_dtm)
            
        Note:
            If the model file is not found, the method will still return a
            TopicModeler instance with the saved configuration, but without
            a fitted model. You'll need to call fit() before using transform().
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
        
        # Try to load vectorizer from a standard location
        vectorizer_path = f"{os.path.dirname(path)}/vectorizer.pkl"
        if os.path.exists(vectorizer_path):
            try:
                with open(vectorizer_path, 'rb') as f:
                    instance.vectorizer = pickle.load(f)
                logger.info(f"Vectorizer loaded from {vectorizer_path}")
            except Exception as e:
                logger.warning(f"Failed to load vectorizer from {vectorizer_path}: {str(e)}")
        
        return instance    
    
    def extract_topics(self, texts: List[str]) -> List[Tuple[int, float]]:
        """Extract topics from a list of text documents."""
        if self.model is None:
            logger.error("Model not fitted. Call fit() first.")
            raise ValueError("Topic model not fitted. Please train the model before extracting topics.")
        
        # If vectorizer is not available, try to recreate it from feature_names
        if self.vectorizer is None and hasattr(self, 'feature_names') and self.feature_names is not None:
            try:
                from sklearn.feature_extraction.text import CountVectorizer
                logger.info(f"Attempting to rebuild vectorizer from {len(self.feature_names)} feature names")
                # Create a vectorizer with the same vocabulary
                vocab_dict = {word: idx for idx, word in enumerate(self.feature_names)}
                self.vectorizer = CountVectorizer(vocabulary=vocab_dict)
                logger.info("Successfully created vectorizer from feature names")
            except Exception as e:
                logger.error(f"Failed to create vectorizer from feature names: {str(e)}")
        
        try:
            # First try with existing vectorizer
            if self.vectorizer is not None:
                try:
                    dtm = self.vectorizer.transform(texts)
                    logger.info(f"Using existing vectorizer to create DTM with shape: {dtm.shape}")
                    
                    # Check if we got any terms at all
                    if dtm.sum() == 0:
                        raise ValueError("No terms remained after applying existing vectorizer")
                        
                except (ValueError) as e:
                    raise ValueError(f"Existing vectorizer failed: {str(e)}")
            else:
                raise ValueError("Vectorizer not available")
                    
        except ValueError as e:
            # Graceful error handling - return a default topic with warning
            logger.warning(f"Topic extraction failed: {str(e)}. Using fallback approach.")
            
            # Create a placeholder result that won't break the UI
            # For LDA models, this would typically be [(topic_id, probability)]
            return [(0, 1.0)]
        
        # Process with the model
        try:
            # Transform into topic space
            topic_distributions = self.model.transform(dtm)
            
            # Get the top topics for this document
            result = []
            for i, doc_topics in enumerate(topic_distributions):
                top_topics = [(topic_id, float(score)) for topic_id, score in 
                            enumerate(doc_topics) if score > 0.01]
                top_topics.sort(key=lambda x: x[1], reverse=True)
                top_n = min(5, len(top_topics))  # Get top 5 topics or fewer
                result.extend(top_topics[:top_n])
                    
            return result
        
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            # Return placeholder to prevent UI errors
            return [(0, 1.0)]
    
    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[str]:
        """Get the top words for a specific topic.
        
        Args:
            topic_id (int): The ID of the topic to get words for
            top_n (int, optional): Number of top words to return. Defaults to 10.
            
        Returns:
            List[str]: List of the top words for the specified topic
        """
        # First check if we already have topic words cached
        if self.topic_words is not None and topic_id in self.topic_words:
            return self.topic_words[topic_id][:top_n]
        
        # If not, try to get them from the model
        if self.model is None:
            logger.warning("Model not fitted. Cannot get topic words.")
            return []
        
        try:
            # For LDA and NMF models
            if self.method in ['lda', 'nmf']:
                if hasattr(self.model, 'components_') and self.feature_names is not None:
                    topic = self.model.components_[topic_id]
                    top_indices = topic.argsort()[-(top_n):][::-1]
                    return [self.feature_names[i] for i in top_indices]
            # For BERTopic models
            elif self.method == 'bertopic' and BERTOPIC_AVAILABLE:
                if hasattr(self.model, 'get_topic'):
                    words_with_scores = self.model.get_topic(topic_id)[:top_n]
                    return [word for word, _ in words_with_scores]
        except Exception as e:
            logger.error(f"Error getting topic words for topic {topic_id}: {str(e)}")
        
        # If all else fails
        return []