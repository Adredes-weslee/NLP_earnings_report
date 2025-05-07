"""
Topic modeling module for earnings report text analysis.
Implements advanced topic modeling with coherence optimization.
"""

import numpy as np
import pandas as pd
import os
import logging
import joblib
import re
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse

# Optional imports for advanced topic modeling
try:
    import gensim
    from gensim.models import LdaModel, CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Optional imports for BERTopic
try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

logger = logging.getLogger('topic_modeler')

class TopicModeler:
    """
    Class for topic modeling on financial text data.
    Supports LDA and BERTopic algorithms with coherence optimization.
    """
    
    def __init__(self, method: str = 'lda', num_topics: int = 10, 
                 random_state: int = 42, model_name: str = None):
        """
        Initialize the topic modeling class.
        
        Args:
            method: Topic modeling algorithm ('lda', 'bertopic')
            num_topics: Number of topics to extract (for LDA)
            random_state: Random state for reproducibility
            model_name: For BERTopic, optional pre-trained model name
        """
        self.method = method
        self.num_topics = num_topics
        self.random_state = random_state
        self.model_name = model_name
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.id2word = None
        self.vectorizer = None
        self.coherence_score = None
        self.optimal_num_topics = None
        self.topic_terms = {}
        
        logger.info(f"Initializing TopicModeler with method={method}, num_topics={num_topics}")
        
        # Check if required libraries are available
        if method == 'lda' and not GENSIM_AVAILABLE:
            logger.warning("LDA selected but gensim not available. "
                          "Install with: pip install gensim")
            raise ImportError("Gensim library not available")
        elif method == 'bertopic' and not BERTOPIC_AVAILABLE:
            logger.warning("BERTopic selected but required libraries not available. "
                          "Install with: pip install bertopic umap-learn hdbscan")
            raise ImportError("BERTopic library not available")
    
    def preprocess(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for topic modeling.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of tokenized documents
        """
        if not texts:
            return []
            
        logger.info(f"Preprocessing {len(texts)} texts for topic modeling")
        
        # Simple tokenization and cleaning
        tokenized_texts = []
        for text in texts:
            if not isinstance(text, str):
                tokenized_texts.append([])
                continue
                
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize (simple whitespace tokenization)
            tokens = re.findall(r'\b\w+\b', text)
            
            # Remove short tokens
            tokens = [t for t in tokens if len(t) > 2]
            
            tokenized_texts.append(tokens)
        
        return tokenized_texts
    
    def _prepare_lda_data(self, tokenized_texts: List[List[str]]):
        """Prepare data for LDA model."""
        # Create dictionary
        self.dictionary = Dictionary(tokenized_texts)
        
        # Filter extreme terms
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Store id2word mapping
        self.id2word = self.dictionary
        
        return self.corpus, self.dictionary
    
    def _prepare_bertopic_data(self, texts: List[str]):
        """Prepare data for BERTopic model."""
        # For BERTopic, we just need the raw texts
        # We use a custom vectorizer for better control
        self.vectorizer = CountVectorizer(
            stop_words='english',
            min_df=5,
            max_df=0.5
        )
        return texts
    
    def evaluate_coherence(self, tokenized_texts: List[List[str]], 
                          topic_range: List[int], coherence: str = 'c_v') -> Tuple[int, float, List[Tuple[int, float]]]:
        """
        Evaluate coherence scores for different numbers of topics.
        
        Args:
            tokenized_texts: List of tokenized documents
            topic_range: List of numbers of topics to evaluate
            coherence: Coherence metric to use
            
        Returns:
            Tuple of (optimal_num_topics, optimal_coherence_score, all_scores)
        """
        if not GENSIM_AVAILABLE:
            logger.warning("Gensim not available, cannot evaluate coherence")
            return self.num_topics, 0.0, []
            
        logger.info(f"Evaluating coherence for {len(topic_range)} different topic counts")
        
        # Prepare data
        if self.dictionary is None or self.corpus is None:
            self._prepare_lda_data(tokenized_texts)
        
        coherence_scores = []
        
        # Evaluate coherence for each number of topics
        for num_topics in topic_range:
            # Train LDA model
            lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.id2word,
                num_topics=num_topics,
                random_state=self.random_state,
                passes=10
            )
            
            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=tokenized_texts,
                dictionary=self.dictionary,
                coherence=coherence
            )
            
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append((num_topics, coherence_score))
            logger.info(f"Coherence for {num_topics} topics: {coherence_score}")
        
        # Find optimal number of topics
        optimal = max(coherence_scores, key=lambda x: x[1])
        optimal_num_topics, optimal_coherence_score = optimal
        
        logger.info(f"Optimal number of topics: {optimal_num_topics} with coherence {optimal_coherence_score}")
        
        return optimal_num_topics, optimal_coherence_score, coherence_scores
    
    def optimize_num_topics(self, dtm, feature_names, topic_range):
        """
        Find the optimal number of topics based on coherence scores.
        
        Args:
            dtm: Document-term matrix
            feature_names: Feature names (vocabulary)
            topic_range: Range of topic counts to evaluate
            
        Returns:
            Tuple of (coherence_values, perplexity_values, optimal_num_topics)
        """
        logger.info(f"Finding optimal number of topics in range {min(topic_range)}-{max(topic_range)}")
        
        # Convert dtm to gensim corpus if needed
        if scipy.sparse.issparse(dtm):
            # Convert sparse matrix to gensim corpus format
            corpus = gensim.matutils.Sparse2Corpus(dtm.T)
            
            # Create dictionary mapping
            id2word = {i: word for i, word in enumerate(feature_names)}
            dictionary = gensim.corpora.Dictionary.from_corpus(
                corpus, id2word=id2word)
        else:
            # Assume it's already in the right format
            corpus = dtm
            dictionary = feature_names
            
        coherence_values = []
        perplexity_values = []
        
        # Evaluate different numbers of topics
        for num_topics in topic_range:
            logger.info(f"Evaluating {num_topics} topics...")
            
            # Train model
            lda_model = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=self.random_state,
                passes=10
            )
            
            # Get perplexity
            perplexity = lda_model.log_perplexity(corpus)
            perplexity_values.append(perplexity)
            
            # Calculate coherence if possible
            try:
                if isinstance(dictionary, dict):
                    # Skip coherence calculation if dictionary isn't in right format
                    coherence_values.append(0)
                    continue
                    
                coherence_model = gensim.models.CoherenceModel(
                    model=lda_model,
                    corpus=corpus,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence = coherence_model.get_coherence()
                coherence_values.append(coherence)
            except Exception as e:
                logger.warning(f"Error calculating coherence for {num_topics} topics: {str(e)}")
                coherence_values.append(0)
        
        # Find optimal number
        if coherence_values:
            max_idx = coherence_values.index(max(coherence_values))
            optimal_num_topics = topic_range[max_idx]
        else:
            # Default to median of range if no coherence values
            optimal_num_topics = topic_range[len(topic_range) // 2]
            
        logger.info(f"Optimal number of topics: {optimal_num_topics}")
        
        return coherence_values, perplexity_values, optimal_num_topics
    
    def fit(self, texts, optimize_topics: bool = False, 
           topic_range: List[int] = None) -> 'TopicModeler':
        """
        Fit topic model to the texts.
        
        Args:
            texts: List of text documents or document-term matrix
            optimize_topics: Whether to optimize number of topics (LDA only)
            topic_range: Range of topics to consider for optimization
            
        Returns:
            Self for method chaining
        """
        # Check if texts is a sparse matrix or list
        if scipy.sparse.issparse(texts):
            logger.info("Received sparse matrix for topic modeling")
            # For sparse matrix input, we assume it's already a document-term matrix
            if not hasattr(texts, 'shape') or texts.shape[0] == 0:
                logger.warning("Empty sparse matrix provided to fit()")
                return self
        elif isinstance(texts, list) and len(texts) == 0:
            logger.warning("Empty text list provided to fit()")
            return self
        else:
            logger.info(f"Fitting {self.method} topic model on {len(texts)} texts")
        
        # Handle different input types
        if scipy.sparse.issparse(texts):
            # Sparse matrix input - this is already a document-term matrix
            dtm = texts
            tokenized_texts = None  # We don't have the original texts
            
            # We need feature names for LDA, if not provided, use index numbers
            if 'feature_names' in locals() and 'feature_names' is not None:
                feature_names = feature_names
            else:
                feature_names = [str(i) for i in range(dtm.shape[1])]
                
            # For LDA with pre-computed DTM
            self.corpus = gensim.matutils.Sparse2Corpus(dtm.T)
            self.id2word = {i: word for i, word in enumerate(feature_names)}
            self.dictionary = Dictionary.from_corpus(self.corpus, id2word=self.id2word)
            
            # Train LDA model
            self.model = LdaModel(
                corpus=self.corpus,
                id2word=self.id2word,
                num_topics=self.num_topics,
                random_state=self.random_state,
                passes=10
            )
            
            # Store topic terms
            for topic_id in range(self.num_topics):
                self.topic_terms[topic_id] = self.model.show_topic(topic_id)
                
        else:
            # Text list input - traditional processing
            tokenized_texts = self.preprocess(texts)
            
            # Optimize number of topics if requested (LDA only)
            if optimize_topics and self.method == 'lda':
                if topic_range is None:
                    topic_range = range(2, min(50, len(texts) // 10 + 1), 3)  # Reasonable range
                
                # Find optimal number of topics
                optimal_num_topics, coherence_score, _ = self.evaluate_coherence(
                    tokenized_texts, topic_range
                )
                
                self.num_topics = optimal_num_topics
                self.coherence_score = coherence_score
                self.optimal_num_topics = optimal_num_topics
                
                logger.info(f"Selected {optimal_num_topics} topics with coherence {coherence_score}")
            
            # Fit actual model
            if self.method == 'lda':
                # Prepare data
                if self.dictionary is None or self.corpus is None:
                    self._prepare_lda_data(tokenized_texts)
                
                # Train LDA model
                self.model = LdaModel(
                    corpus=self.corpus,
                    id2word=self.id2word,
                    num_topics=self.num_topics,
                    random_state=self.random_state,
                    passes=10
                )
                
                # Store topic terms
                for topic_id in range(self.num_topics):
                    self.topic_terms[topic_id] = self.model.show_topic(topic_id)
                
                # Calculate coherence if not already done
                if self.coherence_score is None:
                    coherence_model = CoherenceModel(
                        model=self.model,
                        texts=tokenized_texts,
                        dictionary=self.dictionary,
                        coherence='c_v'
                    )
                    self.coherence_score = coherence_model.get_coherence()
                
            elif self.method == 'bertopic':
                # Prepare data
                data = self._prepare_bertopic_data(texts)
                
                # Configure BERTopic
                umap_model = UMAP(
                    n_neighbors=15,
                    n_components=5,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=self.random_state
                )
                
                hdbscan_model = HDBSCAN(
                    min_cluster_size=10,
                    metric='euclidean',
                    prediction_data=True
                )
                
                # Initialize and train BERTopic model
                self.model = BERTopic(
                    vectorizer_model=self.vectorizer,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    language="english"
                )
                
                # Fit model
                topics, _ = self.model.fit_transform(data)
                
                # Store topic terms
                for topic_id in self.model.get_topic_info()['Topic'].values:
                    if topic_id != -1:  # Skip outlier topic
                        self.topic_terms[topic_id] = self.model.get_topic(topic_id)
        
        return self

    # ... rest of the code remains unchanged ...