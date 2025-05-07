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
    
    def fit(self, texts: List[str], optimize_topics: bool = False, 
           topic_range: List[int] = None) -> 'TopicModeler':
        """
        Fit topic model to the texts.
        
        Args:
            texts: List of text documents
            optimize_topics: Whether to optimize number of topics (LDA only)
            topic_range: Range of topics to consider for optimization
            
        Returns:
            Self for method chaining
        """
        if not texts:
            logger.warning("Empty text list provided to fit()")
            return self
        
        logger.info(f"Fitting {self.method} topic model on {len(texts)} texts")
        
        # Preprocess texts
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
    
    def transform(self, texts: List[str]) -> List[List[Tuple[int, float]]]:
        """
        Transform texts to topic distributions.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of topic distributions for each document
        """
        if not texts:
            logger.warning("Empty text list provided to transform()")
            return []
            
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Transforming {len(texts)} texts to topic distributions")
        
        if self.method == 'lda':
            # Preprocess texts
            tokenized_texts = self.preprocess(texts)
            
            # Convert to bag-of-words format
            corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
            
            # Get topic distributions
            topic_distributions = [self.model[doc] for doc in corpus]
            
            return topic_distributions
            
        elif self.method == 'bertopic':
            # For BERTopic, we need raw texts
            topics, probs = self.model.transform(texts)
            
            # Convert to same format as LDA
            topic_distributions = []
            for i, topic in enumerate(topics):
                if topic == -1:  # Outlier topic
                    topic_distributions.append([])
                else:
                    # BERTopic gives a single topic with probability, convert to LDA-like format
                    topic_distributions.append([(topic, probs[i][topic])])
            
            return topic_distributions
    
    def fit_transform(self, texts: List[str], optimize_topics: bool = False,
                     topic_range: List[int] = None) -> List[List[Tuple[int, float]]]:
        """
        Fit model and transform texts in one step.
        
        Args:
            texts: List of text documents
            optimize_topics: Whether to optimize number of topics
            topic_range: Range of topics to consider for optimization
            
        Returns:
            List of topic distributions for each document
        """
        self.fit(texts, optimize_topics, topic_range)
        return self.transform(texts)
    
    def get_topic_words(self, topic_id: int, num_words: int = 10) -> List[Tuple[str, float]]:
        """
        Get words for a specific topic.
        
        Args:
            topic_id: Topic ID
            num_words: Number of words to return
            
        Returns:
            List of (word, probability) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if self.method == 'lda':
            return self.model.show_topic(topic_id, num_words)
        elif self.method == 'bertopic':
            if topic_id == -1:  # Outlier topic
                return [("outlier", 1.0)]
            return self.model.get_topic(topic_id)[:num_words]
    
    def get_topic_wordcloud(self, topic_id: int, width: int = 800, height: int = 400) -> str:
        """
        Generate a wordcloud for a specific topic.
        
        Args:
            topic_id: Topic ID
            width: Wordcloud width
            height: Wordcloud height
            
        Returns:
            Base64 encoded PNG wordcloud image
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Get topic words
        if self.method == 'lda':
            words = dict(self.model.show_topic(topic_id, 50))
        elif self.method == 'bertopic':
            if topic_id == -1:  # Outlier topic
                words = {"outlier": 1.0}
            else:
                words = dict(self.model.get_topic(topic_id)[:50])
        
        # Generate wordcloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            prefer_horizontal=0.9,
            colormap='viridis',
            random_state=self.random_state
        ).generate_from_frequencies(words)
        
        # Convert wordcloud to image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    
    def extract_topics(self, texts: List[str]) -> List[Tuple[int, float]]:
        """
        Extract dominant topics for each document.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of (topic_id, probability) for each document
        """
        if not texts:
            logger.warning("Empty text list provided to extract_topics()")
            return []
            
        # Transform texts to topic distributions
        topic_distributions = self.transform(texts)
        
        # Extract dominant topics
        dominant_topics = []
        for dist in topic_distributions:
            # Sort by probability and take first (most probable)
            if dist:
                sorted_dist = sorted(dist, key=lambda x: x[1], reverse=True)
                dominant_topics.append(sorted_dist[0])
            else:
                dominant_topics.append((-1, 0.0))  # No topic
        
        return dominant_topics
    
    def visualize_topics(self) -> str:
        """
        Generate interactive topic visualization.
        
        Returns:
            HTML for interactive visualization
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if self.method == 'bertopic':
            # BERTopic has built-in visualization
            try:
                # Use intertopic distance map
                html = self.model.visualize_topics()
                return html.to_html()
            except:
                logger.warning("Could not generate BERTopic visualization")
                return ""
        
        return ""  # Default, no visualization
    
    def save(self, path: str) -> None:
        """
        Save the topic model to disk.
        
        Args:
            path: Directory path to save model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            'method': self.method,
            'num_topics': self.num_topics,
            'random_state': self.random_state,
            'model_name': self.model_name,
            'coherence_score': self.coherence_score,
            'optimal_num_topics': self.optimal_num_topics
        }
        
        joblib.dump(config, os.path.join(path, 'topic_config.joblib'))
        
        # Save topic terms
        joblib.dump(self.topic_terms, os.path.join(path, 'topic_terms.joblib'))
        
        # Save model-specific components
        if self.method == 'lda':
            # Save LDA model
            if self.model is not None:
                self.model.save(os.path.join(path, 'lda_model'))
            
            # Save dictionary
            if self.dictionary is not None:
                self.dictionary.save(os.path.join(path, 'dictionary'))
            
        elif self.method == 'bertopic':
            # Save BERTopic model
            if self.model is not None:
                self.model.save(os.path.join(path, 'bertopic_model'))
        
        logger.info(f"Topic model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TopicModeler':
        """
        Load a topic model from disk.
        
        Args:
            path: Directory path to load model from
            
        Returns:
            Loaded TopicModeler instance
        """
        config = joblib.load(os.path.join(path, 'topic_config.joblib'))
        
        instance = cls(
            method=config['method'],
            num_topics=config['num_topics'],
            random_state=config['random_state'],
            model_name=config['model_name']
        )
        
        instance.coherence_score = config['coherence_score']
        instance.optimal_num_topics = config['optimal_num_topics']
        
        # Load topic terms
        if os.path.exists(os.path.join(path, 'topic_terms.joblib')):
            instance.topic_terms = joblib.load(os.path.join(path, 'topic_terms.joblib'))
        
        # Load method-specific components
        if instance.method == 'lda':
            # Check if gensim is available
            if not GENSIM_AVAILABLE:
                logger.warning("Gensim not available, cannot load LDA model")
                return instance
            
            # Load dictionary
            if os.path.exists(os.path.join(path, 'dictionary')):
                instance.dictionary = Dictionary.load(os.path.join(path, 'dictionary'))
                instance.id2word = instance.dictionary
            
            # Load LDA model
            if os.path.exists(os.path.join(path, 'lda_model')):
                instance.model = LdaModel.load(os.path.join(path, 'lda_model'))
            
        elif instance.method == 'bertopic':
            # Check if BERTopic is available
            if not BERTOPIC_AVAILABLE:
                logger.warning("BERTopic not available, cannot load BERTopic model")
                return instance
            
            # Load BERTopic model
            if os.path.exists(os.path.join(path, 'bertopic_model')):
                instance.model = BERTopic.load(os.path.join(path, 'bertopic_model'))
        
        logger.info(f"Topic model loaded from {path}")
        return instance