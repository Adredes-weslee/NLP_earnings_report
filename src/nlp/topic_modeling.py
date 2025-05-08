"""
Topic modeling module for financial text analysis.
Provides LDA and transformer-based topic modeling for earnings reports.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.decomposition import LatentDirichletAllocation, NMF

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
    
    def __init__(self, method: str = 'lda', num_topics: int = 40, random_state: int = 42):
        """
        Initialize the topic modeler.
        
        Args:
            method: Topic modeling method ('lda', 'nmf', 'bertopic')
            num_topics: Number of topics to extract (not used for BERTopic)
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.num_topics = num_topics
        self.random_state = random_state
        self.model = None
        self.topic_words = None
        self.feature_names = None
        self.topic_word_distributions = None
        
        logger.info(f"Initializing TopicModeler with method={method}, num_topics={num_topics}")
        
        # Check if BERTopic is available when needed
        if method == 'bertopic' and not BERTOPIC_AVAILABLE:
            logger.warning("BERTopic method selected but library not available. "
                          "Install with: pip install bertopic")
            logger.warning("Falling back to LDA")
            self.method = 'lda'
    
    def _create_lda_model(self) -> LatentDirichletAllocation:
        """Create a Latent Dirichlet Allocation model."""
        return LatentDirichletAllocation(
            n_components=self.num_topics,
            max_iter=25,
            learning_method='online',
            learning_offset=50.,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _create_nmf_model(self) -> NMF:
        """Create a Non-Negative Matrix Factorization model."""
        return NMF(
            n_components=self.num_topics,
            random_state=self.random_state,
            max_iter=1000,
            alpha=.1,
            l1_ratio=.5
        )
    
    def _create_bertopic_model(self) -> Any:
        """Create a BERTopic model."""
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic is not available. Install with: pip install bertopic")
        
        return BERTopic(
            nr_topics="auto",
            language="english",
            calculate_probabilities=True,
            verbose=True
        )
    
    def fit(self, document_term_matrix: Any, feature_names: List[str] = None, 
            raw_documents: List[str] = None) -> 'TopicModeler':
        """
        Fit the topic model to the document-term matrix.
        
        Args:
            document_term_matrix: DTM or embeddings depending on method
            feature_names: List of feature/vocabulary names
            raw_documents: Raw text for BERTopic (needed only for BERTopic)
            
        Returns:
            Self for method chaining
        """
        self.feature_names = feature_names
        
        if self.method == 'lda':
            model = self._create_lda_model()
            self.model = model.fit(document_term_matrix)
            self.topic_word_distributions = self.model.components_
            
        elif self.method == 'nmf':
            model = self._create_nmf_model()
            self.model = model.fit(document_term_matrix)
            self.topic_word_distributions = self.model.components_
            
        elif self.method == 'bertopic':
            if raw_documents is None:
                raise ValueError("Raw documents are required for BERTopic model")
            
            model = self._create_bertopic_model()
            self.model = model.fit(raw_documents, document_term_matrix)
            
            # For compatibility with other methods
            if hasattr(self.model, 'get_topic_info'):
                topic_info = self.model.get_topic_info()
                self.num_topics = len(topic_info) - 1  # Excluding -1 topic (outliers)
        
        # Extract top words for each topic
        self._extract_topic_words()
        
        logger.info(f"Topic model fitted with {self.num_topics} topics")
        return self
    
    def _extract_topic_words(self, top_n: int = 20) -> None:
        """
        Extract the top words for each topic.
        
        Args:
            top_n: Number of top words to extract per topic
        """
        if self.model is None:
            logger.warning("Model not fitted yet")
            return
        
        self.topic_words = {}
        
        if self.method in ['lda', 'nmf']:
            if self.topic_word_distributions is None or self.feature_names is None:
                logger.warning("Topic word distributions or feature names not available")
                return
                
            for topic_idx, topic in enumerate(self.topic_word_distributions):
                # Sort words by their weight in the topic
                sorted_indices = topic.argsort()[:-top_n-1:-1]
                top_words = [self.feature_names[i] for i in sorted_indices]
                self.topic_words[topic_idx] = top_words
                
        elif self.method == 'bertopic':
            if hasattr(self.model, 'get_topics'):
                bert_topics = self.model.get_topics()
                
                for topic_id, word_scores in bert_topics.items():
                    if topic_id != -1:  # Skip outlier topic
                        # Get top words and their scores
                        top_words = [word for word, _ in word_scores[:top_n]]
                        self.topic_words[topic_id] = top_words
    
    def transform(self, document_term_matrix: Any) -> np.ndarray:
        """
        Transform documents to topic distributions.
        
        Args:
            document_term_matrix: Document-term matrix or embeddings
            
        Returns:
            Document-topic matrix (probabilities)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        if self.method in ['lda', 'nmf']:
            return self.model.transform(document_term_matrix)
            
        elif self.method == 'bertopic':
            if hasattr(self.model, 'transform'):
                topics, probs = self.model.transform(document_term_matrix)
                
                # Convert to document-topic matrix format
                doc_topic_matrix = np.zeros((len(topics), self.num_topics))
                for i, (doc_topics, doc_probs) in enumerate(zip(topics, probs)):
                    for topic, prob in zip(doc_topics, doc_probs):
                        if topic != -1 and topic < self.num_topics:  # Skip outlier topic
                            doc_topic_matrix[i, topic] = prob
                
                # Normalize rows to sum to 1
                row_sums = doc_topic_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                doc_topic_matrix = doc_topic_matrix / row_sums
                
                return doc_topic_matrix
            else:
                raise NotImplementedError("Transform not implemented for this BERTopic version")
    
    def get_top_words(self, topic_id: int = None, top_n: int = 10) -> Union[Dict[int, List[str]], List[str]]:
        """
        Get top words for a specific topic or all topics.
        
        Args:
            topic_id: Topic ID (if None, return all topics)
            top_n: Number of top words to return
            
        Returns:
            Dictionary of topic ID to list of top words, or list of top words for a specific topic
        """
        if self.topic_words is None:
            self._extract_topic_words(top_n=top_n)
        
        if topic_id is not None:
            if topic_id in self.topic_words:
                return self.topic_words[topic_id][:top_n]
            else:
                logger.warning(f"Topic ID {topic_id} not found")
                return []
        else:
            return {k: v[:top_n] for k, v in self.topic_words.items()}
    
    def get_topics(self, num_words: int = 10) -> Dict[int, List[Union[str, Tuple[str, float]]]]:
        """
        Get top words for all topics - alias for get_top_words for API compatibility.
        
        Args:
            num_words: Number of top words to return per topic
            
        Returns:
            Dictionary of topic ID to list of top words
        """
        return self.get_top_words(topic_id=None, top_n=num_words)
    
    def get_document_topics(self, document_term_matrix) -> np.ndarray:
        """
        Get document-topic distributions for a set of documents.
        
        Args:
            document_term_matrix: Document-term matrix to transform
            
        Returns:
            numpy.ndarray: Document-topic matrix where each row represents 
                        a document and each column represents a topic
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.method == 'lda':
            # LDA transform returns document-topic matrix directly
            doc_topic_matrix = self.model.transform(document_term_matrix)
            return doc_topic_matrix
        
        elif self.method == 'nmf':
            # NMF transform also returns document-topic matrix
            doc_topic_matrix = self.model.transform(document_term_matrix)
            return doc_topic_matrix
        
        elif self.method == 'bertopic' and BERTOPIC_AVAILABLE:
            if hasattr(self.model, 'transform'):
                # Get document-topic distributions
                topics, probs = self.model.transform(document_term_matrix)
                
                # Convert to a proper document-topic matrix
                num_docs = len(topics)
                num_topics = self.num_topics
                doc_topic_matrix = np.zeros((num_docs, num_topics))
                
                # Fill the matrix with topic probabilities
                for i, (doc_topics, doc_probs) in enumerate(zip(topics, probs)):
                    for topic, prob in zip(doc_topics, doc_probs):
                        if topic >= 0 and topic < num_topics:  # Skip noise topic (-1)
                            doc_topic_matrix[i, topic] = prob
                
                return doc_topic_matrix
            else:
                raise NotImplementedError("BERTopic transform not available")
        
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def visualize_topics(self, top_n: int = 10) -> plt.Figure:
        """
        Visualize the top words for each topic.
        
        Args:
            top_n: Number of top words to visualize per topic
            
        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        if self.method == 'bertopic' and hasattr(self.model, 'visualize_topics'):
            try:
                return self.model.visualize_topics()
            except Exception as e:
                logger.warning(f"BERTopic visualization failed: {str(e)}")
                # Fall back to basic visualization
        
        # Basic visualization for LDA/NMF and fallback for BERTopic
        num_topics_to_plot = min(8, self.num_topics)  # Plot at most 8 topics
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, topic_id in enumerate(list(self.topic_words.keys())[:num_topics_to_plot]):
            top_words = self.get_top_words(topic_id, top_n=top_n)
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(top_words))
            
            if self.method in ['lda', 'nmf'] and self.topic_word_distributions is not None:
                # Get weights for the top words
                topic_idx = topic_id
                word_indices = [self.feature_names.index(word) for word in top_words]
                weights = self.topic_word_distributions[topic_idx, word_indices]
                
                # Plot weights in descending order
                sorted_indices = weights.argsort()[::-1]
                weights = weights[sorted_indices]
                top_words = [top_words[j] for j in sorted_indices]
                
                axes[i].barh(y_pos, weights)
            else:
                # Plot without weights
                axes[i].barh(y_pos, np.ones(len(top_words)))
            
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(top_words)
            axes[i].invert_yaxis()
            axes[i].set_title(f'Topic {topic_id}')
        
        plt.tight_layout()
        return fig
    
    def compute_coherence_score(self, texts: List[str] = None) -> float:
        """
        Compute coherence score for the topic model.
        
        Args:
            texts: List of tokenized texts (only needed for some methods)
            
        Returns:
            Coherence score
        """
        try:
            from gensim.models.coherencemodel import CoherenceModel
            import gensim.corpora as corpora
        except ImportError:
            logger.warning("Gensim not available for coherence calculation")
            return -1
        
        if texts is None:
            logger.warning("Texts are required for coherence calculation")
            return -1
        
        try:
            # Preprocess texts to list of lists of tokens
            tokenized_texts = [text.split() if isinstance(text, str) else text for text in texts]
            
            # Create gensim dictionary
            dictionary = corpora.Dictionary(tokenized_texts)
            
            # Extract topic words as lists
            topics = []
            for topic_id in range(self.num_topics):
                if topic_id in self.topic_words:
                    topics.append(self.topic_words[topic_id])
            
            # Create coherence model
            coherence_model = CoherenceModel(
                topics=topics, 
                texts=tokenized_texts, 
                dictionary=dictionary, 
                coherence='c_v'
            )
            
            # Compute coherence
            coherence = coherence_model.get_coherence()
            logger.info(f"Topic coherence score: {coherence:.4f}")
            return coherence
            
        except Exception as e:
            logger.error(f"Error computing coherence: {str(e)}")
            return -1
    
    def save(self, path: str) -> None:
        """
        Save the topic model to disk.
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model configuration
        config = {
            'method': self.method,
            'num_topics': self.num_topics,
            'random_state': self.random_state,
        }
        
        # Add feature_names to config, converting to list if it's a numpy array
        if hasattr(self, 'feature_names'):
            if isinstance(self.feature_names, np.ndarray):
                config['feature_names'] = self.feature_names.tolist()
            else:
                config['feature_names'] = self.feature_names
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save topic words
        if hasattr(self, 'topic_words') and self.topic_words is not None:
            with open(os.path.join(path, 'topic_words.json'), 'w') as f:
                serializable_topics = {str(k): v for k, v in self.topic_words.items()}
                json.dump(serializable_topics, f)
        
        # Save the model
        if self.method in ['lda', 'nmf']:
            joblib.dump(self.model, os.path.join(path, f'{self.method}_model.pkl'))
        elif self.method == 'bertopic':
            if hasattr(self.model, 'save'):
                self.model.save(os.path.join(path, 'bertopic_model'))
        
        logger.info(f"Topic model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TopicModeler':
        """
        Load a topic model from disk.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            Loaded TopicModeler instance
        """
        import json
        
        # Load model configuration
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create an instance with the loaded configuration
        instance = cls(
            method=config['method'],
            num_topics=config['num_topics'],
            random_state=config['random_state']
        )
        
        instance.feature_names = config['feature_names']
        
        # Load topic words
        try:
            with open(os.path.join(path, 'topic_words.json'), 'r') as f:
                topic_words = json.load(f)
                # Convert keys back to integers
                instance.topic_words = {int(k): v for k, v in topic_words.items()}
        except FileNotFoundError:
            instance.topic_words = None
        
        # Load the model
        if config['method'] in ['lda', 'nmf']:
            instance.model = joblib.load(os.path.join(path, f"{config['method']}_model.pkl"))
            if hasattr(instance.model, 'components_'):
                instance.topic_word_distributions = instance.model.components_
        elif config['method'] == 'bertopic':
            if BERTOPIC_AVAILABLE and os.path.exists(os.path.join(path, 'bertopic_model')):
                from bertopic import BERTopic
                instance.model = BERTopic.load(os.path.join(path, 'bertopic_model'))
        
        logger.info(f"Topic model loaded from {path}")
        return instance
    
    def optimize_num_topics(self, document_term_matrix: Any, feature_names: List[str], 
                            topic_range: range = range(2, 51, 4), 
                            raw_documents: List[str] = None) -> Tuple[List[float], List[float], int]:
        """
        Find the optimal number of topics by evaluating coherence and perplexity.
        
        Args:
            document_term_matrix: Document-term matrix for model fitting
            feature_names: List of feature/vocabulary names
            topic_range: Range of number of topics to try
            raw_documents: Raw text documents for coherence calculation
            
        Returns:
            coherence_values: List of coherence scores for each topic count
            perplexity_values: List of perplexity scores for each topic count
            optimal_num_topics: Optimal number of topics
        """
        logger.info(f"Optimizing number of topics in range {min(topic_range)}-{max(topic_range)}...")
        coherence_values = []
        perplexity_values = []
        
        original_num_topics = self.num_topics
        original_method = self.method
        
        # Only support LDA and NMF optimization for now
        if self.method not in ['lda', 'nmf']:
            logger.warning(f"Topic optimization not supported for method {self.method}. Using default.")
            return [], [], self.num_topics
        
        try:
            for num_topics in topic_range:
                logger.info(f"Evaluating model with {num_topics} topics...")
                
                # Update model parameters
                self.num_topics = num_topics
                
                if self.method == 'lda':
                    model = self._create_lda_model()
                    model.fit(document_term_matrix)
                    perplexity = model.perplexity(document_term_matrix)
                    perplexity_values.append(perplexity)
                    
                    # Store components for coherence calculation
                    self.model = model
                    self.topic_word_distributions = model.components_
                else:  # NMF
                    model = self._create_nmf_model()
                    model.fit(document_term_matrix)
                    
                    # NMF doesn't have perplexity, use reconstruction error instead
                    error = model.reconstruction_err_
                    perplexity_values.append(error)
                    
                    # Store components for coherence calculation
                    self.model = model
                    self.topic_word_distributions = model.components_
                
                # Extract topic words
                self.feature_names = feature_names
                self._extract_topic_words()
                
                # Compute coherence if raw documents are provided
                if raw_documents is not None:
                    coherence = self.compute_coherence_score(raw_documents)
                    coherence_values.append(coherence)
                
                logger.info(f"Topics: {num_topics}, " + 
                          f"{'Perplexity' if self.method == 'lda' else 'Error'}: {perplexity_values[-1]:.2f}" + 
                          (f", Coherence: {coherence_values[-1]:.4f}" if raw_documents else ""))
            
            # Determine optimal number of topics
            if raw_documents and coherence_values:
                # Coherence is more important if we have it - higher is better
                optimal_idx = coherence_values.index(max(coherence_values))
                optimal_num_topics = topic_range[optimal_idx]
            else:
                # Lower perplexity/error is better
                optimal_idx = perplexity_values.index(min(perplexity_values))
                optimal_num_topics = topic_range[optimal_idx]
            
            # Set the model to use the optimal number of topics
            self.num_topics = optimal_num_topics
            
            # Fit with optimal number
            if self.method == 'lda':
                self.model = self._create_lda_model().fit(document_term_matrix)
                self.topic_word_distributions = self.model.components_
            else:  # NMF
                self.model = self._create_nmf_model().fit(document_term_matrix)
                self.topic_word_distributions = self.model.components_
            
            # Update topic words with optimal model
            self._extract_topic_words()
            
            logger.info(f"Optimal number of topics: {optimal_num_topics}")
            return coherence_values, perplexity_values, optimal_num_topics
            
        except Exception as e:
            logger.error(f"Error during topic optimization: {str(e)}")
            
            # Restore original settings
            self.num_topics = original_num_topics
            self.method = original_method
            
            # Fit with original settings to have a valid model
            if self.method == 'lda':
                self.model = self._create_lda_model().fit(document_term_matrix)
                self.topic_word_distributions = self.model.components_
            else:
                self.model = self._create_nmf_model().fit(document_term_matrix)
                self.topic_word_distributions = self.model.components_
                
            self.feature_names = feature_names
            self._extract_topic_words()
            
            return [], [], self.num_topics