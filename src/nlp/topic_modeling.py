"""
Topic modeling module for financial text analysis.
Implements enhanced topic modeling with coherence optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional, Tuple
import os
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports for topic modeling
try:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    SKLEARN_MODELS_AVAILABLE = True
except ImportError:
    SKLEARN_MODELS_AVAILABLE = False

try:
    import gensim
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False

try:
    from tmtoolkit.topicmod import evaluate, visualize
    TMTOOLKIT_AVAILABLE = True
except ImportError:
    TMTOOLKIT_AVAILABLE = False

logger = logging.getLogger('topic_modeler')

class TopicModeler:
    """
    Enhanced topic modeling for financial text analysis.
    Supports LDA and NMF with coherence optimization.
    """
    
    def __init__(self, method: str = 'lda', num_topics: int = 10, random_state: int = 42):
        """
        Initialize the topic modeler.
        
        Args:
            method: Topic modeling method ('lda', 'nmf', 'gensim_lda')
            num_topics: Number of topics to extract
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.num_topics = num_topics
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.dtm = None
        self.corpus = None
        self.dictionary = None
        self.coherence_values = []
        self.perplexity_values = []
        self.topic_terms = {}
        self.topic_sentiment = {}  # For tracking sentiment associated with each topic
        
        # Check if required libraries are available
        if method in ['lda', 'nmf'] and not SKLEARN_MODELS_AVAILABLE:
            raise ImportError("scikit-learn is required for LDA/NMF topic modeling")
        
        if method == 'gensim_lda' and not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for gensim_lda topic modeling")
            
        logger.info(f"Initializing TopicModeler with method={method}, num_topics={num_topics}")
    
    def fit(self, dtm, feature_names=None):
        """
        Fit the topic model on a document-term matrix.
        
        Args:
            dtm: Document-term matrix (scipy sparse matrix or numpy array)
            feature_names: List of feature names (terms)
            
        Returns:
            Self for method chaining
        """
        self.dtm = dtm
        self.feature_names = feature_names
        
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.num_topics,
                max_iter=25,
                learning_method='online',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.method == 'nmf':
            self.model = NMF(
                n_components=self.num_topics,
                random_state=self.random_state,
                max_iter=1000
            )
        elif self.method == 'gensim_lda':
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim is required for gensim_lda topic modeling")
            
            # Convert to gensim corpus format if needed
            if self.corpus is None or self.dictionary is None:
                raise ValueError("For gensim_lda, use fit_transform_tokens() instead of fit()")
        
        if self.method in ['lda', 'nmf']:
            self.model.fit(dtm)
            self._extract_topics_from_sklearn()
            
        return self
    
    def fit_transform_tokens(self, texts_tokens: List[List[str]], min_df: int = 2, max_df: float = 0.95):
        """
        Fit the topic model using tokenized texts.
        
        Args:
            texts_tokens: List of tokenized texts (list of token lists)
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            
        Returns:
            Document-topic matrix
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for this functionality")
        
        # Create dictionary and corpus
        self.dictionary = Dictionary(texts_tokens)
        
        # Filter extremes (words that appear in too few or too many documents)
        self.dictionary.filter_extremes(no_below=min_df, no_above=max_df)
        
        # Convert tokenized texts to bag-of-words format
        self.corpus = [self.dictionary.doc2bow(text) for text in texts_tokens]
        
        if self.method == 'gensim_lda':
            # Build LDA model
            self.model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                random_state=self.random_state,
                passes=10,
                alpha='auto',
                eta='auto',
                minimum_probability=0.0
            )
            
            self._extract_topics_from_gensim()
        
        # Get document-topic matrix
        doc_topics = self.get_document_topics()
        
        return doc_topics
    
    def _extract_topics_from_sklearn(self):
        """Extract topics from scikit-learn models."""
        if self.feature_names is None:
            logger.warning("No feature names provided, using indices as feature names")
            self.feature_names = [str(i) for i in range(self.dtm.shape[1])]
        
        # Get topic-term distributions
        topic_term_dist = self.model.components_
        
        # For each topic, get the top terms
        for topic_idx, topic in enumerate(topic_term_dist):
            # Sort terms by importance in topic
            sorted_terms_idx = topic.argsort()[::-1]
            top_terms_idx = sorted_terms_idx[:50]  # Get indices of top 50 terms
            top_terms = [self.feature_names[i] for i in top_terms_idx]
            top_weights = [topic[i] for i in top_terms_idx]
            
            # Store topic terms with weights
            self.topic_terms[topic_idx] = list(zip(top_terms, top_weights))
    
    def _extract_topics_from_gensim(self):
        """Extract topics from gensim models."""
        # Get topic-term distributions
        for topic_idx in range(self.num_topics):
            # Get top terms for topic (returns list of (term_id, weight))
            topic_terms = self.model.get_topic_terms(topic_idx, topn=50)
            
            # Convert term IDs to actual terms
            topic_terms = [(self.dictionary[term_id], weight) for term_id, weight in topic_terms]
            
            # Store topic terms with weights
            self.topic_terms[topic_idx] = topic_terms
    
    def get_topics(self, num_words: int = 15) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top terms for each topic.
        
        Args:
            num_words: Number of top words to include for each topic
            
        Returns:
            Dictionary mapping topic IDs to lists of (term, weight) tuples
        """
        return {topic_id: terms[:num_words] for topic_id, terms in self.topic_terms.items()}
    
    def get_document_topics(self, dtm=None) -> np.ndarray:
        """
        Get topic distributions for documents.
        
        Args:
            dtm: Document-term matrix (if None, use the one from fit)
            
        Returns:
            Document-topic matrix
        """
        if dtm is None:
            dtm = self.dtm
        
        if self.method in ['lda', 'nmf']:
            return self.model.transform(dtm)
        elif self.method == 'gensim_lda':
            # For gensim, we need to convert to a numpy array
            doc_topics = np.zeros((len(self.corpus), self.num_topics))
            
            for i, bow in enumerate(self.corpus):
                topic_dist = self.model.get_document_topics(bow, minimum_probability=0)
                for topic_id, prob in topic_dist:
                    doc_topics[i, topic_id] = prob
            
            return doc_topics
    
    def get_dominant_topics(self, dtm=None) -> pd.DataFrame:
        """
        Get dominant topic for each document.
        
        Args:
            dtm: Document-term matrix (if None, use the one from fit)
            
        Returns:
            DataFrame with document index, dominant topic, topic proportion, and top terms
        """
        # Get document-topic matrix
        doc_topics = self.get_document_topics(dtm)
        
        # Get dominant topic for each document
        dominant_topics = []
        for i, doc_topic in enumerate(doc_topics):
            topic_idx = doc_topic.argmax()
            topic_prop = doc_topic[topic_idx]
            top_terms = [term for term, _ in self.get_topics()[topic_idx]][:5]
            
            dominant_topics.append({
                'document_id': i,
                'dominant_topic': int(topic_idx),
                'topic_proportion': float(topic_prop),
                'top_terms': ', '.join(top_terms)
            })
        
        return pd.DataFrame(dominant_topics)
    
    def optimize_num_topics(self, dtm, feature_names=None, topic_range=range(5, 50, 5)):
        """
        Find optimal number of topics based on coherence and perplexity.
        
        Args:
            dtm: Document-term matrix
            feature_names: List of feature names
            topic_range: Range of topic numbers to try
            
        Returns:
            Tuple of (coherence_values, perplexity_values, optimal_num_topics)
        """
        coherence_values = []
        perplexity_values = []
        
        for num_topics in topic_range:
            logger.info(f"Trying model with {num_topics} topics...")
            
            # Set number of topics
            self.num_topics = num_topics
            
            # Fit model
            if self.method in ['lda', 'nmf']:
                self.fit(dtm, feature_names)
                
                # Calculate perplexity for LDA
                if self.method == 'lda':
                    perplexity = self.model.perplexity(dtm)
                    perplexity_values.append(perplexity)
                else:
                    perplexity_values.append(0)  # NMF doesn't have perplexity
                
                # Calculate topic coherence
                if TMTOOLKIT_AVAILABLE:
                    topic_word_matrix = self.model.components_
                    vocab = feature_names
                    try:
                        coherence = evaluate.metric_coherence_gensim(
                            topic_word_matrix=topic_word_matrix,
                            vocab=vocab,
                            dtm=dtm,
                            measure='c_v'
                        )
                        avg_coherence = np.mean(coherence)
                        coherence_values.append(avg_coherence)
                    except Exception as e:
                        logger.warning(f"Error calculating coherence: {str(e)}")
                        coherence_values.append(0)
                else:
                    coherence_values.append(0)
                    logger.warning("tmtoolkit not available, skipping coherence calculation")
                    
            elif self.method == 'gensim_lda' and GENSIM_AVAILABLE:
                # For gensim, use the existing corpus and dictionary
                if self.corpus is None or self.dictionary is None:
                    raise ValueError("For gensim_lda, use optimize_num_topics_gensim() instead")
            
        # Store results
        self.coherence_values = coherence_values
        self.perplexity_values = perplexity_values
        
        # Find optimal number of topics based on coherence
        if coherence_values:
            optimal_idx = np.argmax(coherence_values)
            optimal_num_topics = topic_range[optimal_idx]
        else:
            # If no coherence values, use perplexity
            if perplexity_values:
                # Lower perplexity is better
                optimal_idx = np.argmin(perplexity_values)
                optimal_num_topics = topic_range[optimal_idx]
            else:
                # If no metrics available, use middle of range
                optimal_num_topics = topic_range[len(topic_range) // 2]
        
        logger.info(f"Optimal number of topics: {optimal_num_topics}")
        
        # Reset to optimal number of topics and refit
        self.num_topics = optimal_num_topics
        self.fit(dtm, feature_names)
        
        return coherence_values, perplexity_values, optimal_num_topics
    
    def optimize_num_topics_gensim(self, texts_tokens, topic_range=range(5, 50, 5)):
        """
        Find optimal number of topics for gensim models based on coherence.
        
        Args:
            texts_tokens: List of tokenized texts
            topic_range: Range of topic numbers to try
            
        Returns:
            Tuple of (coherence_values, optimal_num_topics)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for this functionality")
        
        # Create dictionary and corpus if not already done
        if self.dictionary is None:
            self.dictionary = Dictionary(texts_tokens)
            self.dictionary.filter_extremes(no_below=2, no_above=0.95)
        
        if self.corpus is None:
            self.corpus = [self.dictionary.doc2bow(text) for text in texts_tokens]
        
        coherence_values = []
        
        for num_topics in topic_range:
            logger.info(f"Trying model with {num_topics} topics...")
            
            # Train LDA model
            lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=self.random_state,
                passes=10,
                alpha='auto',
                eta='auto'
            )
            
            # Calculate coherence
            try:
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=texts_tokens,
                    dictionary=self.dictionary,
                    coherence='c_v'
                )
                coherence = coherence_model.get_coherence()
                coherence_values.append(coherence)
                logger.info(f"Num topics = {num_topics}, Coherence = {coherence:.4f}")
            except Exception as e:
                logger.warning(f"Error calculating coherence: {str(e)}")
                coherence_values.append(0)
        
        # Store results
        self.coherence_values = coherence_values
        
        # Find optimal number of topics
        if coherence_values:
            optimal_idx = np.argmax(coherence_values)
            optimal_num_topics = topic_range[optimal_idx]
            logger.info(f"Optimal number of topics: {optimal_num_topics}")
        else:
            optimal_num_topics = topic_range[len(topic_range) // 2]
            logger.warning(f"No valid coherence values, defaulting to {optimal_num_topics} topics")
        
        # Reset to optimal number of topics and refit
        self.num_topics = optimal_num_topics
        
        # Train final model with optimal number of topics
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=optimal_num_topics,
            random_state=self.random_state,
            passes=10,
            alpha='auto',
            eta='auto'
        )
        
        # Extract topics
        self._extract_topics_from_gensim()
        
        return coherence_values, optimal_num_topics
    
    def plot_topic_coherence(self, topic_range=None):
        """
        Plot coherence values for different numbers of topics.
        
        Args:
            topic_range: Range of topic numbers tried
        """
        if not self.coherence_values:
            logger.warning("No coherence values to plot. Run optimize_num_topics() first.")
            return
            
        if topic_range is None:
            # Infer topic range from number of coherence values
            topic_range = range(5, 5 * (len(self.coherence_values) + 1), 5)
        
        plt.figure(figsize=(10, 6))
        plt.plot(topic_range, self.coherence_values, 'o-')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Topic Coherence by Number of Topics')
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_topic_wordcloud(self, topic_id, top_n=30):
        """
        Plot wordcloud for a specific topic.
        
        Args:
            topic_id: ID of the topic to visualize
            top_n: Number of top terms to include
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.error("wordcloud package not available. Install with: pip install wordcloud")
            return None
        
        if topic_id not in self.topic_terms:
            logger.error(f"Topic {topic_id} not found")
            return None
        
        # Get top terms and their weights
        terms, weights = zip(*self.topic_terms[topic_id][:top_n])
        
        # Create a dictionary of word: weight
        word_weights = dict(zip(terms, weights))
        
        # Create and configure the wordcloud object
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=top_n,
            prefer_horizontal=0.9,
            collocations=False
        ).generate_from_frequencies(word_weights)
        
        # Display the wordcloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id} Word Cloud')
        
        return plt.gcf()
    
    def visualize_topics_pyldavis(self, output_file=None):
        """
        Create an interactive pyLDAvis visualization of the topics.
        
        Args:
            output_file: Path to save the visualization (if None, return the HTML)
            
        Returns:
            HTML visualization if output_file is None, otherwise None
        """
        if not PYLDAVIS_AVAILABLE or not GENSIM_AVAILABLE:
            logger.error("pyLDAvis and/or gensim not available")
            return None
        
        if self.method != 'gensim_lda':
            logger.error("pyLDAvis visualization only available for gensim LDA models")
            return None
        
        if self.model is None or self.corpus is None or self.dictionary is None:
            logger.error("Model, corpus, or dictionary not available")
            return None
        
        try:
            # Create the visualization
            logger.info("Creating pyLDAvis visualization...")
            vis = gensimvis.prepare(
                self.model, self.corpus, self.dictionary,
                mds='tsne', sort_topics=False
            )
            
            # Save or return
            if output_file:
                pyLDAvis.save_html(vis, output_file)
                logger.info(f"Visualization saved to {output_file}")
                return None
            else:
                return pyLDAvis.prepared_data_to_html(vis)
        except Exception as e:
            logger.error(f"Error creating pyLDAvis visualization: {str(e)}")
            return None
    
    def save(self, path: str) -> None:
        """
        Save the topic model and related data.
        
        Args:
            path: Directory path to save model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            'method': self.method,
            'num_topics': self.num_topics,
            'random_state': self.random_state,
            'coherence_values': self.coherence_values,
            'perplexity_values': self.perplexity_values
        }
        
        joblib.dump(config, os.path.join(path, 'topic_model_config.joblib'))
        
        # Save model-specific components
        if self.method in ['lda', 'nmf']:
            joblib.dump(self.model, os.path.join(path, 'topic_model.joblib'))
            if self.feature_names is not None:
                joblib.dump(self.feature_names, os.path.join(path, 'feature_names.joblib'))
        elif self.method == 'gensim_lda':
            self.model.save(os.path.join(path, 'gensim_lda_model'))
            if self.dictionary is not None:
                self.dictionary.save(os.path.join(path, 'gensim_dictionary'))
        
        # Save topic terms
        joblib.dump(self.topic_terms, os.path.join(path, 'topic_terms.joblib'))
        
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
        config = joblib.load(os.path.join(path, 'topic_model_config.joblib'))
        
        instance = cls(
            method=config['method'],
            num_topics=config['num_topics'],
            random_state=config['random_state']
        )
        
        instance.coherence_values = config['coherence_values']
        instance.perplexity_values = config['perplexity_values']
        
        # Load model-specific components
        if instance.method in ['lda', 'nmf']:
            instance.model = joblib.load(os.path.join(path, 'topic_model.joblib'))
            if os.path.exists(os.path.join(path, 'feature_names.joblib')):
                instance.feature_names = joblib.load(os.path.join(path, 'feature_names.joblib'))
        elif instance.method == 'gensim_lda':
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim is required to load this model")
            instance.model = LdaModel.load(os.path.join(path, 'gensim_lda_model'))
            if os.path.exists(os.path.join(path, 'gensim_dictionary')):
                instance.dictionary = Dictionary.load(os.path.join(path, 'gensim_dictionary'))
        
        # Load topic terms
        instance.topic_terms = joblib.load(os.path.join(path, 'topic_terms.joblib'))
        
        logger.info(f"Topic model loaded from {path}")
        return instance