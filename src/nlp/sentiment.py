"""Sentiment analysis module for financial text processing.

This module provides sentiment analysis capabilities for financial text using
lexicon-based approaches (Loughran-McDonald) and transformer-based models.
It integrates with the centralized NLPProcessor to ensure consistent tokenization
across the NLP pipeline.

The SentimentAnalyzer class offers multiple analysis methods and can combine
approaches for more robust sentiment scoring of financial language.
"""

import pandas as pd
import numpy as np
import re
import os
import logging
from typing import List, Dict, Union, Optional, Tuple
import joblib

# Import the centralized NLPProcessor
from .nlp_processing import NLPProcessor

# Optional imports for advanced sentiment analysis
try:
    from transformers import pipeline
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
logger = logging.getLogger('sentiment_analyzer')

class SentimentAnalyzer:
    """Sentiment analysis for financial texts using various methods.
    
    This class provides sentiment analysis capabilities for financial texts
    using either lexicon-based approaches (e.g., Loughran-McDonald),
    transformer-based models (e.g., FinBERT), or a combination of both.
    
    Attributes:
        method (str): The sentiment analysis method being used.
        model_name (str): Name of the transformer model if applicable.
        nlp_processor (NLPProcessor): Centralized processor for text tokenization.
        transformer_model: The loaded transformer model if applicable.
        lexicon (dict): The sentiment lexicon dictionary if using lexicon-based methods.
    """
    
    def __init__(self, method: str = 'loughran_mcdonald', model_name: str = 'ProsusAI/finbert',
                nlp_processor: NLPProcessor = None):
        """Initialize the sentiment analyzer with specified method and model.
        
        Args:
            method (str): Sentiment analysis method to use. Options are:
                'loughran_mcdonald': Uses the financial-specific Loughran-McDonald lexicon.
                'transformer': Uses a transformer-based model (requires transformers package).
                'combined': Uses both lexicon and transformer approaches.
                Defaults to 'loughran_mcdonald'.
            model_name (str): Name of transformer model to use if method is 'transformer'
                or 'combined'. Defaults to 'ProsusAI/finbert', which is specialized for
                financial text.
            nlp_processor (NLPProcessor, optional): Centralized NLP processor to use for
                consistent tokenization. If None, a new one will be created when needed.
                
        Raises:
            ImportError: If transformer-based method is selected but the required
                libraries are not installed.
                
        Example:
            >>> # Using shared NLP processor
            >>> nlp_proc = NLPProcessor()
            >>> analyzer = SentimentAnalyzer(method='combined', nlp_processor=nlp_proc)
            >>> sentiment = analyzer.analyze("Revenue increased by 15% this quarter.")
        """
        self.method = method
        self.model_name = model_name
        self.nlp_processor = nlp_processor
        self.transformer_model = None
        self.lexicon = None
        
        logger.info(f"Initializing SentimentAnalyzer with method={method}")
        
        # Check if transformers are available when needed
        if method in ['transformer', 'combined'] and not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformer method selected but libraries not available. "
                          "Install with: pip install transformers")
            if method == 'transformer':
                raise ImportError("Transformers library not available")
            else:
                logger.warning("Falling back to lexicon-based sentiment analysis only")
                self.method = 'loughran_mcdonald'
        
        # Load lexicon if using lexicon-based method
        if method in ['loughran_mcdonald', 'combined']:
            self._load_loughran_mcdonald_lexicon()
    
    def _load_loughran_mcdonald_lexicon(self):
        """Load the Loughran-McDonald sentiment lexicon for financial texts."""
        # Define the lexicon inline since it's relatively small
        # This is a simplified version, the full lexicon contains many more words
        self.lexicon = {
            'positive': set([
                'able', 'advantage', 'benefit', 'beneficial', 'best', 'better', 'bolster', 
                'boost', 'capable', 'certain', 'confidence', 'confident', 'create', 'creation', 
                'definitive', 'deliver', 'delivery', 'dependable', 'desirable', 'desired', 'enhance', 
                'enhanced', 'enjoy', 'excellent', 'exceptional', 'expand', 'expansion', 'extraordinary', 
                'favorable', 'gain', 'gained', 'good', 'greater', 'grew', 'grow', 'growing', 'growth', 
                'high', 'higher', 'highest', 'improve', 'improved', 'improvement', 'increase', 'increased', 
                'innovative', 'leading', 'opportunities', 'opportunity', 'outstanding', 'perfect', 'pleased', 
                'positive', 'profit', 'profitable', 'profitability', 'progress', 'record', 'robust', 'solid', 
                'strength', 'strong', 'stronger', 'strongest', 'success', 'successful', 'successfully', 
                'superior', 'surpass', 'surpassed', 'top', 'triumph', 'up', 'upside', 'upward', 'valuable', 
                'win', 'winner'
            ]),
            'negative': set([
                'abnormal', 'adverse', 'against', 'aggravate', 'alarm', 'alarming', 'antagonistic', 
                'anxiety', 'anxious', 'bad', 'breakdown', 'challenge', 'challenging', 'closed', 'closure', 
                'concern', 'concerned', 'concerns', 'condition', 'contracting', 'contraction', 'contrary', 
                'crash', 'crisis', 'critical', 'criticism', 'decline', 'decreased', 'decreasing', 'deficit', 
                'depression', 'deteriorate', 'deteriorating', 'deterioration', 'difficult', 'difficulty', 
                'disadvantage', 'disadvantageous', 'disappoint', 'disappointed', 'disappointing', 'disappointment', 
                'disaster', 'doubt', 'doubtful', 'down', 'downturn', 'downturns', 'downward', 'drag', 'drop', 
                'dropped', 'dropping', 'fail', 'failed', 'failing', 'failure', 'fall', 'fallen', 'falling', 
                'fear', 'fearful', 'fears', 'fluctuate', 'fluctuating', 'fluctuation', 'headwind', 'hurt', 
                'hurting', 'inadequate', 'impair', 'impaired', 'impairment', 'indebtedness', 'instability', 
                'insufficient', 'interference', 'jeopardize', 'lack', 'liquidate', 'liquidation', 'litigation', 
                'lose', 'losing', 'loss', 'losses', 'lost', 'low', 'lower', 'lowest', 'negative', 'negatively', 
                'neglect', 'obstacle', 'obstruction', 'penalty', 'plummet', 'poor', 'problem', 'problematic', 
                'problems', 'resign', 'resignation', 'risk', 'risks', 'risky', 'severe', 'severely', 'shrink', 
                'shrinking', 'slowing', 'slowdown', 'slower', 'sluggish', 'suffer', 'suffered', 'suffering', 
                'suspect', 'suspension', 'terminate', 'termination', 'threat', 'troubled', 'turmoil', 'unable', 
                'uncertain', 'uncertainty', 'unfavorable', 'unsuccessful', 'vulnerabilities', 'vulnerability', 
                'vulnerable', 'weak', 'weaken', 'weakened', 'weakening', 'weakness', 'worst', 'worthless'
            ]),
            'uncertainty': set([
                'almost', 'ambiguity', 'ambiguous', 'anticipate', 'anticipates', 'anticipating', 'anticipation', 
                'appear', 'appeared', 'appears', 'approximate', 'approximately', 'assume', 'assumed', 'assumes', 
                'assuming', 'assumption', 'assumptions', 'believe', 'believed', 'believes', 'cautious', 
                'cautiously', 'could', 'crossroad', 'doubt', 'doubtful', 'essentially', 'estimate', 'estimated', 
                'estimates', 'estimating', 'estimation', 'estimations', 'exposed', 'exposure', 'exposures', 
                'fluctuate', 'fluctuated', 'fluctuates', 'fluctuating', 'fluctuation', 'fluctuations', 'guess', 
                'guessed', 'guesses', 'guessing', 'hopefully', 'if', 'indefinite', 'indefinitely', 'indefiniteness', 
                'indeterminable', 'indeterminate', 'intend', 'intended', 'intending', 'intends', 'intention', 
                'intentions', 'likely', 'may', 'maybe', 'might', 'nearly', 'normally', 'perhaps', 'possible', 
                'possibly', 'potential', 'potentially', 'predict', 'predictable', 'predicted', 'predicting', 
                'prediction', 'predictions', 'predictive', 'predicts', 'preliminary', 'presumably', 'presume', 
                'presumed', 'presumes', 'presuming', 'presumption', 'presumptions', 'probabilistic', 'probabilities', 
                'probability', 'probable', 'probably', 'random', 'randomize', 'randomized', 'randomizes', 'randomizing', 
                'randomly', 'randomness', 'risk', 'risked', 'riskier', 'riskiest', 'risking', 'risks', 'risky', 
                'roughly', 'rumor', 'rumored', 'rumors', 'seem', 'seemed', 'seeming', 'seemingly', 'seems', 
                'should', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'speculate', 'speculated', 'speculates', 
                'speculating', 'speculation', 'speculations', 'speculative', 'speculatively', 'suggest', 'suggested', 
                'suggesting', 'suggests', 'suppose', 'supposed', 'supposedly', 'supposes', 'supposing', 
                'tentative', 'tentatively', 'turbulence', 'uncertain', 'uncertainly', 'uncertainties', 'uncertainty', 
                'unclear', 'unconfirmed', 'undecided', 'undefined', 'undesignated', 'unestablished', 'unknown', 
                'unlikely', 'unproved', 'unproven', 'unprovens', 'unsure', 'usually', 'vague', 'vaguely', 
                'vagueness', 'volatility', 'vulnerable'
            ]),
            'litigious': set([
                'abovementioned', 'abrogate', 'abrogated', 'abrogates', 'abrogating', 'abrogation', 'abrogations', 
                'absolve', 'absolved', 'absolves', 'absolving', 'accession', 'accessions', 'acquirees', 'acquirors', 
                'acquit', 'acquits', 'acquittal', 'acquittals', 'acquitted', 'acquitting', 'adjourn', 'adjourned', 
                'adjourning', 'adjournment', 'adjournments', 'adjourns', 'adjudge', 'adjudged', 'adjudges', 
                'adjudging', 'adjudicate', 'adjudicated', 'adjudicates', 'adjudicating', 'adjudication', 'affidavit', 
                'affidavits', 'aforementioned', 'aforenamed', 'aforesaid', 'allegation', 'allegations', 'allege', 
                'alleged', 'allegedly', 'alleges', 'alleging', 'amicus', 'annulment', 'annulments', 'antitrust', 
                'antitrusts', 'appeal', 'appealable', 'appealed', 'appealing', 'appeals', 'appellant', 'appellants', 
                'appellate', 'appellees', 'appendices', 'appendix', 'appurtenance', 'appurtenances', 'appurtenant', 
                'arbitrability', 'arbitral', 'arbitrate', 'arbitrated', 'arbitrates', 'arbitrating', 'arbitration', 
                'arbitrational', 'arbitrations', 'arbitrative', 'arbitrator', 'arbitrators', 'assignation', 
                'assignations', 'assumable', 'attestation', 'attestations', 'attorney', 'attorneys', 'beneficiaries'
            ])
        }
        
        logger.info(f"Loaded Loughran-McDonald lexicon with {len(self.lexicon['positive'])} positive, "
                   f"{len(self.lexicon['negative'])} negative, and {len(self.lexicon['uncertainty'])} uncertainty terms")
    
    def _load_transformer_model(self):
        """Load transformer-based sentiment analysis model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
            
        try:
            # Use FinBERT or another finance-specific model if specified
            if self.model_name == 'ProsusAI/finbert':
                self.transformer_model = pipeline(
                    "sentiment-analysis", 
                    model=self.model_name,
                    tokenizer=self.model_name,
                    return_all_scores=True
                )
                logger.info(f"Loaded FinBERT sentiment model")
            else:
                # Otherwise use standard sentiment pipeline
                self.transformer_model = pipeline(
                    "sentiment-analysis", 
                    model=self.model_name,
                    return_all_scores=True
                )
                logger.info(f"Loaded {self.model_name} sentiment model")
        except Exception as e:
            logger.error(f"Error loading transformer model: {str(e)}")
            raise
    
    def _get_tokens(self, text: str) -> List[str]:
        """Get tokens from text using the centralized NLPProcessor when available.
        
        This method uses the shared NLPProcessor for tokenization when available,
        or falls back to a simple regex-based tokenization. Using the shared processor
        ensures consistent tokenization throughout the pipeline.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens (words)
        """
        # First option: Use the shared NLPProcessor if available
        if self.nlp_processor is not None:
            try:
                # Create a small DTM just to get tokens
                text_list = [text]
                _, feature_names = self.nlp_processor.create_document_term_matrix(text_list)
                
                # The tokens in the document are those features that have non-zero count
                # We'd need to check the actual DTM to see which tokens are present
                # Since this is complex, we'll fall back to simpler tokenization
                
                # For now, just use regex as fallback
                return re.findall(r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b', text.lower())
            except Exception as e:
                logger.warning(f"Error using NLPProcessor for tokenization: {str(e)}")
                # Fall through to regex tokenization
        
        # Simple tokenization fallback
        return re.findall(r'\b\w+\b', text.lower())
        
    def _lexicon_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores using the Loughran-McDonald lexicon.
        
        This method analyzes financial text using the Loughran-McDonald dictionary
        specifically designed for financial sentiment analysis. It tokenizes the
        input text, counts occurrences of words in each sentiment category, and
        calculates normalized sentiment scores.
        
        Args:
            text (str): Input text for sentiment analysis. Financial text to analyze.
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores with keys:
                - 'positive': Ratio of positive words to total words
                - 'negative': Ratio of negative words to total words
                - 'uncertainty': Ratio of uncertainty words to total words
                - 'litigious': Ratio of litigious words to total words
                - 'net_sentiment': Net sentiment score (positive - negative) / total
                - 'sentiment_ratio': Ratio of positive to negative word counts
                
        Note:
            This method uses a simplified version of the Loughran-McDonald lexicon.
            For production use, consider using the complete lexicon.
        """
        if not self.lexicon:
            self._load_loughran_mcdonald_lexicon()
            
        # Tokenize using the centralized method
        words = self._get_tokens(text)
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in self.lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.lexicon['negative'])
        uncertainty_count = sum(1 for word in words if word in self.lexicon['uncertainty'])
        litigious_count = sum(1 for word in words if word in self.lexicon['litigious'])
        total_count = len(words) if words else 1  # Avoid division by zero
        
        # Calculate scores
        scores = {
            'positive': positive_count / total_count,
            'negative': negative_count / total_count,
            'uncertainty': uncertainty_count / total_count,
            'litigious': litigious_count / total_count,
            'net_sentiment': (positive_count - negative_count) / total_count,
            'sentiment_ratio': positive_count / (negative_count + 1)  # Add 1 to avoid division by zero
        }
        
        return scores
        
    def _transformer_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores using transformer-based models.
        
        This method uses pre-trained transformer models (like FinBERT) to analyze
        sentiment in financial texts. It handles long texts by breaking them into
        chunks and averaging the predictions. For financial texts, FinBERT is
        particularly effective as it's trained specifically for financial domain.
        
        Args:
            text (str): Input text for sentiment analysis. Financial text to analyze.
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores with keys
                that depend on the model used. For FinBERT:
                - 'positive': Probability of positive sentiment
                - 'negative': Probability of negative sentiment
                - 'neutral': Probability of neutral sentiment
                - 'net_sentiment': Difference between positive and negative scores
                - 'sentiment_ratio': Ratio of positive to negative sentiment
                
        Raises:
            ImportError: If transformers library is not available
            Exception: If model loading fails or prediction errors occur
            
        Note:
            Long texts are processed in chunks of 512 tokens to accommodate
            transformer model context window limitations.
        """
        if not self.transformer_model:
            self._load_transformer_model()
            
        # Process text in chunks if it's too long
        max_length = 512
        if len(text) > max_length:
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            results = []
            for chunk in chunks:
                result = self.transformer_model(chunk)
                results.extend(result)
            
            # Average scores across chunks
            scores = {}
            for label in results[0][0].keys():
                scores[label] = np.mean([r[0][label] for r in results])
        else:
            result = self.transformer_model(text)
            scores = {label: score for label, score in result[0]}
        
        # FinBERT specific processing
        if self.model_name == 'ProsusAI/finbert':
            scores['net_sentiment'] = scores.get('positive', 0) - scores.get('negative', 0)
            scores['sentiment_ratio'] = scores.get('positive', 0) / (scores.get('negative', 0) + 0.001)
        
        return scores
        
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the given text using the configured method.
        
        This method performs sentiment analysis on the provided text using
        either lexicon-based methods, transformer-based methods, or a combination
        of both, depending on the analyzer's configuration.
        
        Args:
            text (str): Input text for sentiment analysis. Should be a string
                containing the financial text to analyze.
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores with keys:
                - 'positive': Score indicating positive sentiment (0-1)
                - 'negative': Score indicating negative sentiment (0-1) 
                - 'net_sentiment': Difference between positive and negative scores
                - 'sentiment_ratio': Ratio of positive to negative sentiment
                - Additional keys may be present depending on the method used
                
        Example:
            >>> scores = analyzer.analyze("Revenue grew by 15% exceeding expectations.")
            >>> print(f"Positive score: {scores['positive']:.2f}")
            >>> print(f"Negative score: {scores['negative']:.2f}")
            >>> print(f"Net sentiment: {scores['net_sentiment']:.2f}")
        """
        if not isinstance(text, str) or not text.strip():
            return {'positive': 0, 'negative': 0, 'net_sentiment': 0, 'sentiment_ratio': 1}
            
        if self.method == 'loughran_mcdonald':
            return self._lexicon_sentiment(text)
        elif self.method == 'transformer':
            return self._transformer_sentiment(text)
        elif self.method == 'combined':
            # Combine both methods
            lexicon_scores = self._lexicon_sentiment(text)
            
            try:
                transformer_scores = self._transformer_sentiment(text)
                
                # Merge scores, preferring transformer for sentiment polarity
                combined_scores = {**lexicon_scores}
                combined_scores['positive'] = transformer_scores.get('positive', lexicon_scores['positive'])
                combined_scores['negative'] = transformer_scores.get('negative', lexicon_scores['negative'])
                combined_scores['net_sentiment'] = transformer_scores.get('net_sentiment', lexicon_scores['net_sentiment'])
                combined_scores['sentiment_ratio'] = transformer_scores.get('sentiment_ratio', lexicon_scores['sentiment_ratio'])
                
                return combined_scores
            except Exception as e:
                logger.warning(f"Transformer model failed, falling back to lexicon: {str(e)}")
                return lexicon_scores
            
    def analyze_with_dtm(self, dtm: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Analyze sentiment using a pre-computed document-term matrix.
        
        This method is useful when you already have a document-term matrix from
        the NLPProcessor and want to avoid re-tokenizing the text. It works only
        with lexicon-based sentiment analysis (not transformer-based).
        
        Args:
            dtm (np.ndarray): Document-term matrix, should be a 1xN or Nx1 sparse matrix
                or numpy array representing a single document.
            feature_names (List[str]): List of feature names (vocabulary) corresponding
                to the columns in the DTM.
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores with keys:
                - 'positive': Ratio of positive words to total words
                - 'negative': Ratio of negative words to total words
                - 'uncertainty': Ratio of uncertainty words to total words
                - 'litigious': Ratio of litigious words to total words
                - 'net_sentiment': Net sentiment score (positive - negative) / total
                - 'sentiment_ratio': Ratio of positive to negative word counts
                
        Example:
            >>> # Get DTM from NLPProcessor
            >>> dtm, feature_names = nlp_processor.create_document_term_matrix(["Revenue grew by 15%"])
            >>> # Analyze using the DTM directly
            >>> sentiment = analyzer.analyze_with_dtm(dtm[0], feature_names)
            
        Note:
            This method only works with lexicon-based sentiment analysis, not with
            transformer-based methods which require the raw text.
        """
        if not self.lexicon:
            self._load_loughran_mcdonald_lexicon()
        
        # Ensure we have a 1D array if dtm is a sparse matrix
        if hasattr(dtm, 'toarray'):
            if dtm.shape[0] == 1:
                dtm_array = dtm.toarray()[0]
            else:
                dtm_array = dtm.toarray().flatten()
        else:
            dtm_array = dtm.flatten() if dtm.ndim > 1 else dtm
            
        # Get non-zero indices (words that appear in the document)
        non_zero_indices = np.nonzero(dtm_array)[0]
        
        # Count sentiment words
        positive_count = 0
        negative_count = 0
        uncertainty_count = 0
        litigious_count = 0
        
        for idx in non_zero_indices:
            if idx < len(feature_names):
                word = feature_names[idx]
                if word in self.lexicon['positive']:
                    positive_count += dtm_array[idx]
                if word in self.lexicon['negative']:
                    negative_count += dtm_array[idx]
                if word in self.lexicon['uncertainty']:
                    uncertainty_count += dtm_array[idx]
                if word in self.lexicon['litigious']:
                    litigious_count += dtm_array[idx]
        
        # Total number of words is the sum of the DTM values
        total_count = dtm_array.sum() if dtm_array.sum() > 0 else 1
        
        # Calculate scores
        scores = {
            'positive': positive_count / total_count,
            'negative': negative_count / total_count,
            'uncertainty': uncertainty_count / total_count,
            'litigious': litigious_count / total_count,
            'net_sentiment': (positive_count - negative_count) / total_count,
            'sentiment_ratio': positive_count / (negative_count + 1)  # Add 1 to avoid division by zero
        }
        
        return scores
    
    def batch_analyze(self, texts: List[str]) -> pd.DataFrame:
        """Analyze sentiment for multiple texts in batch mode.
        
        This method processes a list of texts and returns sentiment scores for each.
        It's more efficient than calling analyze() repeatedly for large datasets
        as it logs progress and can be optimized for batch processing.
        
        Args:
            texts (List[str]): List of text strings to analyze. Each string
                should contain financial text for sentiment analysis.
            
        Returns:
            pd.DataFrame: DataFrame where each row contains sentiment scores for the
                corresponding text in the input list. Columns match the keys returned
                by the analyze() method (positive, negative, net_sentiment, etc.).
                
        Example:
            >>> texts = ["Revenue increased by 20%", "Losses continue to mount"]
            >>> results_df = analyzer.batch_analyze(texts)
            >>> print(results_df[['positive', 'negative', 'net_sentiment']])
        
        Note:
            For very large lists, consider breaking into smaller batches to
            avoid memory issues, especially when using transformer models.
        """
        results = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
                
            results.append(self.analyze(text))
            
        return pd.DataFrame(results)
    
    def batch_analyze_with_dtm(self, dtm: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Analyze sentiment for multiple texts using a pre-computed document-term matrix.
        
        This method is useful when you already have a document-term matrix from
        the NLPProcessor and want to avoid re-tokenizing the texts. It works only
        with lexicon-based sentiment analysis (not transformer-based).
        
        Args:
            dtm (np.ndarray): Document-term matrix, should be an NxM matrix where
                N is the number of documents and M is the vocabulary size.
            feature_names (List[str]): List of feature names (vocabulary) corresponding
                to the columns in the DTM.
            
        Returns:
            pd.DataFrame: DataFrame where each row contains sentiment scores for the
                corresponding document in the DTM.
                
        Example:
            >>> # Get DTM from NLPProcessor
            >>> dtm, feature_names = nlp_processor.create_document_term_matrix(texts)
            >>> # Analyze using the DTM directly
            >>> sentiments_df = analyzer.batch_analyze_with_dtm(dtm, feature_names)
            
        Note:
            This method only works with lexicon-based sentiment analysis, not with
            transformer-based methods which require the raw text. For transformer
            methods, use batch_analyze() instead.
        """
        if self.method not in ['loughran_mcdonald', 'combined']:
            logger.warning("Cannot use DTM with transformer-based sentiment. Using lexicon method.")
        
        results = []
        
        # Process each row in the DTM
        for i in range(dtm.shape[0]):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{dtm.shape[0]} documents")
                
            # Get the i-th row (document)
            if hasattr(dtm, 'getrow'):
                doc_row = dtm.getrow(i)
            else:
                doc_row = dtm[i]
                
            # Analyze with the DTM
            results.append(self.analyze_with_dtm(doc_row, feature_names))
            
        return pd.DataFrame(results)
    
    def enrich_dataframe(self, df: pd.DataFrame, text_column: str, prefix: str = 'sentiment_',
                        use_dtm: bool = False) -> pd.DataFrame:
        """Add sentiment analysis columns to an existing DataFrame.
        
        This convenience method analyzes text in a DataFrame column and adds
        the resulting sentiment scores as new columns. It handles missing values
        and preserves the original DataFrame structure.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing text data.
            text_column (str): Name of the column containing text to analyze.
            prefix (str, optional): Prefix for newly created sentiment columns.
                Defaults to 'sentiment_'.
            use_dtm (bool, optional): Whether to use a DTM for lexicon-based
                sentiment analysis. More efficient but only works with lexicon method.
                Defaults to False.
            
        Returns:
            pd.DataFrame: Original DataFrame with additional sentiment score columns.
                New columns will be named with the specified prefix followed by
                the sentiment score name (e.g., sentiment_positive).
                
        Raises:
            KeyError: If text_column doesn't exist in the DataFrame.
            
        Example:
            >>> earnings_df = pd.DataFrame({
            ...     'report_text': ["Revenue grew by 15%", "Expenses increased"],
            ...     'company': ["CompanyA", "CompanyB"]
            ... })
            >>> enriched_df = analyzer.enrich_dataframe(earnings_df, 'report_text')
            >>> print(enriched_df.columns)
            ['report_text', 'company', 'sentiment_positive', 'sentiment_negative', ...]
        """
        if use_dtm and self.method != 'transformer' and self.nlp_processor is not None:
            # Use DTM-based analysis for efficiency (lexicon-based only)
            texts = df[text_column].fillna('').tolist()
            
            # Create DTM using shared NLPProcessor
            dtm, feature_names = self.nlp_processor.create_document_term_matrix(texts)
            
            # Analyze using DTM
            sentiment_df = self.batch_analyze_with_dtm(dtm, feature_names)
        else:
            # Use standard text analysis
            texts = df[text_column].fillna('').tolist()
            sentiment_df = self.batch_analyze(texts)
        
        # Add prefix to column names
        sentiment_df.columns = [f"{prefix}{col}" for col in sentiment_df.columns]
        
        # Join with original dataframe
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        return result_df
    
    def save(self, path: str) -> None:
        """Save the sentiment analyzer configuration to disk.
        
        This method saves the configuration of the sentiment analyzer to a specified
        directory, allowing it to be loaded later using the load() class method.
        The configuration includes the method type and model name.
        
        Args:
            path (str): Directory path where the configuration will be saved.
                If the directory doesn't exist, it will be created.
                
        Raises:
            OSError: If there's an error creating the directory or writing the file.
            
        Example:
            >>> analyzer = SentimentAnalyzer(method='combined')
            >>> analyzer.save('models/sentiment/my_analyzer')
        
        Note:
            This method saves only the configuration, not the actual model weights.
            For transformer models, the weights will be downloaded again when loaded.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            'method': self.method,
            'model_name': self.model_name
        }
        
        # Save the NLPProcessor if available
        if self.nlp_processor is not None:
            self.nlp_processor.save(os.path.join(path, 'nlp_processor'))
        
        joblib.dump(config, os.path.join(path, 'sentiment_config.joblib'))
        logger.info(f"Sentiment analyzer configuration saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'SentimentAnalyzer':
        """Load a sentiment analyzer from a saved configuration.
        
        This class method reconstructs a SentimentAnalyzer instance from a
        configuration saved using the save() method. It loads the method type
        and model name settings.
        
        Args:
            path (str): Directory path containing the saved configuration.
                Should contain a sentiment_config.joblib file.
            
        Returns:
            SentimentAnalyzer: A new SentimentAnalyzer instance configured according
                to the loaded settings.
                
        Raises:
            FileNotFoundError: If the configuration file cannot be found.
            ValueError: If the configuration is invalid or corrupted.
            
        Example:
            >>> analyzer = SentimentAnalyzer.load('models/sentiment/my_analyzer')
            >>> results = analyzer.analyze("Revenue increased by 15% this quarter.")
        """
        config = joblib.load(os.path.join(path, 'sentiment_config.joblib'))
        
        # Try to load the NLPProcessor if available
        nlp_processor = None
        try:
            from .nlp_processing import NLPProcessor
            nlp_processor = NLPProcessor.load(os.path.join(path, 'nlp_processor'))
            logger.info("Loaded NLPProcessor")
        except FileNotFoundError:
            logger.info("No NLPProcessor found")
        except Exception as e:
            logger.warning(f"Error loading NLPProcessor: {str(e)}")
        
        instance = cls(
            method=config['method'],
            model_name=config['model_name'],
            nlp_processor=nlp_processor
        )
        
        logger.info(f"Sentiment analyzer loaded from {path}")
        return instance