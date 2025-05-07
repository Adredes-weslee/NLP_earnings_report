"""
Sentiment analysis module for financial text processing.
Incorporates financial-specific lexicons and transformer-based models.
"""

import pandas as pd
import numpy as np
import re
import os
import logging
from typing import List, Dict, Union, Optional
import joblib

# Optional imports for advanced sentiment analysis
try:
    from transformers import pipeline
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
logger = logging.getLogger('sentiment_analyzer')

class SentimentAnalyzer:
    """
    Class for analyzing sentiment in financial texts using various methods.
    Supports lexicon-based and transformer-based sentiment analysis.
    """
    
    def __init__(self, method: str = 'loughran_mcdonald', model_name: str = 'ProsusAI/finbert'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            method: Sentiment analysis method ('loughran_mcdonald', 'transformer', 'combined')
            model_name: Name of transformer model (if method='transformer' or 'combined')
        """
        self.method = method
        self.model_name = model_name
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
    
    def _lexicon_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment scores using lexicon-based approach.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Dict containing sentiment scores
        """
        if not self.lexicon:
            self._load_loughran_mcdonald_lexicon()
            
        # Tokenize and clean text
        words = re.findall(r'\b\w+\b', text.lower())
        
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
        """
        Calculate sentiment scores using transformer-based approach.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Dict containing sentiment scores
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
        """
        Analyze sentiment of the given text using the configured method.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Dict containing sentiment scores
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
    
    def batch_analyze(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            DataFrame with sentiment scores for each text
        """
        results = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
                
            results.append(self.analyze(text))
            
        return pd.DataFrame(results)
    
    def enrich_dataframe(self, df: pd.DataFrame, text_column: str, prefix: str = 'sentiment_') -> pd.DataFrame:
        """
        Add sentiment analysis columns to a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            prefix: Prefix for sentiment columns
            
        Returns:
            DataFrame with sentiment columns added
        """
        texts = df[text_column].fillna('').tolist()
        sentiment_df = self.batch_analyze(texts)
        
        # Add prefix to column names
        sentiment_df.columns = [f"{prefix}{col}" for col in sentiment_df.columns]
        
        # Join with original dataframe
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        return result_df
    
    def save(self, path: str) -> None:
        """
        Save the sentiment analyzer configuration.
        
        Args:
            path: Directory path to save configuration
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            'method': self.method,
            'model_name': self.model_name
        }
        
        joblib.dump(config, os.path.join(path, 'sentiment_config.joblib'))
        logger.info(f"Sentiment analyzer configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SentimentAnalyzer':
        """
        Load a sentiment analyzer from configuration.
        
        Args:
            path: Directory path to load configuration from
            
        Returns:
            Loaded SentimentAnalyzer instance
        """
        config = joblib.load(os.path.join(path, 'sentiment_config.joblib'))
        
        instance = cls(
            method=config['method'],
            model_name=config['model_name']
        )
        
        logger.info(f"Sentiment analyzer loaded from {path}")
        return instance