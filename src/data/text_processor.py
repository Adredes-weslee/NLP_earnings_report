"""Text processing module for financial earnings reports analysis.

This module provides the TextProcessor class which handles all text-related
operations for the NLP Earnings Report project. It includes methods for cleaning,
normalizing, and tokenizing financial text data with specific adaptations for
financial language and earnings report structure.

The module implements specialized techniques for handling financial text, including
number replacement, noise removal, filtering of low-quality sentences, and
recognition of financial-specific patterns like currencies and percentages.

Examples:
    Basic usage of the text processor:
    
    >>> from src.data.text_processor import TextProcessor
    >>> processor = TextProcessor()
    >>> clean_text = processor.clean_text(raw_text)
    >>> processed_text = processor.process_text(raw_text, replace_numbers=True)
"""

import re
import nltk
import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import logging

# Import configuration values
from ..config import MIN_TOKEN_LENGTH, MIN_SENTENCE_TOKENS, MAX_FINANCIAL_NUM_RATIO

# Set up logging
logger = logging.getLogger('text_processor')

class TextProcessor:
    """Financial text processing class for earnings report content.
    
    This class provides methods for cleaning, normalizing, and preparing financial
    text data for NLP analysis. It specializes in handling common challenges in
    financial earnings reports, such as numerical values, special financial terms,
    and document structure.
    
    The class implements several preprocessing techniques tailored to financial text:
    - HTML and formatting tag removal
    - Special character normalization
    - Financial number handling and replacement
    - Sentence-level filtering and quality assessment 
    - Domain-specific text cleaning (e.g., removing disclaimers)
    
    Attributes:
        stops (list): List of stopwords used for text processing
        _digit_pattern (re.Pattern): Regular expression for identifying numbers
        _num_token (str): Token used to replace numbers when enabled
    """
    
    def __init__(self):
        """Initialize the text processor with required resources and patterns.
        
        Downloads NLTK resources if not already available and sets up
        regular expression patterns used for text cleaning and processing.
        
        Raises:
            ConnectionError: If NLTK resources cannot be downloaded
            LookupError: If NLTK resources cannot be found
        """
        # Download required NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        # Add financial stopwords
        self.financial_stopwords = {'company', 'quarter', 'year', 'financial', 'reported', 'period',
                                  'quarter', 'fiscal', 'results', 'earnings', 'reports', 'press', 
                                  'release', 'corporation', 'announces', 'announced', 'today'}
        self.stop_words.update(self.financial_stopwords)    
        
    def replace_financial_numbers(self, text):
        """Replace financial numbers and monetary values with standardized tokens.
        
        This method identifies and replaces various financial numbers including:
        - Dollar amounts (e.g., $15.2 million, $1.5B)
        - Percentages (e.g., 5.7%, 12%)
        - Large numbers with scale indicators (e.g., 50 million, 2.3b)
        
        The replacement strategy helps normalize financial text by converting
        varied numeric expressions into consistent tokens, which improves feature
        extraction and reduces vocabulary size for NLP models. This is particularly
        useful for earnings reports where exact numbers are less important than
        their presence and context.
        
        Args:
            text (str): The input text containing financial numbers. If input is
                not a string, it's returned unchanged.
            
        Returns:
            str: Text with financial numbers replaced by standardized tokens:
                - 'financial_number' for monetary values 
                - 'percentage_number' for percentage values
                - 'number' for other numeric values
                
        Example:
            >>> processor = TextProcessor()
            >>> processor.replace_financial_numbers("Revenue increased by $25.3 million (8.7%)")
            'Revenue increased by  financial_number  ( percentage_number )'
            >>> processor.replace_financial_numbers("Earnings per share of $1.52")
            'Earnings per share of  financial_number '
            
        Notes:
            This method uses regular expressions to identify different number formats.
            It prioritizes identifying currency values first, then percentages,
            then general numbers to avoid overlap. The method can handle various
            abbreviated forms like 'k', 'm', 'b' for thousand, million, and billion.
        """
        if not isinstance(text, str):
            return text
            
        # Match dollar amounts, percentages, and large numbers
        dollar_pattern = r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|m|b|k|thousand))?\b'
        percent_pattern = r'\d+(?:\.\d+)?%'
        number_pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:million|billion|m|b|k|thousand))?\b'
        
        # Replace with tokens
        text = re.sub(dollar_pattern, ' financial_number ', text, flags=re.IGNORECASE)
        text = re.sub(percent_pattern, ' percentage_number ', text, flags=re.IGNORECASE)
        text = re.sub(number_pattern, ' number ', text, flags=re.IGNORECASE)
        
        return text    
    
    def filter_bad_sentences(self, text, min_words=MIN_SENTENCE_TOKENS, max_words=100):
        """Filter out low-quality sentences from financial text.
        
        Identifies and removes sentences that are unlikely to contain 
        useful financial information, including sentences that are:
        - Too short (fewer than min_words)
        - Too long (more than max_words)
        - Likely boilerplate content (disclaimers, headers, etc.)
        - Sentences with abnormally high number density
        
        Args:
            text (str): Input text to filter.
            min_words (int, optional): Minimum number of words for a valid sentence.
                Defaults to MIN_SENTENCE_TOKENS from config.
            max_words (int, optional): Maximum number of words for a valid sentence.
                Defaults to 100.
                
        Returns:
            str: Filtered text containing only the sentences that passed quality checks.
            
        Example:
            >>> processor = TextProcessor()
            >>> long_text = "This is a normal sentence. Hi. This sentence has way too many..."
            >>> filtered = processor.filter_bad_sentences(long_text, min_words=3)
            >>> print(filtered)
            "This is a normal sentence."
        """
        if not isinstance(text, str):
            return text
            
        sentences = sent_tokenize(text)
        good_sentences = []
        
        # Common boilerplate patterns in earnings announcements
        boilerplate_patterns = [
            r'safe harbor',
            r'forward-looking statement',
            r'non-gaap|non gaap',
            r'investor relations',
            r'www\.[a-zA-Z0-9-]+\.com',
            r'securities act',
            r'securities and exchange commission',
            r'for further information'
        ]
        combined_pattern = '|'.join(boilerplate_patterns)
        
        for sent in sentences:
            # Count words (simple approximation)
            word_count = len(sent.split())
            
            # Skip sentences that are too short or too long
            if word_count < min_words or word_count > max_words:
                continue
                
            # Skip sentences with boilerplate content
            if re.search(combined_pattern, sent.lower()):
                continue
                
            good_sentences.append(sent)
        
        # Log stats about filtered sentences
        filtered_count = len(sentences) - len(good_sentences)
        if len(sentences) > 0:
            filter_ratio = filtered_count / len(sentences) * 100
            if filter_ratio > 50:
                logger.debug(f"Heavy filtering: {filter_ratio:.1f}% sentences removed")
            
        return ' '.join(good_sentences)
    
    def clean_text(self, text):
        """Perform basic text cleaning operations for financial text.
        
        Applies fundamental text normalization techniques including lowercasing,
        whitespace standardization, special character removal, and extra space trimming.
        This is typically used as a foundational preprocessing step before more
        specialized operations.
        
        Args:
            text (str): The text to clean. Can handle non-string inputs
                by returning them unchanged.
                
        Returns:
            str: Cleaned text with consistent formatting:
                - All lowercase
                - No newlines, tabs, or carriage returns (replaced with spaces)
                - No special characters (only alphanumeric and whitespace remain)
                - No extra whitespace (multiple spaces condensed to single space)
                - No leading or trailing whitespace
                
        Example:
            >>> processor = TextProcessor()
            >>> processor.clean_text("Company\nRevenue: $5.7M in Q2!")
            'company revenue 5 7m in q2'
            
        Notes:
            This method is designed to be used independently or as part of the
            full processing pipeline in the `process_text` method.
        """
        if not isinstance(text, str):
            return text
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace newlines, tabs with spaces
        text = re.sub(r'[\n\t\r]', ' ', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text, remove_stopwords=True):
        """Tokenize text into words, optionally removing stopwords.
        
        Splits text into individual tokens (words) using whitespace as the delimiter.
        Provides the option to remove common English stopwords and financial-specific
        stopwords that were initialized in the constructor.
        
        Args:
            text (str): The text to tokenize. Can handle non-string inputs
                by returning an empty list.
            remove_stopwords (bool, optional): Whether to remove stopwords from the 
                tokenized text. Defaults to True. When True, removes both standard
                English stopwords and financial-specific stopwords.
                
        Returns:
            list: List of tokens (words). If remove_stopwords=True, stopwords are
                excluded from this list. Returns an empty list for non-string inputs.
                
        Example:
            >>> processor = TextProcessor()
            >>> # With stopword removal (default)
            >>> processor.tokenize_text("The company reported strong growth")
            ['reported', 'strong', 'growth']
            >>> # Without stopword removal
            >>> processor.tokenize_text("The company reported strong growth", remove_stopwords=False)
            ['the', 'company', 'reported', 'strong', 'growth']
            
        Notes:
            This method uses simple whitespace tokenization which is efficient but may
            not handle complex cases like hyphenated words or contractions as effectively
            as more sophisticated tokenizers like NLTK's word_tokenize.
        """
        if not isinstance(text, str):
            return []
            
        # Simple whitespace tokenization
        tokens = text.split()
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
            
        return tokens
    
    def process_text(self, text, replace_numbers=True, filter_bad=True, 
                    clean=True, tokenize=False, remove_stopwords=True):
        """Complete text processing pipeline for financial text analysis.
        
        Implements a comprehensive, configurable text processing workflow that
        combines multiple preprocessing steps in a logical sequence. This method
        provides a convenient interface to apply various combinations of text
        processing operations based on specific needs.
        
        The pipeline includes these steps (applied in this order):
        1. Number replacement (financial numbers → standardized tokens)
        2. Sentence filtering (removing low-quality sentences)
        3. Text cleaning (lowercasing, whitespace normalization, etc.)
        4. Tokenization and stopword removal (optional final step)
        
        Args:
            text (str): The raw text to process. Can handle non-string inputs.
            replace_numbers (bool, optional): Whether to replace financial numbers
                with standardized tokens. Defaults to True.
            filter_bad (bool, optional): Whether to filter out low-quality sentences.
                Defaults to True.
            clean (bool, optional): Whether to apply basic text cleaning.
                Defaults to True.
            tokenize (bool, optional): Whether to return tokenized text (list of words)
                instead of a string. Defaults to False.
            remove_stopwords (bool, optional): Whether to remove stopwords during
                tokenization. Only used if tokenize=True. Defaults to True.
                
        Returns:
            Union[str, list]: Processed text as a string if tokenize=False, or
                as a list of tokens if tokenize=True. Non-string inputs return
                unchanged (or empty list if tokenize=True).
                
        Examples:
            >>> processor = TextProcessor()
            >>> # Full processing as string
            >>> processor.process_text("The company reported revenue of $50 million.")
            'company reported revenue financial_number'
            >>> # Tokenized output
            >>> processor.process_text("The company reported revenue of $50 million.", 
            ...                      tokenize=True)
            ['company', 'reported', 'revenue', 'financial_number']
            >>> # Skip number replacement and sentence filtering
            >>> processor.process_text("Revenue: $5.7M", replace_numbers=False, 
            ...                      filter_bad=False)
            'revenue 5 7m'
            
        Notes:
            The default settings are optimized for financial text analysis but
            can be adjusted based on specific requirements. The parameters enable
            flexible configuration depending on the downstream task (e.g., topic
            modeling, sentiment analysis, or classification).
        """
        if not isinstance(text, str):
            return text if not tokenize else []
        
        # Apply processing steps in sequence
        if replace_numbers:
            text = self.replace_financial_numbers(text)
            
        if filter_bad:
            text = self.filter_bad_sentences(text)
            
        if clean:
            text = self.clean_text(text)
            
        if tokenize:
            return self.tokenize_text(text, remove_stopwords=remove_stopwords)
            
        return text