"""
Enhanced text processor for NLP Earnings Report project.
Handles text preprocessing, cleaning, and tokenization with financial domain focus.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import logging

# Set up logging
logger = logging.getLogger('text_processor')

class TextProcessor:
    def __init__(self):
        """Initialize text processor with default parameters"""
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
        """Replace financial numbers with a token"""
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
    
    def filter_bad_sentences(self, text, min_words=3, max_words=100):
        """Filter out sentences that are too short, too long, or contain unwanted patterns"""
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
        """Basic text cleaning operations"""
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
        """Tokenize text into words, optionally removing stopwords"""
        if not isinstance(text, str):
            return []
            
        # Simple whitespace tokenization
        tokens = text.split()
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
            
        return tokens
    
    def process_text(self, text, replace_numbers=True, filter_bad=True, 
                    clean=True, tokenize=False, remove_stopwords=True):
        """Complete text processing pipeline"""
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