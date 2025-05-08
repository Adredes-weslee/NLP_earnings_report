# data_processor.py
# Functions for loading and preprocessing earnings announcement data

import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import sys
import logging

# Debug print statements for paths
print("Current working directory:", os.getcwd())
print("__file__ value:", __file__)
print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
print("Parent directory:", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("sys.path contains:", sys.path)

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Import configuration using absolute import to avoid relative import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ROOT_DIR, DATA_DIR, RAW_DATA_PATH, PROCESSED_DATA_DIR,
                   VOCAB_SIZE, LARGE_RETURN_THRESHOLD)

# Print path variables from config
print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("RAW_DATA_PATH:", RAW_DATA_PATH)
print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)

# Define constants for text processing
MIN_TOKEN_LENGTH = 3
MAX_FINANCIAL_NUM_RATIO = 0.5
MIN_SENTENCE_TOKENS = 5

def load_data(file_path=None):
    """
    Load earnings announcement data from the specified file
    
    Args:
        file_path (str, optional): Path to the data file. If None, uses the path from config.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    if file_path is None:
        # Use the data path from config
        file_path = RAW_DATA_PATH
    
    df = pd.read_csv(file_path, compression='gzip')
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def clean_sentences(txt):
    """
    Clean and filter sentences from the earnings announcement text.
    
    This function:
      - Replaces financial numbers with the token "financial_number"
      - Tokenizes the text into sentences, then words
      - Keeps only tokens that are either 'financial_number' or words with all letters
        (alphabetic) of at least MIN_TOKEN_LENGTH characters
      - Discards sentences with fewer than MIN_SENTENCE_TOKENS tokens or in which 
        more than MAX_FINANCIAL_NUM_RATIO of tokens are financial numbers
      - Returns the cleaned sentences reassembled as one string
    
    Args:
        txt (str): Raw text from an earnings announcement
        
    Returns:
        str: Cleaned and filtered text
    """
    if not isinstance(txt, str):
        return ""
        
    clean_txt = re.sub(r'\$?\d+(?:,\d{3})*(?:\.\d+)?', " financial_number ", txt)
    sentences = sent_tokenize(clean_txt)
    good_sents = []
    
    for sent in sentences:
        tokens = word_tokenize(sent)
        good_tokens = []
        
        for token in tokens:
            if token == "financial_number":
                good_tokens.append(token)
            elif token.isalpha() and len(token) >= MIN_TOKEN_LENGTH:
                good_tokens.append(token.lower())
                
        if len(good_tokens) >= MIN_SENTENCE_TOKENS:
            num_fin = sum(1 for token in good_tokens if token == "financial_number")
            if (num_fin / len(good_tokens)) <= MAX_FINANCIAL_NUM_RATIO:
                good_sents.append(" ".join(good_tokens))
                
    return " ".join(good_sents)

def process_data(df, text_column='ea_text', save_path=None):
    """
    Process the raw dataframe by cleaning the text in the specified column
    
    Args:
        df (pandas.DataFrame): DataFrame containing earnings announcement data
        text_column (str): Name of the column containing the text to clean
        save_path (str, optional): Path to save the processed DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned text in a new 'clean_sent' column
    """
    print("Cleaning sentences in earnings announcements...")
    df['clean_sent'] = df[text_column].apply(clean_sentences)
    
    if save_path:
        df.to_csv(save_path, compression='gzip', index=False)
        print(f"Processed data saved to {save_path}")
    
    return df

def compute_text_statistics(df, text_column='clean_sent'):
    """
    Compute basic statistics about the texts in the dataset
    
    Args:
        df (pandas.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing the text
        
    Returns:
        dict: Dictionary with text statistics
    """
    # Calculate text lengths
    df['text_length'] = df[text_column].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    
    stats = {
        'mean_length': df['text_length'].mean(),
        'median_length': df['text_length'].median(),
        'min_length': df['text_length'].min(),
        'max_length': df['text_length'].max(),
        'std_length': df['text_length'].std(),
        'total_documents': len(df),
        'empty_documents': sum(df['text_length'] == 0)
    }
    
    return stats

if __name__ == "__main__":
    # This allows the script to be run directly
    print("Loading and processing earnings announcement data...")
    df = load_data()
    processed_df = process_data(df, save_path='./task2_data_clean.csv.gz')
    print("Data processing complete!")