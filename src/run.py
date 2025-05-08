#!/usr/bin/env python
"""
Run script for EarningsNLP package.
This script provides a simple way to execute different components of the analysis.
"""

import os
import sys
import argparse
from main import run_full_pipeline

def setup_nltk_data():
    """Download required NLTK data packages"""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    print("NLTK data setup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EarningsNLP analysis")
    
    parser.add_argument("--mode", type=str, default="full", 
                      choices=["full", "data", "topic", "lasso", "classify", "streamlit"],
                      help="Mode to run (default: full)")
    
    parser.add_argument("--data-path", type=str, help="Path to the data file")
    parser.add_argument("--n-topics", type=int, help="Number of topics for LDA")
    parser.add_argument("--force-reprocess", action="store_true", help="Force data reprocessing")
    parser.add_argument("--no-tune-topics", action="store_true", help="Skip topic tuning")
    
    args = parser.parse_args()
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Execute based on mode
    if args.mode == "full":
        run_full_pipeline(
            data_path=args.data_path,
            n_topics=args.n_topics,
            force_reprocess=args.force_reprocess,
            tune_topics=not args.no_tune_topics
        )
        
    elif args.mode == "data":
        from src.data.data_processor import load_data, process_data
        df = load_data(args.data_path)
        processed_df = process_data(df, save_path="./task2_data_clean.csv.gz")
        print(f"Data processing complete. Shape: {processed_df.shape}")
        
    elif args.mode == "topic":
        from src.data.data_processor import load_data
        from src.models.feature_extractor import create_document_term_matrix, tune_lda_topics, plot_topic_coherence, fit_lda_model, get_top_words
        try:
            df = pd.read_csv("./task2_data_clean.csv.gz")
            print("Using pre-processed data.")
        except:
            print("Pre-processed data not found. Processing raw data...")
            df = load_data(args.data_path)
            from src.data.data_processor import process_data
            df = process_data(df, save_path="./task2_data_clean.csv.gz")
            
        dtm, vec, vocab = create_document_term_matrix(df['clean_sent'])
        if not args.no_tune_topics:
            records = tune_lda_topics(dtm, vocab)
            optimal_topics, _ = plot_topic_coherence(records)
        else:
            optimal_topics = args.n_topics or 40
        lda_model, topics = fit_lda_model(dtm, n_topics=optimal_topics)
        topics_words = get_top_words(lda_model, vocab)
        print("Topic modeling complete!")
        
    elif args.mode == "streamlit":
        os.system("streamlit run streamlit_app.py")
        
    else:
        print("Please run the full pipeline first to generate required intermediate files.")
        print("Use: python run.py --mode full")