# main.py
# Main script for orchestrating the earnings announcement analysis workflow

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config import (OUTPUT_DIR, MODEL_DIR, OPTIMAL_TOPICS, RANDOM_STATE)
from data_processor import load_data, process_data
from feature_extractor import (create_document_term_matrix, tune_lda_topics, 
                            plot_topic_coherence, fit_lda_model, get_top_words)
from model_trainer import train_lasso_model, train_classifiers
from utils import (plot_topic_words, plot_lasso_coefficients, generate_summary_report)


def setup_directories():
    """Create necessary directories for outputs and models"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Created output directories: {OUTPUT_DIR} and {MODEL_DIR}")


def run_data_processing(data_path=None, force_reprocess=False):
    """
    Load and process the raw data
    
    Args:
        data_path (str, optional): Path to the data file
        force_reprocess (bool): Whether to force reprocessing even if processed file exists
        
    Returns:
        pandas.DataFrame: Processed data
    """
    processed_file = "./task2_data_clean.csv.gz"
    
    if os.path.exists(processed_file) and not force_reprocess:
        print(f"Loading pre-processed data from {processed_file}")
        return pd.read_csv(processed_file)
    
    print("Loading and processing raw data...")
    df = load_data(data_path)
    processed_df = process_data(df, save_path=processed_file)
    
    return processed_df


def run_topic_modeling(df, n_topics=None, tune_topics=True):
    """
    Run topic modeling on the processed data
    
    Args:
        df (pandas.DataFrame): Processed data
        n_topics (int, optional): Number of topics to use
        tune_topics (bool): Whether to tune the number of topics
        
    Returns:
        tuple: (LDA model, topic distribution matrix, vocabulary)
    """
    print("Creating document-term matrix...")
    dtm, vec, vocab = create_document_term_matrix(
        df['clean_sent'],
        save_path=os.path.join(MODEL_DIR, 'vectorizer.pkl')
    )
    
    if tune_topics:
        print("Tuning LDA topics...")
        records = tune_lda_topics(dtm, vocab)
        
        print("Plotting topic coherence...")
        optimal_topics, _ = plot_topic_coherence(records)
        
        print(f"Optimal number of topics: {optimal_topics}")
    else:
        optimal_topics = n_topics or OPTIMAL_TOPICS
        print(f"Using specified number of topics: {optimal_topics}")
    
    print(f"Fitting final LDA model with {optimal_topics} topics...")
    lda_model, topics = fit_lda_model(dtm, n_topics=optimal_topics)
    
    print("Extracting top words per topic...")
    topics_words = get_top_words(lda_model, vocab)
    
    # Visualize topics
    print("Visualizing top topics...")
    plot_topic_words(lda_model, vocab, topics_to_show=range(min(20, optimal_topics)))
    
    return lda_model, topics, vec.get_feature_names_out(), topics_words


def run_lasso_regression(topics, returns):
    """
    Run Lasso regression to identify topics that best predict returns
    
    Args:
        topics (numpy.ndarray): Topic distribution matrix
        returns (pandas.Series): Stock returns
        
    Returns:
        tuple: (Lasso model, results dictionary, nonzero topics indices)
    """
    print("Training Lasso regression model...")
    lasso_model, lasso_results, nonzero_topics = train_lasso_model(topics, returns)
    
    print("Visualizing Lasso coefficients...")
    plot_lasso_coefficients(lasso_model.coef_)
    
    return lasso_model, lasso_results, nonzero_topics


def run_classification(topics, returns):
    """
    Train and evaluate classifiers to predict large positive returns
    
    Args:
        topics (numpy.ndarray): Topic distribution matrix
        returns (pandas.Series): Stock returns
        
    Returns:
        tuple: (Best model, results dictionary)
    """
    print("Training classification models...")
    best_classifier, classifier_results = train_classifiers(topics, returns)
    
    best_model_name = max(classifier_results, key=lambda x: classifier_results[x]['test_f1_macro'])
    best_f1 = classifier_results[best_model_name]['test_f1_macro']
    
    print(f"Best model: {best_model_name} with F1 score: {best_f1:.4f}")
    
    return best_classifier, classifier_results


def generate_report(lda_model, lasso_results, classifier_results, topics_words):
    """
    Generate a comprehensive analysis report
    
    Args:
        lda_model: Fitted LDA model
        lasso_results (dict): Results from Lasso regression
        classifier_results (dict): Results from classifier training
        topics_words (dict): Dictionary mapping topic indices to lists of top words
        
    Returns:
        str: Report text
    """
    print("Generating analysis report...")
    
    lda_results = {
        'n_topics': lda_model.n_components,
        'coherence_score': -0.97  # Placeholder, would be from actual results
    }
    
    report_text = generate_summary_report(
        lda_results, lasso_results, classifier_results, topics_words
    )
    
    print(f"Report generated and saved to {os.path.join(OUTPUT_DIR, 'analysis_report.md')}")
    
    return report_text


def run_full_pipeline(data_path=None, n_topics=None, force_reprocess=False, tune_topics=True):
    """
    Run the complete analysis pipeline
    
    Args:
        data_path (str, optional): Path to the data file
        n_topics (int, optional): Number of topics to use
        force_reprocess (bool): Whether to force reprocessing even if processed file exists
        tune_topics (bool): Whether to tune the number of topics
        
    Returns:
        None
    """
    print("Starting earnings announcement analysis pipeline...")
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Data processing
    df = run_data_processing(data_path, force_reprocess)
    print(f"Data processed successfully. Shape: {df.shape}")
    
    # Step 3: Topic modeling
    lda_model, topics, vocab, topics_words = run_topic_modeling(df, n_topics, tune_topics)
    print("Topic modeling completed successfully.")
    
    # Step 4: Lasso regression
    lasso_model, lasso_results, nonzero_topics = run_lasso_regression(topics, df['BHAR0_2'])
    print("Lasso regression analysis completed successfully.")
    
    # Step 5: Classification
    best_classifier, classifier_results = run_classification(topics, df['BHAR0_2'])
    print("Classification analysis completed successfully.")
    
    # Step 6: Generate report
    report = generate_report(lda_model, lasso_results, classifier_results, topics_words)
    
    print("\nAnalysis pipeline completed successfully!")
    print(f"Results and models saved to {OUTPUT_DIR} and {MODEL_DIR}")
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings Announcement Analysis Pipeline")
    
    parser.add_argument("--data-path", type=str, help="Path to the data file")
    parser.add_argument("--n-topics", type=int, help="Number of topics for LDA")
    parser.add_argument("--force-reprocess", action="store_true", help="Force data reprocessing")
    parser.add_argument("--no-tune-topics", action="store_true", help="Skip topic tuning")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        data_path=args.data_path,
        n_topics=args.n_topics,
        force_reprocess=args.force_reprocess,
        tune_topics=not args.no_tune_topics
    )