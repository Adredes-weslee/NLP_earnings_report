# feature_extractor.py
# Functions for feature extraction and topic modeling

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import matplotlib.pyplot as plt
import os
import sys
import pickle

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (MAX_FEATURES, MAX_DOC_FREQ, NGRAM_RANGE, TOPIC_RANGE_MIN, 
                   TOPIC_RANGE_MAX, TOPIC_RANGE_STEP, TOPIC_WORD_PRIOR, 
                   DOC_TOPIC_PRIOR_FACTOR, SAMPLE_SIZE, OPTIMAL_TOPICS,
                   RANDOM_STATE, OUTPUT_DIR, MODEL_DIR)

def create_document_term_matrix(texts, save_path=None):
    """
    Create a document-term matrix from cleaned texts
    
    Args:
        texts (list or pandas.Series): Cleaned text data
        save_path (str, optional): Path to save the vectorizer
        
    Returns:
        tuple: (document-term matrix, vectorizer object, feature names)
    """
    from nltk.corpus import stopwords
    stops = stopwords.words('english')
    
    vec = CountVectorizer(
        token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b',
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        stop_words=stops,
        max_df=MAX_DOC_FREQ
    )
    
    dtm = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    
    print(f"DTM shape (documents x features): {dtm.shape}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vec, f)
        print(f"Vectorizer saved to {save_path}")
    
    return dtm, vec, vocab

def tune_lda_topics(dtm, vocab, sample_size=SAMPLE_SIZE, save_results=True):
    """
    Tune LDA model by testing a range of topic counts
    
    Args:
        dtm (scipy.sparse.csr.csr_matrix): Document-term matrix
        vocab (list): List of vocabulary terms
        sample_size (int): Number of documents to sample for tuning
        save_results (bool): Whether to save the tuning results
        
    Returns:
        dict: Records of coherence scores by topic count
    """
    if dtm.shape[0] <= sample_size:
        sample = pd.DataFrame(dtm.todense())
    else:
        sample = pd.DataFrame(dtm.todense()).sample(sample_size, random_state=RANDOM_STATE)
    
    records = []
    
    for top in range(TOPIC_RANGE_MIN, TOPIC_RANGE_MAX, TOPIC_RANGE_STEP):
        print(f"Fitting LDA with {top} topics...")
        record = {'topics': top}
        
        lda = LDA(
            n_components=top,
            topic_word_prior=TOPIC_WORD_PRIOR,
            doc_topic_prior=DOC_TOPIC_PRIOR_FACTOR/top,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        
        lda.fit(sample)
        
        umass = metric_coherence_gensim(
            'u_mass',
            topic_word_distrib=lda.components_,
            vocab=vocab,
            dtm=sample.values
        )
        
        record['mean_umass'] = np.mean(umass)
        records.append(record)
    
    if save_results and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df = pd.DataFrame(records)
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'lda_tuning_results.csv'), index=False)
    
    return records

def plot_topic_coherence(records, save_plot=True):
    """
    Plot the topic coherence scores from LDA tuning
    
    Args:
        records (list): List of dictionaries with tuning results
        save_plot (bool): Whether to save the plot
        
    Returns:
        tuple: (optimal topic count, plot)
    """
    topic_counts = [rec['topics'] for rec in records]
    umass_means = [rec['mean_umass'] for rec in records]
    
    # Find optimal topic count (least negative coherence score)
    optimal_idx = np.argmax(umass_means)
    optimal_topics = topic_counts[optimal_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(topic_counts, umass_means, marker='o')
    plt.axvline(x=optimal_topics, color='r', linestyle='--', 
                label=f'Optimal: {optimal_topics} topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Mean UMass Coherence')
    plt.title('Topic Coherence vs. Number of Topics')
    plt.legend()
    plt.grid(True)
    
    if save_plot and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'topic_coherence_plot.png'))
    
    return optimal_topics, plt

def fit_lda_model(dtm, n_topics=None, save_model=True):
    """
    Fit the final LDA model with the optimal number of topics
    
    Args:
        dtm (scipy.sparse.csr.csr_matrix): Document-term matrix
        n_topics (int, optional): Number of topics. If None, uses OPTIMAL_TOPICS from config
        save_model (bool): Whether to save the model
        
    Returns:
        tuple: (LDA model, topic distribution matrix)
    """
    if n_topics is None:
        n_topics = OPTIMAL_TOPICS
    
    print(f"Fitting final LDA model with {n_topics} topics...")
    
    final_lda = LDA(
        n_components=n_topics,
        topic_word_prior=TOPIC_WORD_PRIOR,
        doc_topic_prior=DOC_TOPIC_PRIOR_FACTOR/n_topics,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    final_lda.fit(dtm)
    topics = final_lda.transform(dtm)
    
    print(f"Topic distribution matrix shape: {topics.shape}")
    
    if save_model and MODEL_DIR:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, 'lda_model.pkl'), 'wb') as f:
            pickle.dump(final_lda, f)
        print(f"LDA model saved to {os.path.join(MODEL_DIR, 'lda_model.pkl')}")
    
    return final_lda, topics

def get_top_words(lda_model, vocab, n_words=5, save_results=True):
    """
    Get the top words for each topic in the LDA model
    
    Args:
        lda_model: Fitted LDA model
        vocab (list): Vocabulary list
        n_words (int): Number of top words to extract per topic
        save_results (bool): Whether to save the results
        
    Returns:
        dict: Dictionary mapping topic indices to lists of top words
    """
    topics_words = {}
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-(n_words):][::-1]
        top_words = [vocab[i] for i in top_indices]
        topics_words[topic_idx] = top_words
    
    if save_results and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, 'topic_top_words.txt'), 'w') as f:
            for topic_idx, words in topics_words.items():
                f.write(f"Topic {topic_idx}: {', '.join(words)}\n")
    
    return topics_words

if __name__ == "__main__":
    # This allows the script to be run directly
    import data_processor as dp
    
    print("Loading data...")
    try:
        df = pd.read_csv("./task2_data_clean.csv.gz")
        print("Using pre-processed data.")
    except:
        print("Pre-processed data not found. Processing raw data...")
        df = dp.load_data()
        df = dp.process_data(df, save_path='./task2_data_clean.csv.gz')
    
    print("Creating document-term matrix...")
    dtm, vec, vocab = create_document_term_matrix(df['clean_sent'], 
                                                save_path=os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    
    print("Tuning LDA topics...")
    records = tune_lda_topics(dtm, vocab)
    
    print("Plotting topic coherence...")
    optimal_topics, _ = plot_topic_coherence(records)
    
    print(f"Fitting final LDA model with {optimal_topics} topics...")
    lda_model, topics = fit_lda_model(dtm, n_topics=optimal_topics)
    
    print("Extracting top words per topic...")
    topics_words = get_top_words(lda_model, vocab)
    
    print("Feature extraction complete!")
    
    # Save topic distributions
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, 'topic_distributions.npy'), topics)
        print(f"Topic distributions saved to {os.path.join(OUTPUT_DIR, 'topic_distributions.npy')}")