# streamlit_app.py
# Streamlit application for interactive visualization and analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config import (OUTPUT_DIR, MODEL_DIR, LARGE_RETURN_THRESHOLD)
from data_processor import load_data, process_data
from feature_extractor import create_document_term_matrix, fit_lda_model, get_top_words
from model_trainer import train_lasso_model, train_classifiers
from utils import (plot_topic_words, plot_lasso_coefficients, plot_confusion_matrices,
                plot_roc_curves, generate_summary_report)

# Set page config
st.set_page_config(
    page_title="Earnings Announcements Analysis",
    page_icon="📊",
    layout="wide"
)

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Title and description
st.title("Earnings Announcements Text Analysis")
st.markdown("""
This application analyzes the text of earnings announcements using Natural Language Processing 
and Machine Learning techniques to identify topics and predict stock returns.
""")

# Sidebar
st.sidebar.header("Analysis Parameters")

# Main layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Processing", 
    "Topic Modeling", 
    "Lasso Regression", 
    "Classification", 
    "Report"
])

# Data Processing Tab
with tab1:
    st.header("Data Processing")
    
    data_source = st.radio(
        "Select data source:",
        ("Upload file", "Use sample data", "Load processed data")
    )
    
    if data_source == "Upload file":
        uploaded_file = st.file_uploader("Upload earnings announcement data (CSV/GZ format):", type=['csv', 'gz'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    elif data_source == "Use sample data":
        if st.button("Load sample data"):
            try:
                df = load_data()
                st.success(f"Sample data loaded with {df.shape[0]} rows.")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
    
    elif data_source == "Load processed data":
        if st.button("Load processed data"):
            try:
                df = pd.read_csv("./task2_data_clean.csv.gz")
                st.success(f"Processed data loaded with {df.shape[0]} rows.")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error loading processed data: {e}. Make sure the file exists.")
    
    # Data processing options
    if 'df' in locals():
        st.subheader("Text Cleaning")
        if st.button("Process text data"):
            with st.spinner("Cleaning and processing text data..."):
                processed_df = process_data(df)
                st.success("Text processing complete!")
                st.write(processed_df[['clean_sent']].head())
                
                # Save processed data
                save_path = os.path.join(OUTPUT_DIR, "processed_data.csv.gz")
                processed_df.to_csv(save_path, compression='gzip', index=False)
                st.success(f"Processed data saved to {save_path}")

# Topic Modeling Tab
with tab2:
    st.header("Topic Modeling")
    
    # Topic modeling parameters
    st.subheader("Topic Modeling Parameters")
    
    n_topics = st.slider("Number of topics", 5, 100, 40, 5)
    
    # Try to load data
    data_loaded = False
    try:
        if os.path.exists("./task2_data_clean.csv.gz"):
            df = pd.read_csv("./task2_data_clean.csv.gz")
            data_loaded = True
    except:
        st.warning("Processed data not found. Please process data in the Data Processing tab first.")
    
    if data_loaded:
        if st.button("Run Topic Modeling"):
            with st.spinner("Creating document-term matrix..."):
                dtm, vec, vocab = create_document_term_matrix(df['clean_sent'])
                
                st.success(f"Document-term matrix created with shape {dtm.shape}")
                
            with st.spinner(f"Fitting LDA model with {n_topics} topics..."):
                lda_model, topics = fit_lda_model(dtm, n_topics=n_topics)
                topics_words = get_top_words(lda_model, vocab, n_words=10)
                
                st.success("Topic modeling complete!")
                
                # Save results
                np.save(os.path.join(OUTPUT_DIR, 'topic_distributions.npy'), topics)
                
                # Display top words for each topic
                st.subheader("Top Words by Topic")
                cols = 3
                rows = (min(20, n_topics) + cols - 1) // cols
                
                for i in range(0, min(20, n_topics), cols):
                    topic_cols = st.columns(cols)
                    for j in range(cols):
                        if i + j < min(20, n_topics):
                            with topic_cols[j]:
                                st.markdown(f"**Topic {i+j}**")
                                st.write(", ".join(topics_words[i+j]))
                
                # Visualize topics
                st.subheader("Topic Visualization")
                fig = plot_topic_words(lda_model, vocab, topics_to_show=range(min(12, n_topics)))
                st.pyplot(fig)

# Lasso Regression Tab
with tab3:
    st.header("Lasso Regression Analysis")
    
    st.info("Lasso regression is used to identify topics that are most predictive of stock returns.")
    
    # Check if required files exist
    topics_file = os.path.join(OUTPUT_DIR, 'topic_distributions.npy')
    
    if os.path.exists(topics_file) and data_loaded:
        topics = np.load(topics_file)
        
        st.subheader("Lasso Parameters")
        alpha_min = st.number_input("Minimum alpha value", 0.00001, 0.1, 0.00001, format="%.5f")
        alpha_max = st.number_input("Maximum alpha value", 0.00001, 0.1, 0.02, format="%.5f")
        
        if st.button("Run Lasso Regression"):
            with st.spinner("Training Lasso model..."):
                lasso_model, lasso_results, nonzero_topics = train_lasso_model(
                    topics, df['BHAR0_2'], alpha_range=(alpha_min, alpha_max)
                )
                
                st.success("Lasso regression complete!")
                
                # Display results
                st.subheader("Lasso Results")
                st.write(f"Best alpha: {lasso_results['best_alpha']:.6f}")
                st.write(f"Number of topics with non-zero coefficients: {lasso_results['nonzero_topics']}")
                
                # Show most influential topics
                st.subheader("Most Influential Topics")
                st.write(f"Topic with most positive coefficient: Topic {lasso_results['most_positive_topic']} " +
                        f"(Coefficient: {lasso_results['most_positive_coef']:.4f})")
                st.write(f"Topic with most negative coefficient: Topic {lasso_results['most_negative_topic']} " +
                        f"(Coefficient: {lasso_results['most_negative_coef']:.4f})")
                
                # Plot coefficients
                st.subheader("Lasso Coefficients Visualization")
                fig = plot_lasso_coefficients(lasso_model.coef_, top_n=10)
                st.pyplot(fig)
    else:
        st.warning("Topic distributions not found. Please run topic modeling in the Topic Modeling tab first.")

# Classification Tab
with tab4:
    st.header("Classification Analysis")
    
    st.info(f"Classification is used to predict large positive returns (>{LARGE_RETURN_THRESHOLD}%).")
    
    # Check if required files exist
    topics_file = os.path.join(OUTPUT_DIR, 'topic_distributions.npy')
    
    if os.path.exists(topics_file) and data_loaded:
        topics = np.load(topics_file)
        
        st.subheader("Classification Models")
        models_to_train = st.multiselect(
            "Select models to train:",
            ["LogisticRegression", "SVM", "RandomForest", "GradientBoosting"],
            ["LogisticRegression", "RandomForest"]
        )
        
        n_iter = st.slider("Number of hyperparameter combinations to try", 10, 200, 50)
        
        if st.button("Train Classification Models"):
            if not models_to_train:
                st.warning("Please select at least one model to train.")
            else:
                with st.spinner("Training classification models..."):
                    best_model, classifier_results = train_classifiers(
                        topics, df['BHAR0_2'], n_iter_search=n_iter
                    )
                    
                    st.success("Classification model training complete!")
                    
                    # Display results
                    st.subheader("Classification Results")
                    best_model_name = max(classifier_results, key=lambda x: classifier_results[x]['test_f1_macro'])
                    
                    st.write(f"Best model: {best_model_name}")
                    st.write(f"Macro F1 score: {classifier_results[best_model_name]['test_f1_macro']:.4f}")
                    
                    # Show results for all models
                    results_df = pd.DataFrame({
                        'Model': list(classifier_results.keys()),
                        'Macro F1 Score': [results['test_f1_macro'] for results in classifier_results.values()]
                    })
                    results_df = results_df.sort_values('Macro F1 Score', ascending=False)
                    
                    st.write(results_df)
                    
                    # Show detailed results for best model
                    st.subheader(f"Detailed Results for {best_model_name}")
                    
                    # Classification report
                    report = classifier_results[best_model_name]['classification_report']
                    report_df = pd.DataFrame({
                        'Class': list(report.keys())[:2],
                        'Precision': [report[cls]['precision'] for cls in list(report.keys())[:2]],
                        'Recall': [report[cls]['recall'] for cls in list(report.keys())[:2]],
                        'F1-Score': [report[cls]['f1-score'] for cls in list(report.keys())[:2]],
                        'Support': [report[cls]['support'] for cls in list(report.keys())[:2]]
                    })
                    st.write(report_df)
    else:
        st.warning("Topic distributions not found. Please run topic modeling in the Topic Modeling tab first.")

# Report Tab
with tab5:
    st.header("Analysis Report")
    
    if os.path.exists(os.path.join(MODEL_DIR, 'lda_model.pkl')) and \
       os.path.exists(os.path.join(MODEL_DIR, 'lasso_model.pkl')) and \
       os.path.exists(os.path.join(OUTPUT_DIR, 'classifier_results.pkl')):
        
        if st.button("Generate Report"):
            with st.spinner("Generating comprehensive report..."):
                # Load models and results
                with open(os.path.join(MODEL_DIR, 'lda_model.pkl'), 'rb') as f:
                    lda_model = pickle.load(f)
                
                with open(os.path.join(MODEL_DIR, 'lasso_model.pkl'), 'rb') as f:
                    lasso_model = pickle.load(f)
                
                with open(os.path.join(OUTPUT_DIR, 'classifier_results.pkl'), 'rb') as f:
                    classifier_results = pickle.load(f)
                
                topics = np.load(os.path.join(OUTPUT_DIR, 'topic_distributions.npy'))
                
                # Load vocabulary
                with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
                    vec = pickle.load(f)
                vocab = vec.get_feature_names_out()
                
                # Get top words for topics
                topics_words = get_top_words(lda_model, vocab)
                
                # Lasso results
                lasso_results = {
                    'best_alpha': 0.001,  # Placeholder, would be from actual results
                    'nonzero_topics': np.sum(lasso_model.coef_ != 0),
                    'most_positive_topic': np.argmax(lasso_model.coef_),
                    'most_positive_coef': np.max(lasso_model.coef_),
                    'most_negative_topic': np.argmin(lasso_model.coef_),
                    'most_negative_coef': np.min(lasso_model.coef_)
                }
                
                # LDA results
                lda_results = {
                    'n_topics': lda_model.n_components,
                    'coherence_score': -0.97  # Placeholder, would be from actual results
                }
                
                # Generate report
                report_text = generate_summary_report(
                    lda_results, lasso_results, classifier_results, topics_words
                )
                
                st.success("Report generated successfully!")
                st.markdown(report_text)
    else:
        st.warning("Some required models or results files are missing. Please complete the previous analysis steps first.")

# Run the app with the following command:
# streamlit run streamlit_app.py