"""
Streamlit application for interactive visualization and analysis of earnings reports.
Uses NLP techniques to analyze earnings announcement texts and predict market reactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import joblib
import logging
from PIL import Image
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (OUTPUT_DIR, MODEL_DIR, LARGE_RETURN_THRESHOLD, 
                   EMBEDDING_MODEL_PATH, SENTIMENT_MODEL_PATH, 
                   TOPIC_MODEL_PATH, FEATURE_EXTRACTOR_PATH)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('streamlit_app')

# Set page config
st.set_page_config(
    page_title="Earnings Announcements Analysis",
    page_icon="📊",
    layout="wide"
)

# Check if models exist, if not, create placeholders
def check_models_exist():
    """Check if required models exist, if not create placeholders"""
    from create_placeholder_models import create_placeholder_models
    
    # Check for key model files
    if not os.path.exists(os.path.join(EMBEDDING_MODEL_PATH, 'vectorizer.joblib')):
        with st.spinner("Setting up placeholder models for demonstration..."):
            create_placeholder_models()
        st.success("Placeholder models created for demonstration")
        
    return True

# Load models
@st.cache_resource
def load_models():
    """Load the trained models"""
    models = {}
    
    try:
        # Load TF-IDF vectorizer
        models['vectorizer'] = joblib.load(os.path.join(EMBEDDING_MODEL_PATH, 'vectorizer.joblib'))
        models['embedding_config'] = joblib.load(os.path.join(EMBEDDING_MODEL_PATH, 'config.joblib'))
        
        # Load sentiment analyzer config
        models['sentiment_config'] = joblib.load(os.path.join(SENTIMENT_MODEL_PATH, 'sentiment_config.joblib'))
        
        # Load topic model
        topic_model_path = os.path.join(TOPIC_MODEL_PATH, 'lda_model.pkl')
        if os.path.exists(topic_model_path):
            with open(topic_model_path, 'rb') as f:
                models['topic_model'] = pickle.load(f)
        
        # Load feature extractor
        feature_path = os.path.join(FEATURE_EXTRACTOR_PATH, 'feature_extractor.pkl')
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                models['feature_extractor'] = pickle.load(f)
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        
    return models

# Analyze sample text
def analyze_text(text, models):
    """Analyze a sample text using loaded models"""
    results = {}
    
    # TF-IDF embedding
    if 'vectorizer' in models:
        vector = models['vectorizer'].transform([text])
        results['vector_shape'] = vector.shape
        
        # Get top TF-IDF terms
        feature_names = models['vectorizer'].get_feature_names_out()
        scores = zip(feature_names, vector.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        results['top_terms'] = sorted_scores[:20]
    
    # Sentiment analysis
    if 'sentiment_config' in models:
        # Simple word-based sentiment calculation
        positive_words = models['sentiment_config'].get('positive_words', [])
        negative_words = models['sentiment_config'].get('negative_words', [])
        uncertainty_words = models['sentiment_config'].get('uncertainty_words', [])
        litigious_words = models['sentiment_config'].get('litigious_words', [])
        
        # Count occurrences
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words > 0:
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            unc_count = sum(1 for w in words if w in uncertainty_words)
            lit_count = sum(1 for w in words if w in litigious_words)
            
            results['sentiment'] = {
                'positive': pos_count / total_words if total_words > 0 else 0,
                'negative': neg_count / total_words if total_words > 0 else 0,
                'uncertainty': unc_count / total_words if total_words > 0 else 0,
                'litigious': lit_count / total_words if total_words > 0 else 0,
                'net_sentiment': (pos_count - neg_count) / total_words if total_words > 0 else 0
            }
    
    # Topic modeling
    if 'topic_model' in models and 'vectorizer' in models:
        # Get topics from model
        topic_data = models['topic_model'].get('topics', {})
        results['topics'] = {
            'num_topics': models['topic_model'].get('num_topics', 0),
            'top_topics': list(topic_data.items())[:5]  # Get first 5 topics
        }
    
    return results

# Load processed data 
@st.cache_data
def load_processed_data():
    """Load the processed earnings report data"""
    try:
        # Try to load from the data versioner's latest version
        df = pd.read_csv("data/processed/train_edad7fda80.csv")
        return df
    except Exception as e:
        st.warning(f"Could not load processed data: {e}")
        # Provide sample data structure
        return pd.DataFrame({
            'File_Name': ['sample1.txt', 'sample2.txt'],
            'ea_text': [
                'The company reported strong revenue growth of 15% year over year.',
                'Due to challenging market conditions, we experienced a decrease in quarterly earnings.'
            ],
            'datacqtr': ['2024Q1', '2024Q1'],
            'BHAR0_2': [0.025, -0.018],
            'label': [0, 0]
        })

# Main function
def main():
    # Title and description
    st.title("Earnings Announcements Text Analysis")
    st.markdown("""
    This application analyzes the text of earnings announcements using Natural Language Processing 
    and Machine Learning techniques to identify topics, analyze sentiment, and predict stock returns.
    """)
    
    # Check if models exist, if not create placeholders
    check_models_exist()
    
    # Load models
    models = load_models()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a page:", 
        ["Dashboard", "Text Analysis", "Topic Explorer", "Model Zoo", "Prediction Simulator"]
    )
    
    # Dashboard Page
    if page == "Dashboard":
        st.header("NLP Earnings Report Dashboard")
        
        # Display feature importance plot if available
        fig_path = os.path.join(OUTPUT_DIR, 'figures', 'feature_importances.png')
        if os.path.exists(fig_path):
            st.subheader("Feature Importance")
            image = Image.open(fig_path)
            st.image(image, caption="Feature importance for return prediction")
        
        # Display dataset statistics
        st.subheader("Dataset Overview")
        df = load_processed_data()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Reports", f"{len(df)}")
        with col2:
            avg_return = df['BHAR0_2'].mean() * 100 if 'BHAR0_2' in df.columns else 0
            st.metric("Average Return", f"{avg_return:.2f}%")
        with col3:
            positive_pct = (df['BHAR0_2'] > 0).mean() * 100 if 'BHAR0_2' in df.columns else 50
            st.metric("Positive Returns", f"{positive_pct:.1f}%")
        
        # Display sample data
        st.subheader("Sample Data")
        st.write(df.head())
        
    # Text Analysis Page
    elif page == "Text Analysis":
        st.header("Text Analysis Tool")
        
        # Text input area
        sample_text = st.text_area(
            "Enter earnings report text to analyze:", 
            "The company reported strong quarterly results with revenue increasing by 15% year-over-year to $1.2 billion. " +
            "Net income was $240 million, up 12% from the previous year. " +
            "Despite market headwinds, we maintained strong operating margins of 22%.",
            height=200
        )
        
        # Analyze button
        if st.button("Analyze Text"):
            with st.spinner("Analyzing text..."):
                results = analyze_text(sample_text, models)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Display sentiment analysis
                if 'sentiment' in results:
                    st.write("### Sentiment Analysis")
                    sentiment = results['sentiment']
                    
                    sentiment_cols = st.columns(5)
                    with sentiment_cols[0]:
                        st.metric("Positive", f"{sentiment['positive']:.4f}")
                    with sentiment_cols[1]:
                        st.metric("Negative", f"{sentiment['negative']:.4f}")
                    with sentiment_cols[2]:
                        st.metric("Uncertainty", f"{sentiment['uncertainty']:.4f}")
                    with sentiment_cols[3]:
                        st.metric("Litigious", f"{sentiment['litigious']:.4f}")
                    with sentiment_cols[4]:
                        net_sentiment = sentiment['net_sentiment']
                        st.metric("Net Sentiment", f"{net_sentiment:.4f}", 
                                 delta="+positive" if net_sentiment > 0 else "-negative")
                
                # Display top terms
                if 'top_terms' in results:
                    st.write("### Top TF-IDF Terms")
                    top_terms_df = pd.DataFrame(results['top_terms'], columns=['Term', 'TF-IDF Score'])
                    st.write(top_terms_df)
                    
                    # Visualize top terms
                    fig, ax = plt.subplots(figsize=(10, 6))
                    terms = [term for term, _ in results['top_terms'][:10]]
                    scores = [score for _, score in results['top_terms'][:10]]
                    ax.barh(range(len(terms)), scores, align='center')
                    ax.set_yticks(range(len(terms)))
                    ax.set_yticklabels(terms)
                    ax.invert_yaxis()
                    ax.set_xlabel('TF-IDF Score')
                    ax.set_title('Top 10 TF-IDF Terms')
                    st.pyplot(fig)
                
                # Display topics
                if 'topics' in results:
                    st.write("### Topic Analysis")
                    st.write(f"Number of topics: {results['topics']['num_topics']}")
                    
                    for topic_id, terms in results['topics']['top_topics']:
                        st.write(f"**Topic {topic_id}:** " + ", ".join([term for term, _ in terms[:5]]))
    
    # Topic Explorer Page 
    elif page == "Topic Explorer":
        st.header("Topic Explorer")
        
        if 'topic_model' in models:
            topics = models['topic_model'].get('topics', {})
            
            # Topic selection
            topic_options = list(range(models['topic_model'].get('num_topics', 0)))
            selected_topic = st.selectbox("Select a topic to explore:", topic_options)
            
            if selected_topic in topics:
                # Display topic terms
                st.subheader(f"Top Words for Topic {selected_topic}")
                terms = topics[selected_topic]
                
                # Create table
                terms_df = pd.DataFrame(terms, columns=['Term', 'Weight'])
                st.write(terms_df)
                
                # Create word cloud (simplified representation)
                st.subheader("Topic Word Cloud")
                fig, ax = plt.subplots(figsize=(10, 6))
                terms_list = [term for term, _ in terms][:20]
                weights = [weight for _, weight in terms][:20]
                ax.bar(terms_list, weights)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show related documents
                st.subheader("Sample Documents with High Topic Weight")
                st.info("In a full implementation, this would show actual documents from the dataset with high weights for this topic.")
        else:
            st.warning("Topic model not found. Please run the advanced NLP pipeline first.")
    
    # Model Zoo Page
    elif page == "Model Zoo":
        st.header("Model Zoo")
        
        st.write("""
        This page provides access to pre-trained models for analyzing earnings reports.
        Select a model to see details and usage instructions.
        """)
        
        # Model selection
        model_type = st.radio(
            "Select model type:",
            ["Embedding Models", "Sentiment Models", "Topic Models", "Prediction Models"]
        )
        
        if model_type == "Embedding Models":
            st.subheader("Available Embedding Models")
            
            if 'embedding_config' in models:
                st.write("### TF-IDF Vectorizer")
                st.write(f"Method: {models['embedding_config'].get('method', 'tfidf')}")
                st.write(f"Max features: {models['embedding_config'].get('max_features', 'N/A')}")
                st.write(f"Vocabulary size: {models['embedding_config'].get('vocab_size', 'N/A')}")
                
                st.info("This model converts text into TF-IDF vectors, which measure term importance.")
            else:
                st.warning("No embedding models available. Run the advanced NLP pipeline to generate models.")
        
        elif model_type == "Sentiment Models":
            st.subheader("Available Sentiment Models")
            
            if 'sentiment_config' in models:
                st.write("### Loughran-McDonald Sentiment Analyzer")
                st.write(f"Method: {models['sentiment_config'].get('method', 'loughran_mcdonald')}")
                st.write("This model analyzes financial text sentiment using the Loughran-McDonald lexicon.")
                
                st.markdown("""
                **Features:**
                - Positive sentiment score
                - Negative sentiment score
                - Uncertainty score
                - Litigious score
                """)
            else:
                st.warning("No sentiment models available. Run the advanced NLP pipeline to generate models.")
        
        elif model_type == "Topic Models":
            st.subheader("Available Topic Models")
            
            if 'topic_model' in models:
                st.write("### Latent Dirichlet Allocation (LDA) Model")
                st.write(f"Number of topics: {models['topic_model'].get('num_topics', 'N/A')}")
                st.write(f"Coherence score: {models['topic_model'].get('coherence', 'N/A')}")
                st.write(f"Perplexity: {models['topic_model'].get('perplexity', 'N/A')}")
                
                st.markdown("""
                **Features:**
                - Topic-word distributions
                - Document-topic distributions
                - Topic coherence metrics
                """)
            else:
                st.warning("No topic models available. Run the advanced NLP pipeline to generate models.")
        
        elif model_type == "Prediction Models":
            st.subheader("Available Prediction Models")
            
            if 'feature_extractor' in models:
                st.write("### Combined Feature Regression Model")
                st.write("This model predicts stock returns based on text features.")
                
                feature_groups = models['feature_extractor'].get('feature_groups', {})
                st.write("Feature groups:")
                for group, features in feature_groups.items():
                    st.write(f"- **{group}**: {len(features)} features")
                
                st.info("To use this model for predictions, go to the Prediction Simulator page.")
            else:
                st.warning("No prediction models available. Run the advanced NLP pipeline to generate models.")
    
    # Prediction Simulator Page
    elif page == "Prediction Simulator":
        st.header("Return Prediction Simulator")
        
        st.write("""
        This tool simulates predictions of post-earnings announcement returns based on the text.
        Enter an earnings announcement text to get a predicted return.
        """)
        
        # Text input
        prediction_text = st.text_area(
            "Enter earnings announcement text:",
            "We are pleased to report a strong quarter with revenue of $2.5 billion, an increase of 18% " +
            "compared to the same period last year. Operating income was $620 million, representing a " +
            "margin of 24.8%. Net income for the quarter was $450 million, or $2.15 per diluted share.",
            height=200
        )
        
        # Simulate prediction
        if st.button("Predict Return"):
            with st.spinner("Analyzing text and predicting return..."):
                # Analyze text
                results = analyze_text(prediction_text, models)
                
                # Simulate prediction based on sentiment
                if 'sentiment' in results:
                    sentiment = results['sentiment']
                    net_sentiment = sentiment['net_sentiment']
                    
                    # Simple prediction model based on sentiment
                    predicted_return = net_sentiment * 10  # Simple multiplier for demo
                    confidence = abs(net_sentiment) * 2  # Simple confidence calculation
                    
                    # Display prediction
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Predicted Return", 
                            f"{predicted_return:.2f}%",
                            delta="positive" if predicted_return > 0 else "negative"
                        )
                    with col2:
                        st.metric("Confidence", f"{min(confidence * 100, 100):.1f}%")
                    
                    # Display explanation
                    st.subheader("Prediction Explanation")
                    
                    if predicted_return > 3:
                        st.write("**Strong Positive Prediction**: The announcement contains very positive language and likely indicates better-than-expected results.")
                    elif predicted_return > 1:
                        st.write("**Positive Prediction**: The announcement contains positive language that suggests good results.")
                    elif predicted_return > -1:
                        st.write("**Neutral Prediction**: The announcement is relatively neutral, suggesting results in line with expectations.")
                    elif predicted_return > -3:
                        st.write("**Negative Prediction**: The announcement contains negative language that suggests disappointing results.")
                    else:
                        st.write("**Strong Negative Prediction**: The announcement contains very negative language and likely indicates worse-than-expected results.")
                    
                    # Add disclaimer
                    st.info("**Disclaimer**: This is a simulated prediction for demonstration purposes only. It should not be used for actual investment decisions.")

if __name__ == "__main__":
    main()