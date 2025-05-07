"""
Streamlit dashboard application for NLP earnings report analysis.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import base64
from datetime import datetime
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dashboard')

# Add the parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import dashboard utils
from dashboard.utils import (
    load_models, 
    format_topics,
    classify_sentiment,
    format_sentiment_result,
    extract_topic_visualization,
    get_feature_importance_plot
)

# Import NLP components
from nlp.embedding import EmbeddingProcessor
from nlp.sentiment import SentimentAnalyzer
from nlp.topic_modeling import TopicModeler
from nlp.feature_extraction import FeatureExtractor

class EarningsReportDashboard:
    """
    Interactive dashboard for NLP earnings report analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.title = "NLP Earnings Report Analysis Dashboard"
        self.models = {}
        self.sample_data = None
        
        # Configure page settings
        st.set_page_config(
            page_title=self.title,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
    def initialize(self):
        """Initialize dashboard components and load models."""
        try:
            # Load models and data
            with st.spinner("Loading models and data..."):
                self.models = load_models()
                if 'sample_data' in self.models:
                    self.sample_data = self.models['sample_data']
                    del self.models['sample_data']  # Remove from models dict
            
            # Display warning if no models could be loaded
            if not self.models:
                st.warning("⚠️ No pre-trained models could be loaded. Some features may be unavailable.")
        
        except Exception as e:
            logger.error(f"Error initializing dashboard: {str(e)}")
            st.error(f"An error occurred during initialization: {str(e)}")
    
    def render_header(self):
        """Render the dashboard header."""
        st.title(self.title)
        st.markdown("""
        This interactive dashboard allows you to explore and analyze earnings reports using 
        advanced Natural Language Processing techniques. Upload your own text or use the provided
        sample data to perform sentiment analysis, topic modeling, and examine key financial metrics.
        """)
        
    def render_sidebar(self):
        """Render the sidebar with options."""
        st.sidebar.title("Options")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["Home", "Text Analysis", "Dataset Analysis", "Model Performance", "About"]
        )
        
        # Models info
        st.sidebar.subheader("Loaded Models")
        for model_name, model in self.models.items():
            st.sidebar.success(f"✓ {model_name.replace('_', ' ').title()} loaded")
        
        # Sample data info
        if self.sample_data is not None:
            st.sidebar.subheader("Sample Data")
            st.sidebar.info(f"✓ {len(self.sample_data)} samples available")
        
        # Upload option
        st.sidebar.subheader("Upload Data")
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV file with earnings report data",
            type=["csv"]
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✓ Loaded {len(data)} records")
                return page, data
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
        
        return page, None
    
    def render_text_analysis(self):
        """Render the text analysis page."""
        st.header("Text Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Text input area
            st.subheader("Input Text")
            text = st.text_area(
                "Enter earnings report text to analyze",
                height=200,
                key="text_input",
                help="Paste earnings report text here for analysis"
            )
            
            # Sample text selection
            if self.sample_data is not None and 'text' in self.sample_data.columns:
                st.subheader("Or select a sample")
                sample_idx = st.selectbox(
                    "Select a sample text",
                    range(min(10, len(self.sample_data))),
                    format_func=lambda i: f"Sample {i+1}: {self.sample_data.iloc[i]['text'][:50]}..."
                )
                
                if st.button("Use this sample"):
                    text = self.sample_data.iloc[sample_idx]['text']
                    st.session_state.text_input = text
        
        with col2:
            # Analysis options
            st.subheader("Analysis Options")
            
            run_sentiment = st.checkbox("Sentiment Analysis", value=True)
            run_topics = st.checkbox("Topic Extraction", value=True)
            run_features = st.checkbox("Feature Extraction", value=True)
            
            if st.button("Analyze Text", key="analyze_btn"):
                if not text:
                    st.error("Please enter or select some text to analyze")
                    return
                
                # Analysis results container
                with st.spinner("Analyzing text..."):
                    self._analyze_text(text, run_sentiment, run_topics, run_features)
    
    def _analyze_text(self, text, run_sentiment=True, run_topics=True, run_features=True):
        """Perform text analysis and display results."""
        st.subheader("Analysis Results")
        
        # Sentiment Analysis
        if run_sentiment and 'sentiment' in self.models:
            st.markdown("### Sentiment Analysis")
            try:
                sentiment_result = classify_sentiment(text, self.models['sentiment'])
                result_df = format_sentiment_result(sentiment_result)
                
                # Display as bar chart
                fig = px.bar(
                    result_df, 
                    x='Score', 
                    y='Dimension',
                    orientation='h',
                    color='Score',
                    color_continuous_scale='RdBu',
                    title='Sentiment Analysis Results'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Also display as table
                st.dataframe(result_df)
                
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")
        
        # Topic Modeling
        if run_topics and 'topic' in self.models:
            st.markdown("### Topic Analysis")
            try:
                # Create a corpus with just this document
                corpus = [text]
                
                # Extract topics from the text
                topics = self.models['topic'].extract_topics(corpus)
                
                if topics is not None:
                    # Display top topics for this document
                    st.subheader("Document Topics")
                    
                    # If topics is a list of (topic_id, score) tuples
                    if isinstance(topics, list) and len(topics) > 0 and isinstance(topics[0], tuple):
                        topics_df = pd.DataFrame(topics, columns=['Topic ID', 'Score'])
                        topics_df['Topic Words'] = topics_df['Topic ID'].apply(
                            lambda tid: ", ".join(self.models['topic'].get_topic_words(tid, 10))
                        )
                        
                        st.dataframe(topics_df)
                    else:
                        st.write("Topics extracted but format not recognized")
                
                # Visualize topics
                html = extract_topic_visualization(self.models['topic'])
                if html:
                    st.components.v1.html(html, height=800)
                    
            except Exception as e:
                st.error(f"Error in topic analysis: {str(e)}")
        
        # Feature Extraction
        if run_features and 'feature_extractor' in self.models:
            st.markdown("### Key Features")
            try:
                features = self.models['feature_extractor'].extract_features(text)
                
                # Create columns for feature display
                cols = st.columns(2)
                
                with cols[0]:
                    st.subheader("Extracted Features")
                    if isinstance(features, dict):
                        feature_df = pd.DataFrame(
                            {'Feature': list(features.keys()), 'Value': list(features.values())}
                        )
                        st.dataframe(feature_df)
                    elif isinstance(features, pd.DataFrame):
                        st.dataframe(features)
                    else:
                        st.write("Features extracted but format not recognized")
                
                with cols[1]:
                    # Display feature importance plot if available
                    fig = get_feature_importance_plot(self.models['feature_extractor'])
                    if fig:
                        st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error in feature extraction: {str(e)}")
    
    def render_dataset_analysis(self):
        """Render the dataset analysis page."""
        st.header("Dataset Analysis")
        
        dataset = None
        if self.sample_data is not None:
            dataset = self.sample_data
        
        if dataset is None:
            st.warning("No dataset available. Please upload data from the sidebar.")
            return
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(dataset))
        with col2:
            st.metric("Columns", len(dataset.columns))
        with col3:
            # Show most recent date if available
            date_cols = [col for col in dataset.columns if 'date' in col.lower()]
            if date_cols:
                try:
                    latest_date = pd.to_datetime(dataset[date_cols[0]]).max()
                    st.metric("Latest Date", latest_date.strftime('%Y-%m-%d'))
                except:
                    pass
        
        # Display sample of the dataset
        with st.expander("View data sample", expanded=True):
            st.dataframe(dataset.head(10))
        
        # Column selection for analysis
        text_columns = [col for col in dataset.columns if dataset[col].dtype == 'object']
        if text_columns:
            text_col = st.selectbox("Select text column for analysis", text_columns)
            
            # Word count histogram
            if st.button("Generate Word Count Distribution"):
                with st.spinner("Calculating word counts..."):
                    try:
                        dataset['word_count'] = dataset[text_col].apply(lambda x: len(str(x).split()))
                        fig = px.histogram(
                            dataset, 
                            x='word_count',
                            nbins=50,
                            title=f"Word Count Distribution in '{text_col}'"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating word count distribution: {str(e)}")
        
        # Target variable analysis if available
        numeric_cols = [col for col in dataset.columns if pd.api.types.is_numeric_dtype(dataset[col])]
        if numeric_cols:
            st.subheader("Target Variable Analysis")
            target_col = st.selectbox("Select target variable", numeric_cols)
            
            if st.button("Analyze Target Variable"):
                with st.spinner("Analyzing target variable..."):
                    try:
                        # Basic statistics
                        stats = dataset[target_col].describe()
                        st.write(stats)
                        
                        # Distribution
                        fig = px.histogram(
                            dataset, 
                            x=target_col,
                            title=f"Distribution of '{target_col}'",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error analyzing target variable: {str(e)}")
    
    def render_model_performance(self):
        """Render the model performance page."""
        st.header("Model Performance")
        
        # Select model for performance analysis
        model_options = list(self.models.keys())
        if not model_options:
            st.warning("No models available for performance analysis.")
            return
        
        selected_model = st.selectbox("Select model", model_options)
        
        if selected_model == 'topic':
            self._render_topic_model_performance()
        elif selected_model == 'sentiment':
            self._render_sentiment_model_performance()
        elif selected_model == 'feature_extractor':
            self._render_feature_extractor_performance()
        else:
            st.write(f"Performance visualization not implemented for {selected_model} model.")
    
    def _render_topic_model_performance(self):
        """Render topic model performance metrics."""
        if 'topic' not in self.models:
            st.warning("Topic model not available.")
            return
        
        st.subheader("Topic Model Performance")
        
        # Coherence score
        topic_model = self.models['topic']
        if hasattr(topic_model, 'coherence_score'):
            st.metric("Model Coherence Score", f"{topic_model.coherence_score:.4f}")
        
        # Topic distribution
        st.subheader("Topic Distribution")
        topics_df = format_topics(topic_model)
        
        if not topics_df.empty:
            st.dataframe(topics_df)
            
            # Topic word cloud or bar chart
            if hasattr(topic_model, 'get_topic_words'):
                # Create visualization for selected topic
                topic_id = st.slider(
                    "Select topic to visualize", 
                    0, 
                    max(0, topic_model.num_topics - 1), 
                    0
                )
                
                words = topic_model.get_topic_words(topic_id, 20)
                
                if isinstance(words, list):
                    # Simple list of words
                    word_data = pd.DataFrame({
                        'word': words,
                        'importance': range(len(words), 0, -1)  # Placeholder importance
                    })
                else:
                    # (word, score) tuples
                    word_data = pd.DataFrame({
                        'word': [w[0] for w in words],
                        'importance': [w[1] for w in words]
                    })
                
                fig = px.bar(
                    word_data, 
                    x='importance', 
                    y='word',
                    orientation='h',
                    title=f"Top Words for Topic {topic_id}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_sentiment_model_performance(self):
        """Render sentiment model performance metrics."""
        if 'sentiment' not in self.models:
            st.warning("Sentiment model not available.")
            return
        
        st.subheader("Sentiment Model Information")
        
        sentiment_model = self.models['sentiment']
        
        # Show model method/type
        if hasattr(sentiment_model, 'method'):
            st.write(f"**Model Type:** {sentiment_model.method}")
        
        # Show sample sentiment analysis
        st.subheader("Try Sentiment Analysis")
        sample_text = st.text_area(
            "Enter text for sentiment analysis",
            "We are pleased to report a strong quarter with revenues exceeding expectations."
        )
        
        if st.button("Analyze Sentiment"):
            result = sentiment_model.analyze(sample_text)
            
            if result:
                result_df = format_sentiment_result(result)
                
                # Display as bar chart
                fig = px.bar(
                    result_df, 
                    x='Score', 
                    y='Dimension',
                    orientation='h',
                    color='Score',
                    color_continuous_scale='RdBu',
                    title='Sentiment Analysis Results'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Also display as table
                st.dataframe(result_df)
    
    def _render_feature_extractor_performance(self):
        """Render feature extractor performance."""
        if 'feature_extractor' not in self.models:
            st.warning("Feature extractor not available.")
            return
        
        st.subheader("Feature Extractor Information")
        
        feature_extractor = self.models['feature_extractor']
        
        # Show feature importance
        st.subheader("Feature Importance")
        
        fig = get_feature_importance_plot(feature_extractor)
        if fig:
            st.pyplot(fig)
        else:
            st.write("Feature importance visualization not available.")
        
        # Show sample feature extraction
        st.subheader("Try Feature Extraction")
        sample_text = st.text_area(
            "Enter text for feature extraction",
            "Q1 results: Revenue: $10.5M (+15% YoY), EPS: $0.25 (+10%), Gross margin: 35%"
        )
        
        if st.button("Extract Features"):
            features = feature_extractor.extract_features(sample_text)
            
            if isinstance(features, dict):
                feature_df = pd.DataFrame({
                    'Feature': list(features.keys()),
                    'Value': list(features.values())
                })
                st.dataframe(feature_df)
            elif isinstance(features, pd.DataFrame):
                st.dataframe(features)
            else:
                st.write("Features extracted but format not recognized")
    
    def render_about(self):
        """Render the about page."""
        st.header("About This Dashboard")
        
        st.markdown("""
        ## NLP Earnings Report Analysis
        
        This dashboard uses advanced Natural Language Processing techniques to analyze earnings reports and
        extract valuable insights from financial text data.
        
        ### Features:
        
        * **Sentiment Analysis**: Analyze the tone and sentiment of earnings reports using financial domain-specific lexicons
        * **Topic Modeling**: Discover key themes and topics discussed in earnings reports
        * **Feature Extraction**: Extract structured data from unstructured text, including financial metrics and KPIs
        * **Dataset Analysis**: Explore patterns and trends in earnings report data
        
        ### Models:
        
        The dashboard uses multiple NLP models:
        
        * **Embedding Model**: Transforms text into numerical representations for analysis
        * **Sentiment Analysis**: Uses financial domain-specific lexicons to analyze sentiment
        * **Topic Model**: Identifies key topics and themes in text
        * **Feature Extraction**: Extracts structured information from text
        
        ### Data:
        
        The sample data includes earnings report text and associated metadata, allowing for exploration
        of financial text data and its relationship to financial performance metrics.
        """)
    
    def render_home(self):
        """Render the home page."""
        st.header("Welcome to the NLP Earnings Report Dashboard")
        
        # Quick overview
        st.markdown("""
        ### Analyze earnings reports with advanced NLP
        
        This dashboard provides tools to analyze earnings reports using:
        
        * 📊 **Sentiment Analysis**
        * 🔍 **Topic Modeling**
        * 📈 **Feature Extraction**
        
        Get started by selecting a page from the sidebar navigation.
        """)
        
        # Quick links
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Text Analysis")
            st.markdown("Analyze individual earnings report text for sentiment, topics, and key features.")
            if st.button("Go to Text Analysis"):
                st.session_state.page = "Text Analysis"
                st.experimental_rerun()
        
        with col2:
            st.subheader("Dataset Analysis")
            st.markdown("Explore patterns and trends across a dataset of earnings reports.")
            if st.button("Go to Dataset Analysis"):
                st.session_state.page = "Dataset Analysis"
                st.experimental_rerun()
                
        with col3:
            st.subheader("Model Performance")
            st.markdown("Examine the performance and output of the NLP models.")
            if st.button("Go to Model Performance"):
                st.session_state.page = "Model Performance"
                st.experimental_rerun()
    
    def run(self):
        """Run the dashboard application."""
        try:
            # Initialize models and data
            self.initialize()
            
            # Render header
            self.render_header()
            
            # Render sidebar and get selected page
            page, uploaded_data = self.render_sidebar()
            
            # If data was uploaded, use it
            if uploaded_data is not None:
                self.sample_data = uploaded_data
            
            # Render the selected page
            if page == "Home":
                self.render_home()
            elif page == "Text Analysis":
                self.render_text_analysis()
            elif page == "Dataset Analysis":
                self.render_dataset_analysis()
            elif page == "Model Performance":
                self.render_model_performance()
            elif page == "About":
                self.render_about()
            
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            st.error(f"An error occurred: {str(e)}")


def main():
    """Run the dashboard application."""
    dashboard = EarningsReportDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()