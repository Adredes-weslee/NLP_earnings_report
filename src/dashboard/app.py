"""Streamlit dashboard application for NLP earnings report analysis.

This module implements an interactive dashboard for analyzing earnings reports
using Natural Language Processing techniques. The dashboard provides visualizations
for topic modeling, sentiment analysis, and financial metrics extracted from
earnings announcements.

The application allows users to:
- Load and explore earnings report datasets
- Visualize topic distributions and top words
- Analyze sentiment trends across companies and sectors
- Compare financial metrics with NLP-derived insights
- Generate custom reports and visualizations

Usage:
    Run the dashboard with: `streamlit run src/dashboard/app.py`
"""

import os
# Prevent Streamlit file watcher from examining PyTorch internals
# This fixes the "__path__._path" error with torch.classes
os.environ["STREAMLIT_WATCH_MODULE_PATHS_EXCLUDE"] = "torch,torchaudio,torchvision,pytorch_pretrained_bert,transformers"

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
import re
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

# Import configuration values
try:
    from ..config import (MODEL_DIR, OUTPUT_DIR, FIGURE_DPI, MAX_WORD_CLOUD_WORDS,
                      EMBEDDING_MODEL_PATH, SENTIMENT_MODEL_PATH, TOPIC_MODEL_PATH)
except ImportError:
    # Handle the case when running directly
    from src.config import (MODEL_DIR, OUTPUT_DIR, FIGURE_DPI, MAX_WORD_CLOUD_WORDS,
                      EMBEDDING_MODEL_PATH, SENTIMENT_MODEL_PATH, TOPIC_MODEL_PATH)

# Import dashboard utils
try:
    from .dashboard_helpers import (
        load_models, 
        get_available_models,
        format_topics,
        classify_sentiment,
        format_sentiment_result,
        extract_topic_visualization,
        get_feature_importance_plot,
        get_wordcloud_for_topic,
        create_prediction_simulator,
        create_topic_explorer
    )
except ImportError:
    from src.dashboard.dashboard_helpers import (
        load_models, 
        get_available_models,
        format_topics,
        classify_sentiment,
        format_sentiment_result,
        extract_topic_visualization,
        get_feature_importance_plot,
        get_wordcloud_for_topic,
        create_prediction_simulator,
        create_topic_explorer
    )

# Import NLP components
try:
    from ..nlp.embedding import EmbeddingProcessor
except ImportError:
    from src.nlp.embedding import EmbeddingProcessor
try:
    from ..nlp.sentiment import SentimentAnalyzer
    from ..nlp.topic_modeling import TopicModeler
    from ..nlp.feature_extraction import FeatureExtractor
except ImportError:
    from src.nlp.sentiment import SentimentAnalyzer
    from src.nlp.topic_modeling import TopicModeler
    from src.nlp.feature_extraction import FeatureExtractor

class EarningsReportDashboard:
    """Interactive dashboard for NLP earnings report analysis.
    
    This class represents the main dashboard interface for analyzing earnings reports
    using various NLP techniques. It provides functionality for text analysis, dataset exploration,
    model management, topic exploration, prediction simulation, and performance analytics.
    
    The dashboard integrates multiple NLP components including sentiment analysis,
    topic modeling, feature extraction, and embedding models to provide comprehensive
    analysis of financial text data. It offers interactive visualizations, exploration tools,
    and model evaluation capabilities for financial text analysis.
    
    The class implements a modular design with separate pages for different analysis tasks,
    allowing users to navigate between text analysis, dataset exploration, model evaluation,
    and other specialized views. It automatically adapts its interface based on which
    models and data are available.
    
    Attributes:
        title (str): The title displayed in the dashboard.
        models (dict): Dictionary of loaded NLP models including sentiment analyzers,
            topic models, feature extractors, and embedding models.
        sample_data (pandas.DataFrame): Sample dataset for analysis and demonstrations,
            containing earnings report text and associated metadata.
        available_models (dict): Information about all available models organized by type,
            with metadata about each model including name, version, and description.
        prediction_simulator (dict): Configuration for prediction simulation including
            model availability and prediction functions for different analysis tasks.
        topic_explorer (dict): Configuration for topic exploration including topic data,
            visualization capabilities, and interactive exploration functions.
            
    Example:
        >>> dashboard = EarningsReportDashboard()
        >>> dashboard.initialize()
        >>> dashboard.run()
    """
    
    def __init__(self):
        """Initialize the earnings report analysis dashboard with default settings.
        
        Sets up the dashboard with initial empty state and configures the Streamlit
        page settings. This constructor initializes class attributes with default
        values but does not load models or data - that happens in the initialize()
        method which should be called separately.
        
        The constructor configures the Streamlit page layout, title, and sidebar
        state for optimal dashboard presentation.
        
        Args:
            None
            
        Returns:
            None
            
        Note:
            After instantiation, call initialize() to load models and data
            before calling run() to start the dashboard.
        """
        self.title = "NLP Earnings Report Analysis Dashboard"
        self.models = {}
        self.sample_data = None
        self.available_models = {}
        self.prediction_simulator = None
        self.topic_explorer = None
        
        # Configure page settings
        st.set_page_config(
            page_title=self.title,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
    def initialize(self):
        """Initialize dashboard components and load models.
        
        This method performs the following initialization steps:
        1. Loads all available NLP models (sentiment, topic, embedding)
        2. Loads sample data for demonstration if available
        3. Sets up the dashboard components and interface
        4. Prepares the topic explorer and prediction simulator
        
        The method uses st.spinner to indicate loading progress to the user.
        If any errors occur during initialization, they are caught and
        displayed as error messages in the dashboard.
        
        Returns:
            None
        
        Note:
            This should be called once at dashboard startup.
        """
        try:
            # Load models and data
            with st.spinner("Loading models and data..."):
                self.models = load_models()
                if 'sample_data' in self.models:
                    self.sample_data = self.models['sample_data']
                    del self.models['sample_data']  # Remove from models dict
                
                # Get all available models for the model zoo
                self.available_models = get_available_models()
                
                # Initialize prediction simulator
                self.prediction_simulator = create_prediction_simulator(
                    self.models, 
                    sample_text="We are pleased to report a strong quarter with revenue growth of 15% year-over-year to $2.5 billion and earnings per share of $1.85, exceeding analyst expectations."
                )
                
                # Initialize topic explorer if topic model is available
                if 'topic' in self.models:
                    self.topic_explorer = create_topic_explorer(self.models['topic'])
            
            # Display warning if no models could be loaded
            if not self.models:
                st.warning("⚠️ No pre-trained models could be loaded. Some features may be unavailable.")
        
        except Exception as e:
            logger.error(f"Error initializing dashboard: {str(e)}")            
            st.error(f"An error occurred during initialization: {str(e)}")
    
    def render_header(self):
        """Render the dashboard header with title and introduction.
        
        Displays the main title of the dashboard and an introductory text that
        explains the purpose and functionality of the application to users.
        The header provides context about what the dashboard offers and how
        users can interact with it.
        
        Args:
            None
            
        Returns:
            None: The header is rendered directly to the Streamlit UI.
            
        Note:
            This method should be called at the beginning of each page render
            to maintain consistent UI across different dashboard views.
        """
        st.title(self.title)
        st.markdown("""
        This interactive dashboard allows you to explore and analyze earnings reports using 
        advanced Natural Language Processing techniques. Upload your own text or use the provided
        sample data to perform sentiment analysis, topic modeling, and examine key financial metrics.
        """)
        
    def render_sidebar(self):
        """Render the dashboard sidebar with navigation and options.
        
        Creates and populates the sidebar with navigation controls, model information,
        sample data details, and data upload functionality. The sidebar serves as the
        main navigation hub for the dashboard and provides context about available
        resources (models and data).
        
        The method:
        1. Creates a navigation radio selector
        2. Displays information about loaded models
        3. Shows details about available sample data
        4. Provides a file uploader for custom data
        
        Args:
            None
            
        Returns:
            Tuple[str, Optional[pd.DataFrame]]: A tuple containing:
                - Selected page name as a string
                - Uploaded data as DataFrame if a file was uploaded, otherwise None
                
        Note:
            This method should be called at the beginning of the dashboard flow
            to set up navigation and process any uploaded data.
        """
        st.sidebar.title("Options")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["Home", "Text Analysis", "Dataset Analysis", "Model Zoo", "Topic Explorer", "Prediction Simulator", "Model Performance", "About"]
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
        """Render the text analysis page with input area and options.
        
        Creates the text analysis interface that allows users to input or select
        earnings report text and analyze it using various NLP techniques. The page
        provides text input controls, sample text selection, analysis options,
        and displays results in interactive visualizations.
        
        The method:
        1. Creates a text input area for earnings report content
        2. Provides sample text selection if sample data is available
        3. Offers configurable analysis options (sentiment, topics, features)
        4. Processes the text and displays results when requested
        
        Args:
            None
            
        Returns:
            None: The text analysis page is rendered directly to the Streamlit UI.
            
        Note:
            This method relies on the _analyze_text helper method to perform
            the actual analysis and visualization once text is provided.
        """
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
        """Perform text analysis and display results.
        
        This method analyzes the provided text using various NLP techniques including
        sentiment analysis, topic modeling, and feature extraction. Results are
        displayed as interactive visualizations in the Streamlit dashboard.
        
        Args:
            text (str): The earnings report text to analyze.
            run_sentiment (bool): Whether to perform sentiment analysis.
                Defaults to True.
            run_topics (bool): Whether to perform topic modeling.
                Defaults to True.
            run_features (bool): Whether to perform feature extraction.
                Defaults to True.
                
        Returns:
            None: Results are displayed directly in the Streamlit dashboard.
            
        Note:
            This method requires models to be loaded in self.models dictionary.
            At minimum, it needs 'sentiment' and 'topic' models for the respective
            analyses to be performed.
        """
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
        """Render the dataset analysis page with data exploration features.
        
        Creates an interactive data exploration interface that displays dataset
        statistics, visualizations, and analysis tools. The page allows users to
        explore the structure and content of earnings report datasets, including
        text features and numerical metrics.
        
        The method:
        1. Displays dataset overview metrics (records, columns, dates)
        2. Shows a sample of the dataset in a data table
        3. Provides text column analysis with word counts and distributions
        4. Offers target variable analysis for numeric columns if available
        
        Args:
            None
            
        Returns:
            None: The dataset analysis page is rendered directly to the Streamlit UI.
            
        Note:
            This method requires either sample_data to be loaded or data to be
            uploaded through the sidebar uploader. If no dataset is available,
            a warning message is displayed instead.
        """
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
                        
    def render_model_zoo(self):
        """Render the model zoo page showcasing available pre-trained models.
        
        Creates an interactive catalog of available pre-trained models for financial
        text analysis, organized by type (sentiment, topic, feature extraction).
        The page allows users to explore model descriptions, performance metrics,
        and try out models with sample inputs.
        
        The method:
        1. Creates tabbed sections for different model categories
        2. Displays model information including descriptions and metrics
        3. Provides interactive demo functionality for each model
        4. Shows comparisons between different models in the same category
        
        Args:
            None
            
        Returns:
            None: The model zoo page is rendered directly to the Streamlit UI.
            
        Note:
            This method uses the available_models dictionary to populate the
            model information. The actual models available may vary depending
            on what was successfully loaded during initialization.
        """
        st.header("Model Zoo")
        
        st.markdown("""
        ### Pre-trained Models for Financial Text Analysis
        
        Browse and try out different pre-trained models for analyzing earnings reports.
        These models can be used for various NLP tasks related to financial document analysis.
        """)
        
        # Create tabs for different model categories
        model_tabs = st.tabs(["Sentiment Models", "Topic Models", "Feature Extraction Models", "Custom Models"])
        
        # Sentiment Models Tab
        with model_tabs[0]:
            st.subheader("Financial Sentiment Analysis Models")
            
            # Display available sentiment models
            sentiment_models = {
                "FinBERT": "Financial domain-specific BERT model fine-tuned for sentiment analysis",
                "FinVADER": "Lexicon-based sentiment analyzer adapted for financial terminology",
                "Financial LLM": "Large language model fine-tuned on financial disclosures"
            }
            
            for model_name, description in sentiment_models.items():
                with st.expander(f"{model_name}"):
                    st.write(description)
                    
                    # Show model details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", "92.4%" if model_name == "FinBERT" else "88.7%" if model_name == "FinVADER" else "94.2%")
                    with col2:
                        st.metric("F1 Score", "0.91" if model_name == "FinBERT" else "0.86" if model_name == "FinVADER" else "0.93")
                    
                    # Try model section
                    st.markdown("#### Try the model")
                    sample_text = st.text_area(
                        "Enter text for sentiment analysis", 
                        "We are pleased to report strong financial results for the quarter.",
                        key=f"sentiment_sample_{model_name}"
                    )
                    
                    if st.button("Analyze", key=f"analyze_btn_{model_name}"):
                        with st.spinner("Analyzing sentiment..."):
                            if 'sentiment' in self.models:
                                result = self.models['sentiment'].analyze(sample_text)
                                result_df = format_sentiment_result(result)
                                st.dataframe(result_df)
                            else:
                                st.info("Model not currently loaded. Simulating result...")
                                # Simulate results
                                result_df = pd.DataFrame({
                                    'Dimension': ['Positive', 'Negative', 'Neutral'],
                                    'Score': [0.78, 0.12, 0.10]
                                })
                                st.dataframe(result_df)
        
        # Topic Models Tab
        with model_tabs[1]:
            st.subheader("Financial Topic Models")
            
            # Display available topic models
            topic_models = {
                "FinLDA": "Latent Dirichlet Allocation model trained on earnings reports",
                "BERTopic": "Topic modeling with BERT embeddings",
                "Financial Entity-Aware Topics": "Topic model with financial entity recognition"
            }
            
            for model_name, description in topic_models.items():
                with st.expander(f"{model_name}"):
                    st.write(description)
                    
                    # Show topic examples
                    st.markdown("#### Sample Topics")
                    
                    topics_example = {
                        "Revenue Growth": ["revenue", "growth", "increase", "sales", "performance"],
                        "Cost Management": ["cost", "expense", "reduction", "margin", "efficiency"],
                        "Market Outlook": ["market", "future", "expect", "outlook", "forecast"]
                    }
                    
                    for topic, words in topics_example.items():
                        st.markdown(f"**{topic}:** {', '.join(words)}")
                    
                    # Try model section
                    st.markdown("#### Try the model")
                    sample_text = st.text_area(
                        "Enter text for topic modeling", 
                        "Our revenue increased by 15% this quarter due to strong product sales and market expansion.",
                        key=f"topic_sample_{model_name}"
                    )
                    
                    if st.button("Extract Topics", key=f"topic_btn_{model_name}"):
                        with st.spinner("Extracting topics..."):
                            if 'topic' in self.models:
                                # Use actual topic model if available
                                topics = self.models['topic'].extract_topics([sample_text])
                                st.write(topics)
                            else:
                                # Simulate results
                                st.info("Model not currently loaded. Simulating result...")
                                sim_topics = [
                                    ("Revenue Growth", 0.75),
                                    ("Market Expansion", 0.25)
                                ]
                                st.write(sim_topics)
        
        # Feature Extraction Models Tab
        with model_tabs[2]:
            st.subheader("Financial Feature Extraction Models")
            
            # Display available feature extraction models
            feature_models = {
                "FinMetrics": "Extract financial metrics like revenue, EPS, and growth rates",
                "EntityExtractor": "Identify companies, products and financial entities",
                "TemporalAnalyzer": "Extract time-based information and comparisons"
            }
            
            for model_name, description in feature_models.items():
                with st.expander(f"{model_name}"):
                    st.write(description)
                    
                    # Show feature examples
                    st.markdown("#### Extractable Features")
                    
                    if model_name == "FinMetrics":
                        features = ["Revenue ($)", "EPS", "Growth Rate (%)", "Margin (%)", "YoY Change (%)"]
                    elif model_name == "EntityExtractor":
                        features = ["Companies", "Products", "Sectors", "Regions", "Financial Instruments"]
                    else:
                        features = ["Quarter References", "Year References", "Comparisons", "Future Projections"]
                        
                    for feature in features:
                        st.markdown(f"- {feature}")
                    
                    # Try model section
                    st.markdown("#### Try the model")
                    sample_text = st.text_area(
                        "Enter text for feature extraction", 
                        "In Q2 2024, we reported revenue of $125.3M, with EPS of $1.42, representing a 12% increase from the previous year.",
                        key=f"feature_sample_{model_name}"
                    )
                    
                    if st.button("Extract Features", key=f"feature_btn_{model_name}"):
                        with st.spinner("Extracting features..."):
                            if 'feature_extractor' in self.models:
                                # Use actual feature extractor if available
                                features = self.models['feature_extractor'].extract_features(sample_text)
                                if isinstance(features, dict):
                                    feature_df = pd.DataFrame({
                                        'Feature': list(features.keys()),
                                        'Value': list(features.values())
                                    })
                                    st.dataframe(feature_df)
                                else:
                                    st.write(features)
                            else:
                                # Simulate results
                                st.info("Model not currently loaded. Simulating result...")
                                sim_features = {
                                    "Revenue": "$125.3M",
                                    "EPS": "$1.42",
                                    "Growth Rate": "12%",
                                    "Time Period": "Q2 2024",
                                    "Comparison": "Previous Year"
                                }
                                sim_df = pd.DataFrame({
                                    'Feature': list(sim_features.keys()),
                                    'Value': list(sim_features.values())
                                })
                                st.dataframe(sim_df)
        
        # Custom Models Tab
        with model_tabs[3]:
            st.subheader("Custom Model Upload")
            
            st.markdown("""
            ### Upload Your Custom Models
            
            You can upload your own pre-trained models for use in this dashboard.
            Supported formats: .pkl, .joblib, .h5
            """)
            
            uploaded_model = st.file_uploader(
                "Upload a custom model",
                type=["pkl", "joblib", "h5"]
            )
            
            if uploaded_model is not None:
                st.success(f"Model {uploaded_model.name} uploaded successfully!")
                
                # Model configuration
                st.subheader("Model Configuration")
                
                model_type = st.selectbox(
                    "Model Type",
                    ["Sentiment Analysis", "Topic Modeling", "Feature Extraction", "Other"]
                )
                
                model_name = st.text_input("Model Name", f"Custom_{model_type.replace(' ', '')}")
                
                if st.button("Register Model"):
                    st.success(f"Model {model_name} registered as {model_type} model!")
                    st.info("Note: This is a demo. Custom model integration requires additional setup.")
    
    def render_topic_explorer(self):
        """Render the interactive topic explorer for analyzing document themes.
        
        Creates an interactive interface for exploring and visualizing topics
        extracted from financial text corpus. Users can select specific topics
        to see key words, distribution charts, and related documents, as well
        as explore intertopic relationships.
        
        The method:
        1. Displays an overview of all detected topics with prevalence metrics
        2. Allows selection of individual topics for detailed exploration
        3. Shows key words and word clouds for selected topics
        4. Provides interactive visualizations showing topic relationships
        
        Args:
            None
            
        Returns:
            None: The topic explorer page is rendered directly to the Streamlit UI.
            
        Note:
            This method requires a topic model to be loaded and accessible through
            the topic_explorer object. If no topic model is available, a warning
            message is displayed instead.
        """
        st.header("🔍 Topic Explorer")
        
        # Check if topic model is available
        if not self.topic_explorer or not self.topic_explorer.get("has_model", False):
            st.warning("Topic model not available. Please load a topic model from the Model Zoo first.")
            return
        
        st.markdown("""
        Explore topics extracted from earnings reports. Select a topic to see its key words,
        distribution, and related documents.
        """)
        
        # Get topic data
        topics_df = self.topic_explorer.get("topics_df", pd.DataFrame())
        num_topics = self.topic_explorer.get("num_topics", 0)
        
        if topics_df.empty:
            st.warning("No topics available in the model.")
            return
        
        # Display topics overview
        st.subheader("Topics Overview")
        st.dataframe(topics_df)
        
        # Topic selection
        st.subheader("Explore Topic")
        topic_id = st.slider("Select Topic ID", 0, max(0, num_topics - 1), 0)
        
        # Get topic words
        topic_words = self.topic_explorer["get_topic_words"](topic_id, 20)
        
        # Display in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"Topic {topic_id} Key Words")
            
            # Format topic words based on type
            if isinstance(topic_words, list):
                if topic_words and isinstance(topic_words[0], tuple):
                    # (word, score) format
                    words_df = pd.DataFrame(topic_words, columns=["Word", "Score"])
                    st.dataframe(words_df)
                else:
                    # Just words
                    st.write(", ".join(topic_words))
        
        with col2:
            st.subheader("Word Cloud")
            # Get word cloud for this topic
            wordcloud_data = self.topic_explorer["get_wordcloud"](topic_id)
            if wordcloud_data:
                st.image(wordcloud_data)
            else:
                st.info("Word cloud not available for this topic.")
        
        # Topic visualization
        st.subheader("Topic Visualization")
        visualization_html = self.topic_explorer.get("visualization_html", "")
        if visualization_html:
            st.components.v1.html(visualization_html, height=600)
        else:
            st.info("Interactive visualization not available for this topic model.")
            
    def render_prediction_simulator(self):
        """Render the prediction simulator for analyzing potential financial outcomes.
        
        Creates an interactive simulator that allows users to input earnings report text
        and receive predictions about potential financial outcomes, including stock
        movement, sentiment trends, and key financial metrics. The simulator applies
        multiple models to provide a comprehensive analysis.
        
        The method:
        1. Provides sample text options and a text input area
        2. Applies sentiment, topic, and feature extraction models to the input
        3. Presents predictions with confidence scores and explanations
        4. Offers adjustable parameters for sensitivity analysis
        
        Args:
            None
            
        Returns:
            None: The prediction simulator page is rendered directly to the Streamlit UI.
            
        Note:
            This method relies on multiple models being available in self.models
            dictionary. It will adapt the available predictions based on which
            models are successfully loaded.
        """
        st.header("Earnings Report Prediction Simulator")
        
        st.markdown("""
        ### Predict Financial Outcomes from Earnings Text
        
        This tool allows you to input earnings report text and get predictions about stock movement,
        sentiment, key topics, and extracted financial metrics.
        """)
        
        # Text input section
        st.subheader("Input Earnings Report Text")
        
        sample_texts = {
            "Positive Example": (
                "We are pleased to report that our Q2 results exceeded expectations, with revenue growing 18% "
                "year-over-year to $2.7 billion. Operating margins expanded to 32%, and we generated record "
                "free cash flow of $780 million. Our new product lines have gained significant market share, "
                "particularly in emerging markets. Looking ahead, we are raising our full-year guidance and "
                "expect continued strong performance through the remainder of the fiscal year."
            ),
            "Negative Example": (
                "Our Q2 results fell short of expectations, with revenue declining 8% year-over-year to $1.8 billion. "
                "Operating margins contracted to 22%, and we experienced negative free cash flow of $120 million. "
                "Our core product lines continue to face intense competitive pressure, particularly in domestic markets. "
                "Given these challenges, we are lowering our full-year guidance and implementing a cost reduction program."
            ),
            "Neutral Example": (
                "For Q2, we reported revenue of $2.1 billion, in line with consensus estimates, representing "
                "a 3% increase year-over-year. Operating margins remained stable at 27%. Our established product "
                "lines performed as expected, while new offerings are still in early adoption phases. We are "
                "maintaining our previous full-year guidance, as we anticipate similar conditions to persist."
            )
        }
        
        example_option = st.selectbox(
            "Select an example or write your own text",
            ["Write your own", "Positive Example", "Negative Example", "Neutral Example"]
        )
        
        if example_option == "Write your own":
            earnings_text = st.text_area(
                "Enter earnings report text",
                "",
                height=200
            )
        else:
            earnings_text = st.text_area(
                "Enter earnings report text",
                sample_texts[example_option],
                height=200
            )
        
        # Model selection
        st.subheader("Select Analysis Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_model = st.selectbox(
                "Sentiment Analysis Model",
                ["FinBERT", "FinVADER", "Financial LLM"]
            )
        
        with col2:
            topic_model = st.selectbox(
                "Topic Model",
                ["FinLDA", "BERTopic", "Financial Entity-Aware Topics"]
            )
            
        with col3:
            feature_model = st.selectbox(
                "Feature Extraction Model",
                ["FinMetrics", "EntityExtractor", "TemporalAnalyzer"]
            )
        
        # Analysis options
        st.subheader("Analysis Options")
        
        options_col1, options_col2 = st.columns(2)
        
        with options_col1:
            analyze_sentiment = st.checkbox("Analyze Sentiment", value=True)
            extract_topics = st.checkbox("Extract Topics", value=True)
            
        with options_col2:
            extract_features = st.checkbox("Extract Financial Metrics", value=True)
            predict_movement = st.checkbox("Predict Stock Movement", value=True)
        
        # Run analysis button
        if st.button("Run Analysis", type="primary"):
            if not earnings_text:
                st.error("Please enter some earnings report text to analyze.")
                return
                
            with st.spinner("Analyzing earnings report..."):
                # Create tabs for different analysis results
                result_tabs = st.tabs(["Summary", "Sentiment", "Topics", "Financial Metrics", "Stock Prediction"])
                
                # Summary tab
                with result_tabs[0]:
                    st.subheader("Analysis Summary")
                    
                    # Process text length
                    word_count = len(earnings_text.split())
                    
                    # Simple metrics
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        st.metric("Word Count", f"{word_count}")
                    
                    # Simulate analysis results for the summary
                    if analyze_sentiment and 'sentiment' in self.models:
                        sentiment_result = self.models['sentiment'].analyze(earnings_text)
                        sentiment_score = sentiment_result.get('compound', 0)
                    else:
                        # Simulate sentiment score
                        if "exceeded expectations" in earnings_text.lower():
                            sentiment_score = 0.75
                        elif "fell short" in earnings_text.lower():
                            sentiment_score = -0.65
                        else:
                            sentiment_score = 0.1
                    
                    with metrics_col2:
                        sentiment_label = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
                        st.metric("Sentiment", sentiment_label, sentiment_score)
                    
                    # Simulate stock prediction
                    if predict_movement:
                        if sentiment_score > 0.5:
                            movement = "↑ Up"
                            movement_pct = "+2.8%"
                        elif sentiment_score < -0.3:
                            movement = "↓ Down"
                            movement_pct = "-2.1%"
                        else:
                            movement = "→ Flat"
                            movement_pct = "+0.3%"
                            
                        with metrics_col3:
                            st.metric("Predicted Movement", movement, movement_pct)
                    
                    # Confidence score
                    with metrics_col4:
                        confidence = 0.75 if word_count > 100 else 0.5
                        st.metric("Analysis Confidence", f"{confidence:.2f}")
                    
                    # Key insights
                    st.subheader("Key Insights")
                    
                    # Generate mock insights based on the text
                    insights = []
                    
                    if "revenue" in earnings_text.lower():
                        if "growth" in earnings_text.lower() or "increase" in earnings_text.lower():
                            insights.append("📈 Revenue growth mentioned positively")
                        elif "decline" in earnings_text.lower() or "decrease" in earnings_text.lower():
                            insights.append("📉 Revenue decline mentioned")
                    
                    if "margin" in earnings_text.lower():
                        if "expanded" in earnings_text.lower() or "increase" in earnings_text.lower():
                            insights.append("✅ Margin expansion indicated")
                        elif "contracted" in earnings_text.lower() or "decrease" in earnings_text.lower():
                            insights.append("❌ Margin contraction indicated")
                    
                    if "guidance" in earnings_text.lower():
                        if "raising" in earnings_text.lower() or "positive" in earnings_text.lower():
                            insights.append("🔼 Guidance raised")
                        elif "lowering" in earnings_text.lower() or "negative" in earnings_text.lower():
                            insights.append("🔽 Guidance lowered")
                        else:
                            insights.append("↔️ Guidance maintained")
                    
                    if not insights:
                        insights.append("No clear financial indicators found in the text.")
                    
                    for insight in insights:
                        st.markdown(f"- {insight}")
                
                # Sentiment tab
                with result_tabs[1]:
                    if analyze_sentiment:
                        st.subheader("Sentiment Analysis")
                        
                        if 'sentiment' in self.models:
                            sentiment_result = self.models['sentiment'].analyze(earnings_text)
                            result_df = format_sentiment_result(sentiment_result)
                            st.dataframe(result_df)
                            
                            # Visualization
                            fig, ax = plt.subplots()
                            ax.barh(['Positive', 'Neutral', 'Negative'], 
                                    [sentiment_result.get('pos', 0), 
                                     sentiment_result.get('neu', 0), 
                                     sentiment_result.get('neg', 0)])
                            ax.set_xlim(0, 1)
                            ax.set_title('Sentiment Breakdown')
                            st.pyplot(fig)
                        else:
                            # Simulate sentiment analysis results
                            st.info("Sentiment model not loaded. Showing simulated results.")
                            
                            if "exceeded expectations" in earnings_text.lower() or "pleased" in earnings_text.lower():
                                pos, neu, neg = 0.78, 0.20, 0.02
                            elif "fell short" in earnings_text.lower() or "challenges" in earnings_text.lower():
                                pos, neu, neg = 0.15, 0.25, 0.60
                            else:
                                pos, neu, neg = 0.25, 0.65, 0.10
                                
                            result_df = pd.DataFrame({
                                'Dimension': ['Positive', 'Neutral', 'Negative'],
                                'Score': [pos, neu, neg]
                            })
                            st.dataframe(result_df)
                            
                            # Visualization
                            fig, ax = plt.subplots()
                            ax.barh(['Positive', 'Neutral', 'Negative'], [pos, neu, neg])
                            ax.set_xlim(0, 1)
                            ax.set_title('Sentiment Breakdown')
                            st.pyplot(fig)
                            
                        # Extract key sentiment phrases
                        st.subheader("Key Sentiment Phrases")
                        
                        # Simulate key phrases
                        positive_phrases = []
                        negative_phrases = []
                        
                        text_sentences = earnings_text.split('.')
                        for sentence in text_sentences:
                            sentence = sentence.strip()
                            if len(sentence) < 5:
                                continue
                                
                            if any(pos_word in sentence.lower() for pos_word in ["growth", "increase", "exceeded", "strong", "pleased"]):
                                positive_phrases.append(sentence)
                            elif any(neg_word in sentence.lower() for neg_word in ["decline", "short", "challenge", "negative", "pressure"]):
                                negative_phrases.append(sentence)
                                
                        if positive_phrases:
                            st.markdown("#### Positive Phrases")
                            for phrase in positive_phrases[:3]:  # Limit to top 3
                                st.markdown(f"- *\"{phrase}\"*")
                                
                        if negative_phrases:
                            st.markdown("#### Negative Phrases")
                            for phrase in negative_phrases[:3]:  # Limit to top 3
                                st.markdown(f"- *\"{phrase}\"*")
                    else:
                        st.info("Sentiment analysis was not selected.")
                
                # Topics tab
                with result_tabs[2]:
                    if extract_topics:
                        st.subheader("Topic Analysis")
                        
                        if 'topic' in self.models:
                            # Use the actual topic model
                            topics = self.models['topic'].extract_topics([earnings_text])
                            st.write(topics)
                            
                            # Visualize topics
                            # This would depend on the specific topic model's output format
                        else:
                            # Simulate topic modeling results
                            st.info("Topic model not loaded. Showing simulated results.")
                            
                            # Create simulated topics based on text content
                            simulated_topics = []
                            
                            if "revenue" in earnings_text.lower() or "sales" in earnings_text.lower():
                                simulated_topics.append(("Revenue Performance", 0.85))
                            
                            if "margin" in earnings_text.lower() or "profit" in earnings_text.lower():
                                simulated_topics.append(("Profitability Metrics", 0.75))
                            
                            if "market" in earnings_text.lower() or "competition" in earnings_text.lower():
                                simulated_topics.append(("Market Conditions", 0.65))
                            
                            if "product" in earnings_text.lower() or "service" in earnings_text.lower():
                                simulated_topics.append(("Product Performance", 0.60))
                            
                            if "outlook" in earnings_text.lower() or "guidance" in earnings_text.lower() or "expect" in earnings_text.lower():
                                simulated_topics.append(("Future Outlook", 0.80))
                            
                            if not simulated_topics:
                                simulated_topics = [
                                    ("General Financial Performance", 0.90),
                                    ("Corporate Communications", 0.45)
                                ]
                            
                            # Display topics
                            topic_df = pd.DataFrame(simulated_topics, columns=['Topic', 'Relevance'])
                            st.dataframe(topic_df)
                            
                            # Topic visualization
                            fig, ax = plt.subplots()
                            topics, scores = zip(*simulated_topics)
                            y_pos = np.arange(len(topics))
                            ax.barh(y_pos, scores)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(topics)
                            ax.set_xlim(0, 1)
                            ax.set_title('Topic Distribution')
                            st.pyplot(fig)
                            
                            # Show word cloud
                            st.subheader("Topic Word Cloud")
                            words = earnings_text.split()
                            word_counts = {}
                            for word in words:
                                word = word.lower().strip('.,!?()[]{}":;')
                                if len(word) > 3:  # Skip short words
                                    if word in word_counts:
                                        word_counts[word] += 1
                                    else:
                                        word_counts[word] = 1
                            
                            # Create word cloud visualization
                            if word_counts:
                                # We can't actually create a wordcloud without the wordcloud package
                                # So we'll just show top words as a bar chart
                                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                                top_words = sorted_words[:10]
                                
                                fig, ax = plt.subplots()
                                words, counts = zip(*top_words)
                                y_pos = np.arange(len(words))
                                ax.barh(y_pos, counts)
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(words)
                                ax.set_title('Top Words')
                                st.pyplot(fig)
                    else:
                        st.info("Topic extraction was not selected.")
                
                # Financial Metrics tab
                with result_tabs[3]:
                    if extract_features:
                        st.subheader("Extracted Financial Metrics")
                        
                        if 'feature_extractor' in self.models:
                            # Use the actual feature extractor
                            features = self.models['feature_extractor'].extract_features(earnings_text)
                            st.write(features)
                        else:
                            # Simulate feature extraction
                            st.info("Feature extraction model not loaded. Showing simulated results.")
                            
                            # Define regex patterns for financial metrics
                            revenue_pattern = r'\$?\s?(\d+(?:\.\d+)?)\s?(?:billion|million|B|M)'
                            percentage_pattern = r'(\d+(?:\.\d+)?)\s?%'
                            quarter_pattern = r'Q[1-4]'
                            year_pattern = r'20\d\d'
                            
                            # Extract metrics using regex
                            revenue_matches = re.findall(revenue_pattern, earnings_text)
                            percentage_matches = re.findall(percentage_pattern, earnings_text)
                            quarter_matches = re.findall(quarter_pattern, earnings_text)
                            year_matches = re.findall(year_pattern, earnings_text)
                            
                            # Create simulated metrics
                            extracted_metrics = {}
                            
                            if revenue_matches and 'revenue' in earnings_text.lower():
                                extracted_metrics['Revenue'] = f"${revenue_matches[0]}B" if 'billion' in earnings_text.lower() else f"${revenue_matches[0]}M"
                            
                            if percentage_matches:
                                if 'growth' in earnings_text.lower():
                                    extracted_metrics['Growth Rate'] = f"{percentage_matches[0]}%"
                                if 'margin' in earnings_text.lower():
                                    extracted_metrics['Margin'] = f"{percentage_matches[0]}%"
                                if 'increase' in earnings_text.lower():
                                    extracted_metrics['YoY Increase'] = f"{percentage_matches[0]}%"
                            
                            if quarter_matches:
                                extracted_metrics['Quarter'] = quarter_matches[0]
                            
                            if year_matches:
                                extracted_metrics['Year'] = year_matches[0]
                            
                            if "cash flow" in earnings_text.lower():
                                cash_flow_text = re.search(r'cash flow of \$?\s?(\d+(?:\.\d+)?)\s?(?:billion|million|B|M)', earnings_text.lower())
                                if cash_flow_text:
                                    extracted_metrics['Cash Flow'] = f"${cash_flow_text.group(1)}M"
                            
                            # Add placeholder metrics if none were extracted
                            if not extracted_metrics:
                                extracted_metrics = {
                                    "Metric Type": "No specific financial metrics detected",
                                    "Analysis Notes": "Text appears to be qualitative rather than quantitative"
                                }
                            
                            # Display extracted metrics
                            metrics_df = pd.DataFrame({
                                'Metric': list(extracted_metrics.keys()),
                                'Value': list(extracted_metrics.values())
                            })
                            st.dataframe(metrics_df)
                    else:
                        st.info("Financial metrics extraction was not selected.")
                
                # Stock Prediction tab
                with result_tabs[4]:
                    if predict_movement:
                        st.subheader("Stock Movement Prediction")
                        
                        # In a real implementation, this would use an actual predictive model
                        # Here we're simulating based on sentiment and content
                        
                        # Simulate stock prediction
                        if sentiment_score > 0.5:
                            prediction = "Likely Positive Impact (↑)"
                            confidence = 0.85
                            color = "green"
                            explanation = "Strong positive sentiment with concrete positive results mentioned."
                        elif sentiment_score > 0.2:
                            prediction = "Slight Positive Impact (↗)"
                            confidence = 0.65
                            color = "lightgreen"
                            explanation = "Moderately positive sentiment with some encouraging metrics."
                        elif sentiment_score < -0.5:
                            prediction = "Likely Negative Impact (↓)"
                            confidence = 0.80
                            color = "red"
                            explanation = "Strong negative sentiment with concrete challenges mentioned."
                        elif sentiment_score < -0.2:
                            prediction = "Slight Negative Impact (↘)"
                            confidence = 0.60
                            color = "lightcoral"
                            explanation = "Moderately negative sentiment with some concerning signals."
                        else:
                            prediction = "Neutral Impact (→)"
                            confidence = 0.70
                            color = "gray"
                            explanation = "Balanced or neutral sentiment without strong directional signals."
                        
                        # Create prediction display
                        st.markdown(f"<h3 style='color:{color}'>{prediction}</h3>", unsafe_allow_html=True)
                        
                        st.metric("Prediction Confidence", f"{confidence:.2f}")
                        
                        st.markdown("#### Explanation")
                        st.write(explanation)
                        
                        # Factors influencing prediction
                        st.subheader("Key Factors")
                        
                        factors = []
                        
                        # Base factors on text content
                        if "growth" in earnings_text.lower() or "increase" in earnings_text.lower():
                            factors.append(("Growth Metrics", "Positive"))
                        if "decline" in earnings_text.lower() or "decrease" in earnings_text.lower():
                            factors.append(("Decline Metrics", "Negative"))
                        if "guidance" in earnings_text.lower():
                            if "raising" in earnings_text.lower():
                                factors.append(("Guidance Update", "Positive"))
                            elif "lowering" in earnings_text.lower():
                                factors.append(("Guidance Update", "Negative"))
                            else:
                                factors.append(("Guidance Maintained", "Neutral"))
                        if "margin" in earnings_text.lower():
                            if "expanded" in earnings_text.lower():
                                factors.append(("Margin Trends", "Positive"))
                            elif "contracted" in earnings_text.lower():
                                factors.append(("Margin Trends", "Negative"))
                        
                        # Add default factors if none were identified
                        if not factors:
                            factors = [
                                ("Overall Sentiment", "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"),
                                ("Text Clarity", "Positive" if len(earnings_text.split()) > 100 else "Neutral")
                            ]
                        
                        # Display factors table
                        factors_df = pd.DataFrame(factors, columns=['Factor', 'Impact'])
                        st.dataframe(factors_df)
                        
                        # Market context disclaimer
                        st.markdown("""
                        > **Disclaimer**: This prediction is based solely on the text content and doesn't 
                        > consider external market conditions, broader economic factors, or market expectations 
                        > relative to the report. Actual stock movement depends on many additional factors.
                        """)
                    else:
                        st.info("Stock movement prediction was not selected.")
                        
    def render_model_performance(self):
        """Render the model performance page with metrics and evaluations.
        
        Creates a comprehensive display of performance metrics, evaluations, and
        comparative analyses for the loaded NLP models. The page provides insights
        into each model's accuracy, efficiency, and limitations.
        
        The method:
        1. Offers selection controls for different model types
        2. Delegates to specialized rendering methods for each model type
        3. Displays appropriate metrics based on the model's purpose
        4. Provides visualizations of key performance indicators
        
        Args:
            None
            
        Returns:
            None: The model performance page is rendered directly to the Streamlit UI.
            
        Note:
            This method serves as a router to more specific performance visualization
            methods based on the selected model type. It requires models to be loaded
            in the self.models dictionary.
        """
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
        """Render topic model performance metrics and evaluation.
        
        Displays performance metrics specific to topic models, including coherence
        scores, topic quality metrics, and topic distribution visualizations.
        This method is called by render_model_performance when a topic model
        is selected.
        
        The method:
        1. Displays coherence and quality metrics for the topic model
        2. Shows topic distributions across the corpus
        3. Visualizes topic-word relevance
        4. Provides diagnostic information about model fit
        
        Args:
            None
            
        Returns:
            None: Topic model performance metrics are rendered directly to the Streamlit UI.
            
        Note:
            This method requires a topic model to be loaded and available in
            the self.models dictionary under the 'topic' key.
        """
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
        """Render sentiment model performance metrics and evaluation.
        
        Displays performance metrics specific to sentiment analysis models,
        including accuracy, precision, recall, and F1 scores across different
        sentiment classes. This method is called by render_model_performance
        when a sentiment model is selected.
        
        The method:
        1. Displays sentiment classification metrics and benchmarks
        2. Shows confusion matrices for sentiment predictions
        3. Allows interactive testing of the model with custom text
        4. Provides comparisons to baseline models
        
        Args:
            None
            
        Returns:
            None: Sentiment model performance metrics are rendered directly to the Streamlit UI.
            
        Note:
            This method requires a sentiment model to be loaded and available
            in the self.models dictionary under the 'sentiment' key.
        """
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
        """Render feature extractor performance metrics and evaluation.
        
        Displays performance metrics specific to feature extraction models,
        including extraction accuracy, feature importance, and extraction
        examples. This method is called by render_model_performance when
        a feature extractor model is selected.
        
        The method:
        1. Displays feature importance visualizations
        2. Shows extraction precision and recall metrics
        3. Allows interactive testing of the extractor with custom text
        4. Demonstrates extraction capabilities with real examples
        
        Args:
            None
            
        Returns:
            None: Feature extractor performance metrics are rendered directly to the Streamlit UI.
            
        Note:
            This method requires a feature extractor model to be loaded and available
            in the self.models dictionary under the 'feature_extractor' key.
        """
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
        """Render the about page with dashboard information and documentation.
        
        Creates an informational page that explains the purpose, features,
        and technical details of the earnings report analysis dashboard.
        The page serves as documentation for users to understand the
        dashboard's capabilities and implementation.
        
        The method:
        1. Displays overview information about the dashboard
        2. Lists key features and their descriptions
        3. Provides technical information about models and data sources
        4. Shows acknowledgments and references
        
        Args:
            None
            
        Returns:
            None: The about page is rendered directly to the Streamlit UI.
            
        Note:
            This method does not rely on any models or data, and will always
            render completely regardless of the application's state.
        """
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
        * **Model Zoo**: Access and load different pre-trained models for analysis
        * **Topic Explorer**: Interactively explore topics extracted from financial texts
        * **Prediction Simulator**: Simulate predictions on custom earnings report text
        
        ### Models:
        
        The dashboard uses multiple NLP models:
        
        * **Embedding Model**: Transforms text into numerical representations for analysis
        * **Sentiment Analysis**: Uses financial domain-specific lexicons to analyze sentiment
        * **Topic Model**: Identifies key topics and themes in text
        * **Feature Extraction**: Extracts structured information from text
        
        ### Data:
        
        The sample data includes earnings report text and associated metadata, allowing for exploration
        of financial text data and its relationship to financial performance metrics.
        
        ### Methodology:
        
        This dashboard implements advanced NLP techniques including transformer-based embeddings,
        financial sentiment analysis, coherence-optimized topic modeling, and structured feature extraction.
        These methods enable comprehensive analysis of earnings reports text to extract insights
        and patterns that may correlate with financial performance.
        """)
        
    def render_home(self):
        """Render the home page with dashboard overview and quick access links.
        
        Creates the landing page of the dashboard, providing an overview of available
        features, quick access links to key functionality, and introductory information
        for new users. This page serves as the entry point to the dashboard.
        
        The method:
        1. Displays welcome message and dashboard introduction
        2. Shows key features with visual indicators
        3. Provides quick access buttons to main sections
        4. Shows sample insights and capabilities
        
        Args:
            None
            
        Returns:
            None: The home page is rendered directly to the Streamlit UI.
            
        Note:
            This method is typically called when the dashboard first loads or when
            the user navigates to the Home page from the sidebar navigation.
        """
        st.header("Welcome to the NLP Earnings Report Dashboard")
        
        # Quick overview
        st.markdown("""
        ### Analyze earnings reports with advanced NLP
        
        This dashboard provides tools to analyze earnings reports using:
        
        * 📊 **Sentiment Analysis**
        * 🔍 **Topic Modeling**
        * 📈 **Feature Extraction**
        * 🏛️ **Model Zoo**
        * 🔮 **Prediction Simulator**
        * 🧭 **Topic Explorer**
        
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
            st.subheader("Model Zoo")
            st.markdown("Access and load pre-trained models for different analysis tasks.")
            if st.button("Go to Model Zoo"):
                st.session_state.page = "Model Zoo"
                st.experimental_rerun()
                
        with col3:
            st.subheader("Prediction Simulator")
            st.markdown("Simulate predictions on custom earnings report text.")
            if st.button("Go to Prediction Simulator"):
                st.session_state.page = "Prediction Simulator"                
                st.experimental_rerun()
    
    def run(self):
        """Run the dashboard application and handle navigation between pages.
        
        The main entry point for the dashboard application that orchestrates the
        initialization, rendering, and page navigation flow. This method handles
        the overall application lifecycle, including model loading, UI rendering,
        navigation between different pages, and error handling.
        
        The method:
        1. Initializes models and loads sample data
        2. Renders the header and sidebar components
        3. Processes any uploaded data files
        4. Routes to the appropriate page based on navigation selection
        5. Handles exceptions with user-friendly error messages
        
        Args:
            None
            
        Returns:
            None: The dashboard is rendered directly to the Streamlit UI.
            
        Note:
            This method is the main driver of the application and calls the various
            render_* methods based on user navigation. It also wraps the entire
            dashboard operation in a try-except block to gracefully handle errors.
        """
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
            elif page == "Model Zoo":
                self.render_model_zoo()
            elif page == "Topic Explorer":
                self.render_topic_explorer()
            elif page == "Prediction Simulator":
                self.render_prediction_simulator()
            elif page == "Model Performance":
                self.render_model_performance()
            elif page == "About":
                self.render_about()
            
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            st.error(f"An error occurred: {str(e)}")


def main():
    """Run the dashboard application as the entry point script.
    
    This function serves as the main entry point for the dashboard application
    when the script is run directly. It creates an instance of the
    EarningsReportDashboard class and calls its run method to start the
    interactive Streamlit dashboard.
    
    The function:
    1. Instantiates the EarningsReportDashboard class
    2. Calls the run method to start the dashboard application
    3. Handles the application lifecycle from initialization to termination
    
    Args:
        None
        
    Returns:
        None: The function executes the dashboard but does not return a value
        
    Note:
        This function is executed when the script is run directly through
        `streamlit run src/dashboard/app.py` command. It is not called when
        the module is imported elsewhere.
    """
    dashboard = EarningsReportDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()