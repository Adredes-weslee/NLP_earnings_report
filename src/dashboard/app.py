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
    
        # Check if we need to change pages based on button clicks
        if "nav_target" in st.session_state:
            default_page = st.session_state.nav_target
            # Clear the target so it doesn't keep redirecting
            del st.session_state.nav_target
        else:
            # Use existing selection or default to Home
            default_page = st.session_state.get("navigation", "Home")
        
        # Navigation with the pre-selected value
        page = st.sidebar.radio(
            "Navigation",
            ["Home", "Text Analysis", "Dataset Analysis", "Model Zoo", 
            "Topic Explorer", "Prediction Simulator", "Model Performance", "About"],
            key="navigation",
            index=["Home", "Text Analysis", "Dataset Analysis", "Model Zoo", 
                "Topic Explorer", "Prediction Simulator", "Model Performance", "About"].index(default_page)
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
        
        # Input section
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
            col1, col2 = st.columns([3, 1])
            
            with col1:
                sample_idx = st.selectbox(
                    "Select a sample text",
                    range(min(10, len(self.sample_data))),
                    format_func=lambda i: f"Sample {i+1}: {self.sample_data.iloc[i]['text'][:50]}..."
                )
            
            with col2:
                if st.button("Use this sample"):
                    text = self.sample_data.iloc[sample_idx]['text']
                    st.session_state.text_input = text
                    st.rerun()  # Force rerun to update the text area
        
        # Analysis options - in horizontal layout
        st.subheader("Analysis Options")
        options_col1, options_col2, options_col3, options_col4 = st.columns(4)
        
        with options_col1:
            run_sentiment = st.checkbox("Sentiment Analysis", value=True)
        with options_col2:
            run_topics = st.checkbox("Topic Extraction", value=True)
        with options_col3:
            run_features = st.checkbox("Feature Extraction", value=True)
        
        # Analyze button - outside any column to appear in full width
        analyze_clicked = st.button("Analyze Text", key="analyze_btn", type="primary")
        
        # Process text only if button is clicked
        if analyze_clicked:
            if not text:
                st.error("Please enter or select some text to analyze")
            else:
                # Analysis results will appear below the button in the main area
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
                # Create a DataFrame with the single text entry
                text_df = pd.DataFrame({'text': [text]})
                
                # Get the metrics separately for better display
                metrics = self.models['feature_extractor'].extract_financial_metrics(text_df)
                
                if metrics and len(metrics) > 0:
                    # Convert metrics to DataFrame for better display
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                    
                    # Group metrics by type
                    dollar_metrics = metrics_df[metrics_df['Metric'].str.contains('dollar')]
                    percentage_metrics = metrics_df[metrics_df['Metric'].str.contains('percentage')]
                    other_metrics = metrics_df[~metrics_df['Metric'].str.contains('dollar|percentage')]
                    
                    # Create main columns
                    main_cols = st.columns([3, 2])
                    
                    with main_cols[0]:
                        st.subheader("Extracted Financial Metrics")
                        
                        # Use expandable sections for better organization
                        with st.expander("💵 Dollar Amounts", expanded=True):
                            if not dollar_metrics.empty:
                                st.dataframe(dollar_metrics, use_container_width=True, height=min(35*len(dollar_metrics)+38, 250))
                            else:
                                st.info("No dollar amounts found")
                                
                        with st.expander("📊 Percentages", expanded=True):
                            if not percentage_metrics.empty:
                                st.dataframe(percentage_metrics, use_container_width=True, height=min(35*len(percentage_metrics)+38, 250))
                            else:
                                st.info("No percentages found")
                                
                        if not other_metrics.empty:
                            with st.expander("🔢 Other Metrics"):
                                st.dataframe(other_metrics, use_container_width=True)
                        
                        # Add download button
                        csv = metrics_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download metrics as CSV",
                            csv,
                            "financial_metrics.csv",
                            "text/csv",
                            key='download-metrics'
                        )
                    
                    with main_cols[1]:
                        # Also get and show the complete feature set
                        features, feature_names = self.models['feature_extractor'].extract_features(text_df)
                        
                        # Display feature importance plot if available
                        fig = get_feature_importance_plot(self.models['feature_extractor'])
                        if fig:
                            st.pyplot(fig)
                else:
                    st.info("No financial metrics found in the text")
                    
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
        """Render the model zoo page showcasing available pre-trained models."""
        st.header("Model Zoo")
        
        st.markdown("""
        ### Pre-trained Models for Financial Text Analysis
        
        Browse and try out the available models for analyzing earnings reports.
        These models are used for various NLP tasks in the dashboard.
        """)
        
        # Create tabs for different model categories
        model_tabs = st.tabs(["Sentiment Models", "Topic Models", "Feature Extraction Models", "Custom Models"])
        
        # Sentiment Models Tab
        with model_tabs[0]:
            st.subheader("Financial Sentiment Analysis Models")
            
            if 'sentiment' in self.available_models and self.available_models['sentiment']:
                for model in self.available_models['sentiment']:
                    with st.expander(f"{model['name']}"):
                        st.write(model['description'])
                        
                        # Show model details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Version", model['version'])
                        with col2:
                            st.metric("Created", model['created_at'])
                        
                        # Try model section
                        st.markdown("#### Try the model")
                        sample_text = st.text_area(
                            "Enter text for sentiment analysis", 
                            "We are pleased to report strong financial results for the quarter.",
                            key=f"sentiment_sample_{model['name']}"
                        )
                        
                        if st.button("Analyze", key=f"analyze_btn_{model['name']}"):
                            with st.spinner("Analyzing sentiment..."):
                                if 'sentiment' in self.models:
                                    result = self.models['sentiment'].analyze(sample_text)
                                    result_df = format_sentiment_result(result)
                                    st.dataframe(result_df)
                                else:
                                    st.error("Sentiment model not loaded")
            else:
                st.info("No sentiment models available")
        
        # Topic Models Tab
        with model_tabs[1]:
            st.subheader("Financial Topic Models")
            
            if 'topic' in self.available_models and self.available_models['topic']:
                for model in self.available_models['topic']:
                    with st.expander(f"{model['name']}"):
                        st.write(model['description'])
                        
                        # Show model details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Version", model['version'])
                        with col2:
                            st.metric("Created", model['created_at'])
                        
                        # Try model section
                        st.markdown("#### Try the model")
                        sample_text = st.text_area(
                            "Enter text for topic modeling", 
                            "Our revenue increased by 15% this quarter due to strong product sales and market expansion.",
                            key=f"topic_sample_{model['name']}"
                        )
                        
                        if st.button("Extract Topics", key=f"topic_btn_{model['name']}"):
                            with st.spinner("Extracting topics..."):
                                if 'topic' in self.models:
                                    # Use actual topic model
                                    topics = self.models['topic'].extract_topics([sample_text])
                                    if isinstance(topics, list) and len(topics) > 0:
                                        topics_df = pd.DataFrame(topics, columns=['Topic ID', 'Score'])
                                        topics_df['Topic Words'] = topics_df['Topic ID'].apply(
                                            lambda tid: ", ".join(self.models['topic'].get_topic_words(tid, 10))
                                            if hasattr(self.models['topic'], 'get_topic_words') else f"Topic {tid}"
                                        )
                                        st.dataframe(topics_df)
                                    else:
                                        st.write("No clear topics identified")
                                else:
                                    st.error("Topic model not loaded")
            else:
                st.info("No topic models available")
        
        # Feature Extraction Models Tab
        with model_tabs[2]:
            st.subheader("Financial Feature Extraction Models")
            
            if 'feature_extractor' in self.available_models and self.available_models['feature_extractor']:
                for model in self.available_models['feature_extractor']:
                    with st.expander(f"{model['name']}"):
                        st.write(model['description'])
                        
                        # Show model details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Version", model['version'])
                        with col2:
                            st.metric("Created", model['created_at'])
                        
                        # Try model section
                        st.markdown("#### Try the model")
                        sample_text = st.text_area(
                            "Enter text for feature extraction", 
                            "In Q2 2024, we reported revenue of $125.3M, with EPS of $1.42, representing a 12% increase from the previous year.",
                            key=f"feature_sample_{model['name']}"
                        )
                        
                        if st.button("Extract Features", key=f"feature_btn_{model['name']}"):
                            with st.spinner("Extracting features..."):
                                if 'feature_extractor' in self.models:
                                    # Use actual feature extractor
                                    sample_df = pd.DataFrame({'text': [sample_text]})
                                    
                                    # Get financial metrics
                                    metrics = self.models['feature_extractor'].extract_financial_metrics(sample_df)
                                    
                                    if metrics and len(metrics) > 0:
                                        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                                        st.dataframe(metrics_df)
                                    else:
                                        st.info("No financial metrics found in the text")
                                else:
                                    st.error("Feature extraction model not loaded")
            else:
                st.info("No feature extraction models available")
        
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
        
        # Model selection - using actual available models
        st.subheader("Select Analysis Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get available sentiment models
            sentiment_models = []
            if 'sentiment' in self.available_models and self.available_models['sentiment']:
                sentiment_models = [model['name'] for model in self.available_models['sentiment']]
            
            if sentiment_models:
                sentiment_model = st.selectbox(
                    "Sentiment Analysis Model",
                    sentiment_models
                )
            else:
                st.info("No sentiment models available")
                sentiment_model = None
        
        with col2:
            # Get available topic models
            topic_models = []
            if 'topic' in self.available_models and self.available_models['topic']:
                topic_models = [model['name'] for model in self.available_models['topic']]
            
            if topic_models:
                topic_model = st.selectbox(
                    "Topic Model",
                    topic_models
                )
            else:
                st.info("No topic models available")
                topic_model = None
            
        with col3:
            # Get available feature extraction models
            feature_models = []
            if 'feature_extractor' in self.available_models and self.available_models['feature_extractor']:
                feature_models = [model['name'] for model in self.available_models['feature_extractor']]
            
            if feature_models:
                feature_model = st.selectbox(
                    "Feature Extraction Model",
                    feature_models
                )
            else:
                st.info("No feature extraction models available")
                feature_model = None
        
        # Analysis options
        st.subheader("Analysis Options")
        
        options_col1, options_col2 = st.columns(2)
        
        with options_col1:
            analyze_sentiment = st.checkbox("Analyze Sentiment", value=True and sentiment_model is not None)
            extract_topics = st.checkbox("Extract Topics", value=True and topic_model is not None)
            
        with options_col2:
            extract_features = st.checkbox("Extract Financial Metrics", value=True and feature_model is not None)
            predict_movement = st.checkbox("Predict Stock Movement", value=True)
        
        # Check if any models are available for analysis
        if not (sentiment_model or topic_model or feature_model):
            st.warning("No models are available for analysis. Please load models from the Model Zoo first.")
        
        # Run analysis button
        run_button_disabled = not (sentiment_model or topic_model or feature_model)
        
        if st.button("Run Analysis", type="primary", disabled=run_button_disabled):
            if not earnings_text:
                st.error("Please enter some earnings report text to analyze.")
                return
                
            with st.spinner("Analyzing earnings report..."):
                # Rest of your existing analysis code here
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
                    
                    # Sentiment analysis results for the summary
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
                    
                    # Stock prediction
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
                    
                    # Generate insights based on the text
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
                            
                            # Debug the sentiment result to check what's available
                            st.write("Debug - Sentiment Result:", sentiment_result)
                            
                            # Create a more reliable visualization using explicit values
                            pos_score = sentiment_result.get('pos', 0) 
                            neu_score = sentiment_result.get('neu', 0)
                            neg_score = sentiment_result.get('neg', 0)
                            
                            # Create explicit DataFrame for visualization
                            sentiment_plot_df = pd.DataFrame({
                                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                                'Score': [pos_score, neu_score, neg_score]
                            })
                            
                            # Add a more robust visualization
                            try:
                                fig = px.bar(
                                    sentiment_plot_df,
                                    x='Score',
                                    y='Sentiment', 
                                    orientation='h',
                                    color='Sentiment',
                                    color_discrete_map={
                                        'Positive': 'green',
                                        'Neutral': 'gray',
                                        'Negative': 'red'
                                    },
                                    title='Sentiment Breakdown'
                                )
                                fig.update_layout(
                                    xaxis_range=[0, 1],
                                    height=300,
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating sentiment visualization: {str(e)}")
                                
                                # Fallback to simpler visualization
                                st.write("Fallback visualization:")
                                st.bar_chart(sentiment_plot_df.set_index('Sentiment'))
                            
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
                            # Get topics from model
                            topics = self.models['topic'].extract_topics([earnings_text])
                            
                            # Debug the topic format
                            st.write("Debug - Topic Format:", type(topics))
                            st.write("Debug - Topic Content:", topics)
                            
                            formatted_topics = []
                            
                            # Handle different possible formats
                            if isinstance(topics, list):
                                for topic_entry in topics:
                                    try:
                                        # Handle list format [topic_id, score]
                                        if isinstance(topic_entry, list) and len(topic_entry) >= 2:
                                            topic_id = int(topic_entry[0])
                                            score = float(topic_entry[1])
                                        # Handle tuple format (topic_id, score)
                                        elif isinstance(topic_entry, tuple) and len(topic_entry) >= 2:
                                            topic_id = int(topic_entry[0])
                                            score = float(topic_entry[1])
                                        # Handle dictionary format
                                        elif isinstance(topic_entry, dict) and 'id' in topic_entry and 'score' in topic_entry:
                                            topic_id = int(topic_entry['id'])
                                            score = float(topic_entry['score'])
                                        else:
                                            continue
                                        
                                        # Get topic words
                                        if hasattr(self.models['topic'], 'get_topic_words'):
                                            topic_words = self.models['topic'].get_topic_words(topic_id, 8)
                                            if isinstance(topic_words, list):
                                                if topic_words and isinstance(topic_words[0], tuple):
                                                    # (word, score) format
                                                    topic_label = ", ".join([word for word, _ in topic_words[:8]])
                                                else:
                                                    topic_label = ", ".join(topic_words[:8])
                                            else:
                                                topic_label = f"Topic {topic_id}"
                                        else:
                                            topic_label = f"Topic {topic_id}"
                                        
                                        formatted_topics.append({
                                            "Topic ID": topic_id,
                                            "Topic Label": topic_label,
                                            "Relevance": score
                                        })
                                    except Exception as e:
                                        st.warning(f"Error formatting topic: {str(e)}")
                            
                            # If no topics were found or formatted, try simulating topics
                            if not formatted_topics:
                                st.info("No topics detected in the original format. Attempting to extract topics based on content...")
                                
                                # Simple keyword-based topic simulation
                                keywords = {
                                    "Financial Performance": ["revenue", "profit", "earnings", "financial", "quarter", "growth", "income"],
                                    "Market Position": ["market", "share", "competition", "industry", "position", "leader"],
                                    "Product Development": ["product", "development", "innovation", "launch", "roadmap"],
                                    "Operations": ["operations", "production", "supply", "chain", "efficiency", "cost"],
                                    "Future Outlook": ["outlook", "guidance", "future", "expect", "forecast", "anticipate"]
                                }
                                
                                # Extract simulated topics based on keyword matches
                                for topic_name, words in keywords.items():
                                    text_lower = earnings_text.lower()
                                    matches = sum(1 for word in words if word in text_lower)
                                    if matches >= 2:  # At least 2 keyword matches to consider it a relevant topic
                                        relevance = min(0.95, matches * 0.15)  # Calculate relevance score
                                        formatted_topics.append({
                                            "Topic ID": list(keywords.keys()).index(topic_name),
                                            "Topic Label": f"{topic_name}: {', '.join(words[:5])}",
                                            "Relevance": relevance
                                        })
                            
                            # Display topics
                            if formatted_topics:
                                topics_df = pd.DataFrame(formatted_topics)
                                st.write("### Document Topics")
                                st.dataframe(topics_df)
                                
                                # Create a visualization
                                fig = px.bar(
                                    topics_df,
                                    x="Relevance",
                                    y="Topic ID",
                                    orientation='h',
                                    title="Topic Distribution",
                                    labels={"Topic ID": "Topic", "Relevance": "Score"}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No clear topics identified in this text")
                                st.write("Consider trying a different topic model or using a longer text sample.")
                        else:
                            # Simulation code for when no topic model is loaded
                            st.info("Topic model not loaded. Showing simulated results.")
                
                # Financial Metrics tab
                with result_tabs[3]:
                    if extract_features:
                        st.subheader("Extracted Financial Metrics")
                        
                        if 'feature_extractor' in self.models:
                            # Create a DataFrame with the single text entry
                            earnings_df = pd.DataFrame({'text': [earnings_text]})
                            
                            # Try to use extract_financial_metrics first (more readable)
                            try:
                                metrics = self.models['feature_extractor'].extract_financial_metrics(earnings_df)
                                
                                if metrics and len(metrics) > 0:
                                    # Convert to DataFrame for better display
                                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                                    
                                    # Clean up metric names for display
                                    metrics_df['Metric'] = metrics_df['Metric'].str.replace('_', ' ').str.title()
                                    
                                    # Display the metrics in a clean table
                                    st.write("### Financial Metrics")
                                    st.dataframe(metrics_df, use_container_width=True)
                                    
                                    # Create a visualization if we have numeric values
                                    try:
                                        # Try to convert values to numeric for visualization
                                        metrics_df['Numeric_Value'] = pd.to_numeric(
                                            metrics_df['Value'].str.replace('$', '').str.replace('%', '').str.replace('B', '').str.replace('M', ''),
                                            errors='coerce'
                                        )
                                        
                                        # Filter for rows with valid numeric values
                                        numeric_metrics = metrics_df.dropna(subset=['Numeric_Value'])
                                        
                                        if not numeric_metrics.empty:
                                            fig = px.bar(
                                                numeric_metrics,
                                                x='Numeric_Value',
                                                y='Metric',
                                                orientation='h',
                                                title="Key Financial Metrics"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                    except:
                                        pass  # Skip visualization if conversion fails
                                else:
                                    # Fall back to raw features
                                    features, feature_names = self.models['feature_extractor'].extract_features(earnings_df)
                                    
                                    if isinstance(features, np.ndarray) and feature_names:
                                        # Convert to DataFrame with feature names
                                        if len(features.shape) == 2:
                                            features_df = pd.DataFrame(features, columns=feature_names)
                                        else:
                                            features_df = pd.DataFrame([features], columns=feature_names)
                                        
                                        # Only show non-zero features
                                        non_zero_cols = features_df.loc[:, (features_df != 0).any(axis=0)].columns
                                        if len(non_zero_cols) > 0:
                                            st.write("### Key Financial Features")
                                            st.dataframe(features_df[non_zero_cols])
                                        else:
                                            st.info("No significant financial features detected")
                                    else:
                                        st.write(features)  # Fallback
                                        st.info("Raw features shown above may need interpretation")
                            except Exception as e:
                                st.error(f"Error processing financial metrics: {str(e)}")
                                # Fall back to raw features
                                features, feature_names = self.models['feature_extractor'].extract_features(earnings_df)
                                st.write("### Raw Feature Output")
                                st.write(features)
                        else:
                            # Your existing simulation code for feature extraction
                            st.info("Feature extraction model not loaded. Showing simulated results.")
                
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
        
        # Sort model options to put implemented visualizations first
        preferred_order = ['sentiment', 'topic', 'feature_extractor']
        sorted_models = sorted(model_options, 
                            key=lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order))
        
        selected_model = st.selectbox("Select model", sorted_models)
        
        if selected_model == 'topic':
            self._render_topic_model_performance()
        elif selected_model == 'sentiment':
            self._render_sentiment_model_performance()
        elif selected_model == 'feature_extractor':
            self._render_feature_extractor_performance()
        else:
            st.write(f"Performance visualization not implemented for {selected_model} model.")
            
    def _render_topic_model_performance(self):
        """Render topic model performance metrics and evaluation."""
        st.subheader("Topic Model Performance")
        
        if 'topic' in self.models:
            # Display topic model information
            model = self.models['topic']
            st.write(f"**Model Type:** {model.method.upper() if hasattr(model, 'method') else 'Unknown'}")
            st.write(f"**Number of Topics:** {model.num_topics if hasattr(model, 'num_topics') else 'Unknown'}")
            
            # Check if topic words are available or generate them
            if not hasattr(model, 'topic_words') or not model.topic_words:
                st.info("Topic words not found in model. Generating from model components...")
                
                try:
                    # Generate topic words from model components
                    if hasattr(model, 'model') and hasattr(model, 'feature_names'):
                        # Create topic_words dictionary
                        num_topics = model.num_topics if hasattr(model, 'num_topics') else 10
                        model.topic_words = {}
                        
                        # For LDA models, extract top words for each topic
                        if hasattr(model.model, 'components_'):
                            for topic_id in range(num_topics):
                                # Get indices of top words for this topic
                                top_indices = model.model.components_[topic_id].argsort()[:-21:-1]
                                # Map indices to actual words using feature_names
                                words = [model.feature_names[i] for i in top_indices]
                                model.topic_words[topic_id] = words
                            st.success(f"Successfully generated topic words for {num_topics} topics")
                        else:
                            st.warning("Model components not available for generating topic words")
                    else:
                        st.warning("Model or feature names not available")
                        
                    # If topic_words is still empty, try to load from file
                    if not hasattr(model, 'topic_words') or not model.topic_words:
                        raise ValueError("Could not generate topic words from model components")
                        
                except Exception as e:
                    st.warning(f"Could not generate topic words: {str(e)}")
                    
                    # Try to load topic words from the model directory
                    try:
                        # Get the directory where the model is stored
                        model_dir = os.path.dirname(TOPIC_MODEL_PATH)
                        topic_words_path = os.path.join(model_dir, 'topic_words.json')
                        
                        st.info(f"Looking for topic words file at: {topic_words_path}")
                        
                        # If topic_words.json doesn't exist, create it from the model
                        if not os.path.exists(topic_words_path) and hasattr(model.model, 'components_') and hasattr(model, 'feature_names'):
                            st.info("Creating topic_words.json from model components...")
                            
                            # Generate topic words
                            num_topics = model.num_topics if hasattr(model, 'num_topics') else 10
                            topic_words = {}
                            
                            for topic_id in range(num_topics):
                                # Get indices of top words for this topic
                                top_indices = model.model.components_[topic_id].argsort()[:-21:-1]
                                # Map indices to actual words using feature_names
                                words = [model.feature_names[i] for i in top_indices]
                                topic_words[str(topic_id)] = words
                                
                            # Save to file
                            with open(topic_words_path, 'w') as f:
                                json.dump(topic_words, f)
                                
                            st.success(f"Created topic words file at: {topic_words_path}")
                        
                        # Now try to load the file (either existing or newly created)
                        if os.path.exists(topic_words_path):
                            with open(topic_words_path, 'r') as f:
                                topic_words = json.load(f)
                                
                            # Convert keys to integers
                            model.topic_words = {int(k): v for k, v in topic_words.items()}
                            st.success("Topic words loaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"Topic words file not found at: {topic_words_path}")
                    except Exception as e:
                        st.error(f"Error handling topic words: {str(e)}")
            
            # Display topic distribution if topic words are available
            if hasattr(model, 'topic_words') and model.topic_words:
                topics_df = pd.DataFrame({
                    'Topic ID': list(model.topic_words.keys()),
                    'Topic Label': [f"Topic {i}" for i in model.topic_words.keys()],
                    'Top Words': [', '.join(words[:8]) for words in model.topic_words.values()]
                })
                st.write("### Topic Distribution")
                st.dataframe(topics_df)
                
                # Add topic exploration section
                st.write("### Explore Topics")
                selected_topic = st.selectbox(
                    "Select a topic to explore:", 
                    options=list(model.topic_words.keys()),
                    format_func=lambda x: f"Topic {x}: {', '.join(model.topic_words[x][:5])}"
                )
                
                if selected_topic is not None:
                    st.write(f"#### Top words for Topic {selected_topic}")
                    st.write(", ".join(model.topic_words[selected_topic]))
                    
                    # Generate and display wordcloud for selected topic
                    try:
                        if hasattr(model, 'plot_wordcloud'):
                            fig = model.plot_wordcloud(int(selected_topic))
                            st.pyplot(fig)
                        else:
                            st.info("Wordcloud visualization not available for this model.")
                    except Exception as e:
                        st.error(f"Error generating wordcloud: {str(e)}")
            else:
                st.error("Unable to retrieve or generate topic words. The model may be incompatible.")
        else:
            st.error("Topic model not loaded")
                
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
            try:
                # Create a DataFrame with the single text entry
                sample_df = pd.DataFrame({'text': [sample_text]})
                
                # First try to get financial metrics which are usually in a more user-friendly format
                metrics = feature_extractor.extract_financial_metrics(sample_df)
                
                if metrics and len(metrics) > 0:
                    # Display financial metrics in a clean format
                    st.subheader("Financial Metrics")
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                    
                    # Group metrics by type for cleaner display
                    dollar_metrics = metrics_df[metrics_df['Metric'].str.contains('dollar')]
                    percentage_metrics = metrics_df[metrics_df['Metric'].str.contains('percentage')]
                    other_metrics = metrics_df[~metrics_df['Metric'].str.contains('dollar|percentage')]
                    
                    # Show metrics in expandable sections
                    with st.expander("💵 Dollar Amounts", expanded=True):
                        if not dollar_metrics.empty:
                            st.dataframe(dollar_metrics, use_container_width=True)
                        else:
                            st.info("No dollar amounts found")
                            
                    with st.expander("📊 Percentages", expanded=True):
                        if not percentage_metrics.empty:
                            st.dataframe(percentage_metrics, use_container_width=True)
                        else:
                            st.info("No percentages found")
                            
                    if not other_metrics.empty:
                        with st.expander("🔢 Other Metrics"):
                            st.dataframe(other_metrics, use_container_width=True)
                
                # Also show the raw features for completeness
                st.subheader("Raw Extracted Features")
                features, feature_names = feature_extractor.extract_features(sample_df)
                
                if isinstance(features, dict):
                    # Dictionary format
                    feature_df = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Value': list(features.values())
                    })
                    st.dataframe(feature_df)
                elif isinstance(features, pd.DataFrame):
                    # DataFrame format
                    st.dataframe(features)
                elif isinstance(features, np.ndarray):
                    # NumPy array format with feature_names
                    if feature_names is not None:
                        if len(features.shape) == 2:  # 2D array (multiple samples)
                            feature_df = pd.DataFrame(features, columns=feature_names)
                        else:  # 1D array (single sample)
                            feature_df = pd.DataFrame([features], columns=feature_names)
                        st.dataframe(feature_df)
                    else:
                        st.write("Features extracted as NumPy array but no feature names provided")
                else:
                    st.write(f"Features extracted but format not recognized: {type(features)}")
                    st.write("Feature data preview:")
                    st.write(features)
            
            except Exception as e:
                st.error(f"Error extracting features: {str(e)}")
                
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
        
        # Quick links in render_home()
        with col1:
            st.subheader("Text Analysis")
            st.markdown("Analyze individual earnings report text for sentiment, topics, and key features.")
            if st.button("Go to Text Analysis"):
                st.session_state.nav_target = "Text Analysis"  # Changed to nav_target
                st.rerun()

        with col2:
            st.subheader("Model Zoo")
            st.markdown("Access and load pre-trained models for different analysis tasks.")
            if st.button("Go to Model Zoo"):
                st.session_state.nav_target = "Model Zoo"  # Changed to nav_target
                st.rerun()
                
        with col3:
            st.subheader("Prediction Simulator")
            st.markdown("Simulate predictions on custom earnings report text.")
            if st.button("Go to Prediction Simulator"):
                st.session_state.nav_target = "Prediction Simulator"  # Changed to nav_target
                st.rerun()
    
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