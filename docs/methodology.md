# NLP Earnings Report Analysis: Methodology

## Overview

This document provides a detailed explanation of the methodologies used in the NLP Earnings Report Analysis project. The project implements various Natural Language Processing (NLP) techniques to analyze earnings announcement texts from publicly traded companies and extract insights that may correlate with stock price movements.

## Data Pipeline

### Data Loading and Versioning

The data pipeline implements a robust approach to data management:

- **Data Versioning**: Each dataset is assigned a unique version ID based on a hash of its contents, enabling reproducible analysis and experiments
- **Train/Validation/Test Splitting**: Data is split using stratified sampling (based on the target variable) to ensure representative distribution across splits
- **Configuration Tracking**: All preprocessing parameters are tracked and saved with each data version

### Text Preprocessing

The preprocessing pipeline includes the following steps:

1. **Financial Number Replacement**: Dollar amounts, percentages, and large numbers are replaced with special tokens (`financial_number`, `percentage_number`, `number`) to reduce vocabulary size and improve generalization
2. **Sentence Filtering**: Sentences that are too short, too long, or contain boilerplate content (e.g., "safe harbor statements") are removed
3. **Text Cleaning**: Standard text cleaning operations including lowercase conversion, special character removal, and whitespace normalization
4. **Tokenization**: Text is split into tokens with optional stopword removal

## NLP Techniques

### Text Embedding

Multiple embedding approaches are supported:

1. **Bag-of-Words (BoW)**: Simple count-based document representation
2. **TF-IDF**: Term frequency-inverse document frequency weighting to emphasize important terms
3. **Transformer-based Embeddings**: Support for modern contextual embeddings using models like BERT/FinBERT through the `sentence-transformers` library

### Sentiment Analysis

Financial text sentiment is analyzed using:

1. **Loughran-McDonald Financial Lexicon**: A domain-specific lexicon for financial text that categorizes words into positive, negative, uncertainty, and litigious categories
2. **FinBERT Sentiment Analysis**: A transformer-based model fine-tuned on financial text (when available)
3. **Combined Approach**: Merges lexicon-based and transformer-based approaches for robust sentiment analysis

### Topic Modeling

Two main approaches to topic modeling are implemented:

1. **Latent Dirichlet Allocation (LDA)**:
   - Optimal number of topics determined through coherence score optimization
   - Topic quality evaluated using c_v coherence metric
   - Topics visualized using word distributions and word clouds

2. **BERTopic** (when available):
   - Combines transformer embeddings with UMAP dimensionality reduction and HDBSCAN clustering
   - Enables more coherent topic identification leveraging contextual embeddings
   - Provides interactive topic visualizations

### Feature Extraction

The feature extraction process identifies:

1. **Numerical Metrics**: Revenue, EPS, growth rates, margins extracted via regex patterns
2. **Named Entities**: Companies, people, locations, and financial entities identified using spaCy or transformer-based NER
3. **Readability Metrics**: Flesch Reading Ease, Gunning Fog Index, and other complexity measures
4. **Financial Sentiment**: Based on domain-specific lexicons and accounting for financial terminology

## Predictive Modeling

### Lasso Regression

For identifying which topics best predict stock returns:

1. **Feature Creation**: Topic distributions (probabilities) used as features
2. **Target Variable**: Buy-and-Hold Abnormal Returns (BHAR0_2) over 3-day period
3. **Regularization**: Lasso regression (L1 regularization) for feature selection and to prevent overfitting
4. **Cross-Validation**: K-fold cross-validation for hyperparameter tuning and performance estimation

### Classification Models

For predicting large positive stock returns (>5%):

1. **Feature Creation**: Combination of topic distributions, sentiment scores, and extracted metrics
2. **Target Creation**: Binary target based on whether returns exceed 5% threshold
3. **Models Implemented**:
   - Random Forest
   - Logistic Regression
   - Support Vector Machines
   - Gradient Boosting
4. **Evaluation Metrics**: Precision, recall, F1-score, and ROC AUC
5. **Cross-Validation**: Stratified K-fold to handle class imbalance

## Visualization and Interpretation

The project provides several visualization methods:

1. **Topic Word Clouds**: Visual representation of important words in each topic
2. **Feature Importance Plots**: Visualization of feature contribution to predictions
3. **Interactive Topic Exploration**: Interactive tools for exploring topic relationships
4. **Sentiment Distribution**: Visual breakdown of sentiment components in texts

## Implementation Details

### Model Storage and Loading

1. **Centralized Configuration**: All model paths are defined in `config.py`
2. **Standardized Loading Process**: Each model class implements a `load()` class method
3. **Error Handling**: Graceful degradation when specific models cannot be loaded
4. **Permission Management**: Directory permissions are handled programmatically
5. **PyTorch Integration**: Environment variables are used to prevent conflicts between PyTorch and Streamlit

## References

1. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10‐Ks. The Journal of Finance, 66(1), 35-65.
2. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.
3. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.