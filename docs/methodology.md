# NLP Earnings Report Analysis: Methodology

## Overview

This document provides a detailed explanation of the methodologies used in the NLP Earnings Report Analysis project. The project implements various Natural Language Processing (NLP) techniques to analyze earnings announcement texts from publicly traded companies and extract insights that may correlate with stock price movements.

The documentation follows Google-style standards throughout the codebase to ensure clarity, consistency, and maintainability across all components. Every function, class, and module includes comprehensive docstrings with clear descriptions, argument specifications, return types, and usage examples.

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

### Code Documentation

The project implements comprehensive documentation following Google-style docstring format:

1. **Module-Level Documentation**: Each module begins with a detailed docstring explaining its purpose, components, and usage patterns
2. **Class Documentation**: Classes include comprehensive descriptions, attribute details, and usage examples
3. **Method Documentation**: Each method includes:
   - One-line summary description
   - Detailed multi-line explanation
   - Args section with parameter types and descriptions
   - Returns section with return value types and descriptions
   - Examples section showing typical usage patterns
   - Notes section for additional context and implementation details

The Google-style documentation standard was rigorously applied across all components:

```python
def method_name(param1, param2):
    """One-line summary description of the method's purpose.
    
    Detailed multi-line explanation of what the method does, how it works,
    when it should be used, and any other relevant contextual information.
    The description explains the method's role in the larger system.
    
    Args:
        param1 (type): Description of the first parameter.
        param2 (type): Description of the second parameter.
            
    Returns:
        return_type: Description of the return value.
            
    Note:
        Additional implementation details, edge cases, or usage constraints.
    """
```

This standardized documentation approach ensures consistency across the codebase and facilitates both maintenance and knowledge transfer.

### Dashboard Documentation

The Streamlit dashboard components follow particularly rigorous documentation standards:

1. **EarningsReportDashboard Class**: Comprehensive class-level documentation with detailed attribute descriptions
2. **Rendering Methods**: Each `render_*` method includes:
   - Purpose and functionality description
   - UI components created and their relationships
   - Data dependencies and state management details
   - User interaction handling
3. **Helper Methods**: Internal helpers include documentation on:
   - Data transformation logic
   - Parameter validation
   - Error handling approach
   - UI component generation patterns

The dashboard's documentation ensures that all UI components are consistently implemented and easily maintained, with special attention to error handling for file access issues and PyTorch/Streamlit integration. It is particularly important for the dashboard components, where clear documentation improves UI element consistency and developer onboarding.

### Model Storage and Loading

1. **Centralized Configuration**: All model paths are defined in `config.py`
2. **Standardized Loading Process**: Each model class implements a `load()` class method
3. **Error Handling**: Graceful degradation when specific models cannot be loaded
4. **Permission Management**: Directory permissions are handled programmatically
5. **PyTorch Integration**: Environment variables are used to prevent conflicts between PyTorch and Streamlit

## Code Quality and Documentation Standards

The project adheres to high code quality standards through several mechanisms:

1. **Comprehensive Google-Style Documentation**
   - Every function, class, and module has complete Google-style docstrings
   - Documentation covers purpose, parameters, return values, and usage examples
   - Special notes sections explain edge cases and implementation details

2. **Standardized Error Handling**
   - Consistent error handling patterns across all components
   - Graceful degradation when optional components are unavailable
   - User-friendly error messages with actionable information

3. **Maintainable Code Organization**
   - Logical module separation with clear responsibilities
   - Consistent naming conventions across the codebase
   - Separation of concerns between data, analysis, and presentation layers

4. **Testing Strategy**
   - Comprehensive test suite with both full and quick testing options
   - Test utilities for consistent test setup and execution
   - Testing documentation follows the same Google-style format

These standards ensure the codebase remains maintainable, extensible, and accessible to new contributors.

## References

1. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10‐Ks. The Journal of Finance, 66(1), 35-65.
2. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.
3. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
4. Google Python Style Guide. (n.d.). https://google.github.io/styleguide/pyguide.html