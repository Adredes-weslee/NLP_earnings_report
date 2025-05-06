# EarningsNLP

A Python package for analyzing earnings announcements using NLP and machine learning techniques.

## Overview

This project analyzes the text of earnings announcements from publicly traded companies using natural language processing (NLP) and machine learning techniques. The main tasks performed are:

1. **Data Preparation** - Loading and cleaning earnings announcement text data
2. **Topic Modeling** - Using LDA (Latent Dirichlet Allocation) to discover topics in the announcements
3. **Lasso Regression** - Evaluating which topics best predict stock returns
4. **Classification** - Building models to predict large positive stock returns

## Project Structure

```
EarningsNLP/
├── config.py               # Configuration parameters
├── data_processor.py       # Data loading and text preprocessing
├── feature_extractor.py    # Feature extraction and topic modeling
├── model_trainer.py        # Lasso regression and classifier models
├── utils.py                # Utility functions for visualization and reporting
├── streamlit_app.py        # Interactive Streamlit application
├── main.py                 # Main script to orchestrate the workflow
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository or download the files
2. Navigate to the project directory
3. Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Analysis

### Using the Command Line Interface

The `main.py` script provides a command-line interface for running the full analysis pipeline:

```powershell
python main.py [OPTIONS]
```

Options:
- `--data-path PATH`: Path to the data file
- `--n-topics N`: Number of topics for LDA (default: automatically determined)
- `--force-reprocess`: Force data reprocessing even if processed data exists
- `--no-tune-topics`: Skip topic tuning and use the default number of topics

Example:
```powershell
python main.py --data-path "path\to\ExpTask2Data.csv.gz" --n-topics 40
```

### Using the Streamlit App

For an interactive experience, you can run the Streamlit application:

```powershell
streamlit run streamlit_app.py
```

This will open a web interface where you can:
- Upload and process data
- Run topic modeling with customizable parameters
- Perform Lasso regression analysis
- Train classification models
- Generate a comprehensive analysis report

### Running Individual Components

You can also run each component separately:

```powershell
# Data processing
python data_processor.py

# Feature extraction and topic modeling
python feature_extractor.py

# Model training
python model_trainer.py
```

## Output

The analysis results are saved to:
- `./outputs/`: Reports, visualizations, and processed data
- `./models/`: Trained models and vectorizers

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`