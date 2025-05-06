# model_trainer.py
# Functions for training and evaluating Lasso regression and classifier models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from scipy.stats import randint, uniform, loguniform
import os
import sys
import pickle

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (ALPHA_MIN, ALPHA_MAX, N_ALPHAS, TEST_SIZE, CV_FOLDS, RANDOM_STATE,
                   LARGE_RETURN_THRESHOLD, N_ITER_SEARCH, OUTPUT_DIR, MODEL_DIR)

def train_lasso_model(topics, returns, alpha_range=(ALPHA_MIN, ALPHA_MAX), n_alphas=N_ALPHAS,
                     test_size=TEST_SIZE, cv_folds=CV_FOLDS, save_model=True):
    """
    Train a Lasso regression model to evaluate which topics best predict stock returns
    
    Args:
        topics (numpy.ndarray): Topic distribution matrix
        returns (pandas.Series): Buy-and-hold abnormal returns
        alpha_range (tuple): Min and max values for alpha hyperparameter
        n_alphas (int): Number of alpha values to test
        test_size (float): Proportion of data to use for testing
        cv_folds (int): Number of cross-validation folds
        save_model (bool): Whether to save the model
        
    Returns:
        tuple: (Best Lasso model, training results, topics with non-zero coefficients)
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        topics, returns, test_size=test_size, random_state=RANDOM_STATE
    )
    
    # Set up parameter grid and randomized search
    param_grid = {
        'alpha': np.random.uniform(alpha_range[0], alpha_range[1], n_alphas),
        'random_state': [RANDOM_STATE]
    }
    
    lasso = Lasso(max_iter=10000)
    
    # Perform randomized search with cross-validation
    search = RandomizedSearchCV(
        lasso, 
        param_grid, 
        n_iter=min(50, n_alphas),
        cv=cv_folds,
        scoring='neg_root_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    # Get best model and its predictions
    best_lasso = search.best_estimator_
    best_alpha = search.best_params_['alpha']
    y_pred = best_lasso.predict(X_test)
    
    # Analyze coefficients
    coefs = best_lasso.coef_
    nonzero_count = np.sum(coefs != 0)
    most_positive_idx = np.argmax(coefs)
    most_negative_idx = np.argmin(coefs)
    
    # Results dictionary
    results = {
        'best_alpha': best_alpha,
        'nonzero_topics': nonzero_count,
        'most_positive_topic': most_positive_idx,
        'most_positive_coef': coefs[most_positive_idx],
        'most_negative_topic': most_negative_idx,
        'most_negative_coef': coefs[most_negative_idx],
        'alphas': search.cv_results_['param_alpha'].data,
        'scores': search.cv_results_['mean_test_score']
    }
    
    # Save the model if requested
    if save_model and MODEL_DIR:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, 'lasso_model.pkl'), 'wb') as f:
            pickle.dump(best_lasso, f)
        
        # Save results
        if OUTPUT_DIR:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            pd.DataFrame({
                'Topic': range(len(coefs)),
                'Coefficient': coefs
            }).to_csv(os.path.join(OUTPUT_DIR, 'lasso_coefficients.csv'), index=False)
            
            # Also save a summary of the nonzero coefficients
            nonzero_topics = np.where(coefs != 0)[0]
            nonzero_coefs = coefs[nonzero_topics]
            pd.DataFrame({
                'Topic': nonzero_topics,
                'Coefficient': nonzero_coefs
            }).sort_values('Coefficient', ascending=False).to_csv(
                os.path.join(OUTPUT_DIR, 'nonzero_lasso_coefficients.csv'), index=False
            )
    
    return best_lasso, results, nonzero_topics

def plot_lasso_alpha(alphas, scores, save_plot=True):
    """
    Plot the validation scores vs alpha values from Lasso regression tuning
    
    Args:
        alphas (array): Alpha values tested
        scores (array): Corresponding validation scores
        save_plot (bool): Whether to save the plot
        
    Returns:
        matplotlib.pyplot.Figure: The plot figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, scores)
    plt.xlabel('Alpha')
    plt.ylabel('Mean CV Neg RMSE')
    plt.title('Lasso: Alpha vs. Mean CV Neg RMSE')
    plt.grid(True)
    
    if save_plot and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'lasso_alpha_plot.png'))
    
    return plt

def train_classifiers(topics, returns, large_return_threshold=LARGE_RETURN_THRESHOLD,
                     test_size=TEST_SIZE, n_iter_search=N_ITER_SEARCH, save_models=True):
    """
    Train multiple classifiers to predict large positive returns
    
    Args:
        topics (numpy.ndarray): Topic distribution matrix
        returns (pandas.Series): Buy-and-hold abnormal returns
        large_return_threshold (float): Threshold for considering returns as "large"
        test_size (float): Proportion of data to use for testing
        n_iter_search (int): Number of iterations for RandomizedSearchCV
        save_models (bool): Whether to save the models
        
    Returns:
        tuple: (Best model, results dictionary)
    """
    # Create binary target variable for large returns
    y_binary = (returns > large_return_threshold).astype(int)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        topics, y_binary, test_size=test_size, random_state=RANDOM_STATE
    )
    
    # Define models and their parameter distributions
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'params': {
                'C': loguniform(1e-4, 1e3),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'class_weight': [None, 'balanced']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=RANDOM_STATE),
            'params': {
                'kernel': ['linear', 'rbf'],
                'C': loguniform(1e-4, 1e3),
                'gamma': loguniform(1e-4, 1e2),
                'class_weight': [None, 'balanced']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(2, 50),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 20),
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced']
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': randint(50, 500),
                'learning_rate': loguniform(1e-4, 1),
                'max_depth': randint(2, 50),
                'subsample': uniform(0.5, 0.5),  # from 0.5 to 1.0
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 20)
            }
        }
    }
    
    results = {}
    best_models = {}
    
    # Train and evaluate each model
    for model_name, model_info in models.items():
        print(f"Training {model_name}...")
        model = model_info['model']
        params = model_info['params']
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter_search,
            scoring='f1_macro',
            cv=CV_FOLDS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        best_models[model_name] = best_model
        
        y_pred = best_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results[model_name] = {
            'best_params': search.best_params_,
            'cv_score': search.best_score_,
            'test_f1_macro': f1,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"{model_name} - Test Macro F1: {f1:.4f}")
        
    # Find the best model
    best_model_name = max(results, key=lambda x: results[x]['test_f1_macro'])
    best_model = best_models[best_model_name]
    
    # Save models and results if requested
    if save_models and MODEL_DIR:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for model_name, model in best_models.items():
            with open(os.path.join(MODEL_DIR, f'{model_name.lower()}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Save best model separately
        with open(os.path.join(MODEL_DIR, 'best_classifier_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save results
        if OUTPUT_DIR:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, 'classifier_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
    
    return best_model, results

def print_classification_report(model_name, y_test, y_pred):
    """
    Print classification report for a model
    
    Args:
        model_name (str): Name of the model
        y_test (array): True labels
        y_pred (array): Predicted labels
        
    Returns:
        dict: Classification report as a dictionary
    """
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    
    print(f"Classification Report for {model_name}:")
    print(report_str)
    
    return report_dict

if __name__ == "__main__":
    # This allows the script to be run directly
    import os
    
    print("Loading data and topics...")
    try:
        df = pd.read_csv("./task2_data_clean.csv.gz")
        topics = np.load(os.path.join(OUTPUT_DIR, 'topic_distributions.npy'))
        print("Data and topics loaded successfully.")
    except FileNotFoundError:
        print("Error: Required data files not found. Please run data_processor.py and feature_extractor.py first.")
        sys.exit(1)
    
    # Train Lasso model
    print("Training Lasso regression model...")
    lasso_model, lasso_results, nonzero_topics = train_lasso_model(topics, df['BHAR0_2'])
    
    # Plot Lasso alpha values
    print("Plotting Lasso alpha values...")
    plot_lasso_alpha(lasso_results['alphas'], lasso_results['scores'])
    
    # Train classifier models
    print("Training classifier models...")
    best_classifier, classifier_results = train_classifiers(topics, df['BHAR0_2'])
    
    print("Model training complete!")