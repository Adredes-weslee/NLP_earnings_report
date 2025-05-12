"""
Functions for training and evaluating Lasso regression and classifier models.

This module provides functionality for:
1. Training and evaluating Lasso regression models for feature selection
2. Training and evaluating various classifier models for predicting large returns
3. Cross-validation and hyperparameter tuning
4. Feature importance extraction from trained models

The module supports both regression (predicting exact return values) and 
classification (predicting whether returns exceed a threshold) approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate as sklearn_cross_validate
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from scipy.stats import randint, uniform, loguniform
import os
import sys
import pickle

# Import configuration
from ..config import (ALPHA_MIN, ALPHA_MAX, N_ALPHAS, TEST_SIZE, CV_FOLDS, RANDOM_STATE,
                   LARGE_RETURN_THRESHOLD, N_ITER_SEARCH, OUTPUT_DIR, MODEL_DIR)

def train_model(X, y, model_type='classifier', **kwargs):
    """General entry point for training models.
    
    This function serves as a unified interface for training different types of models.
    It delegates the actual training to specialized functions based on the model_type
    parameter.
    
    Args:
        X (numpy.ndarray): Features matrix, such as topic distributions or embeddings.
        y (numpy.ndarray or pandas.Series): Target variable - either continuous returns 
            for regression or binary/categorical labels for classification.
        model_type (str): Type of model to train. Options:
            - 'lasso': Train Lasso regression for sparse feature selection
            - 'classifier': Train classification models (logistic regression, SVM, etc.)
            - 'all': Train both regression and classification models
        **kwargs: Additional arguments to pass to the specific training function.
            For Lasso: alpha_range, n_alphas, test_size, cv_folds, save_model
            For Classifiers: large_return_threshold, test_size, n_iter_search, save_models
    
    Returns:
        dict: Dictionary containing trained models and results. Format:
            {
                'lasso': {
                    'model': trained_lasso_model,
                    'results': lasso_training_metrics,
                    'nonzero_topics': list_of_nonzero_feature_indices
                },
                'classifier': {
                    'model': best_classifier_model,
                    'results': classifier_training_metrics
                }
            }
            Only includes keys for the model types that were trained.
    
    Examples:
        >>> # Train just a classifier model
        >>> results = train_model(topic_distributions, returns)
        >>> 
        >>> # Train both model types
        >>> results = train_model(topic_distributions, returns, model_type='all')
        >>> lasso_model = results['lasso']['model']
        >>> classifier = results['classifier']['model']
    """
    results = {}
    
    if model_type.lower() in ['lasso', 'all']:
        lasso_model, lasso_results, nonzero_topics = train_lasso_model(X, y, **kwargs)
        results['lasso'] = {
            'model': lasso_model,
            'results': lasso_results,
            'nonzero_topics': nonzero_topics
        }
        
    if model_type.lower() in ['classifier', 'all']:
        best_classifier, classifier_results = train_classifiers(X, y, **kwargs)
        results['classifier'] = {
            'model': best_classifier,
            'results': classifier_results
        }
        
    return results

def evaluate_model(model, X_test, y_test, model_type='classifier'):
    """Evaluate a trained model on test data.
    
    Computes relevant evaluation metrics based on the model type (regression or
    classification) and returns them in a structured dictionary.
    
    Args:
        model: The trained model to evaluate. Should have a predict() method.
        X_test (numpy.ndarray): Test features matrix.
        y_test (numpy.ndarray or pandas.Series): True test labels or target values.
        model_type (str): Type of model to evaluate. Options:
            - 'lasso': Evaluate regression metrics (MSE, RMSE, MAE, R²)
            - 'classifier': Evaluate classification metrics (accuracy, precision, 
               recall, F1, ROC AUC if available)
        
    Returns:
        dict: Dictionary containing evaluation metrics. Contents depend on model_type:
            For 'lasso':
            {
                'mse': mean_squared_error,
                'rmse': root_mean_squared_error,
                'mae': mean_absolute_error,
                'r2': r2_score,
                'predictions': predicted_values
            }
            
            For 'classifier':
            {
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score,
                'predictions': predicted_classes,
                'classification_report': detailed_classification_report,
                'roc_auc': roc_auc_score  # Only if model supports predict_proba()
            }
    
    Examples:
        >>> # Evaluate a classifier
        >>> results = evaluate_model(trained_classifier, X_test, y_test)
        >>> print(f"Accuracy: {results['accuracy']:.4f}")
        >>> print(f"F1 Score: {results['f1']:.4f}")
        >>> 
        >>> # Evaluate a regression model
        >>> results = evaluate_model(trained_lasso, X_test, y_test, model_type='lasso')
        >>> print(f"RMSE: {results['rmse']:.4f}")
        >>> print(f"R²: {results['r2']:.4f}")
    """
    results = {}
    
    if model_type.lower() == 'lasso':
        # For regression models
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
    elif model_type.lower() == 'classifier':
        # For classification models
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = model.predict(X_test)
        y_prob = None
        
        # Try to get probability predictions if model supports it
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError):
            pass
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'classification_report': print_classification_report('Model', y_test, y_pred)
        }
        
        # Add ROC AUC if probabilities are available
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
                results['roc_auc'] = roc_auc
            except ValueError:
                # ROC AUC can fail with multiclass or if only one class is present
                pass
    
    return results

def cross_validate(model, X, y, cv_folds=CV_FOLDS, model_type='classifier', scoring=None):
    """Perform cross-validation for a model.
    
    Evaluates model performance using k-fold cross-validation, providing both
    training and test performance metrics for each fold.
    
    Args:
        model: The model to cross-validate. Must implement scikit-learn estimator API.
        X (numpy.ndarray): Features matrix.
        y (numpy.ndarray or pandas.Series): Target values.
        cv_folds (int): Number of cross-validation folds. Default is defined in config.
        model_type (str): Type of model ('lasso' or 'classifier'). Determines the
            default scoring metrics if none are provided.
        scoring (str, callable, list, dict, None): Scoring strategy for 
            sklearn.model_selection.cross_validate. If None, uses default metrics:
            - For 'lasso': ['neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
            - For classifiers: ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
    Returns:
        dict: Cross-validation results with structure:
            {
                'cv_folds': number_of_folds,
                'test_scores': {
                    'metric_name': {
                        'mean': mean_score_across_folds,
                        'std': standard_deviation,
                        'values': [score_for_each_fold]
                    },
                    ...
                },
                'train_scores': {
                    'metric_name': {
                        'mean': mean_score_across_folds,
                        'std': standard_deviation,
                        'values': [score_for_each_fold]
                    },
                    ...
                }
            }
            For regression models, negative metrics (like neg_mean_squared_error) 
            will be converted to their positive equivalent (mean_squared_error).
    
    Examples:
        >>> # Cross-validate a classifier
        >>> cv_results = cross_validate(classifier, X, y)
        >>> print(f"Mean accuracy: {cv_results['test_scores']['accuracy']['mean']:.4f}")
        >>> print(f"Standard deviation: {cv_results['test_scores']['accuracy']['std']:.4f}")
        >>>
        >>> # Cross-validate a regressor with custom scoring
        >>> cv_results = cross_validate(lasso, X, y, 
        ...                            model_type='lasso', 
        ...                            scoring=['neg_mean_squared_error', 'r2'])
    
    Notes:
        Uses parallel processing (n_jobs=-1) to speed up cross-validation.
    """
    # Default scoring metrics based on model type
    if scoring is None:
        if model_type.lower() == 'lasso':
            scoring = ['neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
        else:  # classifier
            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    
    # Perform cross-validation
    cv_results = sklearn_cross_validate(
        model, 
        X, 
        y, 
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Process and format results
    results = {
        'cv_folds': cv_folds,
        'test_scores': {},
        'train_scores': {}
    }
    
    # Extract test scores
    for metric in cv_results:
        if metric.startswith('test_'):
            metric_name = metric[5:]  # Remove 'test_' prefix
            mean_score = np.mean(cv_results[metric])
            std_score = np.std(cv_results[metric])
            results['test_scores'][metric_name] = {
                'mean': mean_score,
                'std': std_score,
                'values': list(cv_results[metric])
            }
        elif metric.startswith('train_'):
            metric_name = metric[6:]  # Remove 'train_' prefix
            mean_score = np.mean(cv_results[metric])
            std_score = np.std(cv_results[metric])
            results['train_scores'][metric_name] = {
                'mean': mean_score,
                'std': std_score,
                'values': list(cv_results[metric])
            }
    
    # For regression models, convert negative metrics back to positive
    if model_type.lower() == 'lasso':
        for score_type in ['test_scores', 'train_scores']:
            for metric in list(results[score_type].keys()):
                if metric.startswith('neg_'):
                    # Remove 'neg_' prefix and negate scores
                    new_metric = metric[4:]
                    results[score_type][new_metric] = {
                        'mean': -results[score_type][metric]['mean'],
                        'std': results[score_type][metric]['std'],
                        'values': [-val for val in results[score_type][metric]['values']]
                    }
    
    return results

def train_lasso_model(topics, returns, alpha_range=(ALPHA_MIN, ALPHA_MAX), n_alphas=N_ALPHAS,
                     test_size=TEST_SIZE, cv_folds=CV_FOLDS, save_model=True):
    """Train a Lasso regression model to evaluate which topics best predict stock returns.
    
    Uses randomized search cross-validation to find the optimal alpha parameter for Lasso
    regression. The trained model performs feature selection by assigning zero coefficients
    to irrelevant features, making it useful for identifying the most predictive topics.
    
    Args:
        topics (numpy.ndarray): Topic distribution matrix where each row is a document
            and each column represents the weight of a topic.
        returns (pandas.Series): Buy-and-hold abnormal returns (BHAR) - the target variable
            to predict.
        alpha_range (tuple): Min and max values for the alpha regularization hyperparameter.
            Higher alpha values lead to more sparse models (more zeroed coefficients).
            Default values are defined in config.
        n_alphas (int): Number of alpha values to test during hyperparameter tuning.
            Default is defined in config.
        test_size (float): Proportion of data to use for testing (0.0 to 1.0).
            Default is defined in config.
        cv_folds (int): Number of cross-validation folds for hyperparameter tuning.
            Default is defined in config.
        save_model (bool): Whether to save the model and results to disk. If True,
            saves the model, all coefficients, and non-zero coefficients to files.
        
    Returns:
        tuple: A tuple containing three elements:
            - best_lasso (Lasso): The trained Lasso model with optimal alpha
            - results (dict): Dictionary with training results including:
                - best_alpha: Optimal alpha value
                - nonzero_topics: Count of topics with non-zero coefficients
                - most_positive_topic: Index of topic with highest positive coefficient
                - most_positive_coef: Value of highest positive coefficient
                - most_negative_topic: Index of topic with lowest negative coefficient
                - most_negative_coef: Value of lowest negative coefficient
                - alphas: All alpha values tested
                - scores: Corresponding cross-validation scores
            - nonzero_topics (array): Indices of topics with non-zero coefficients
    
    Examples:
        >>> lasso_model, results, nonzero_topics = train_lasso_model(topic_matrix, returns)
        >>> print(f"Best alpha: {results['best_alpha']}")
        >>> print(f"Number of predictive topics: {results['nonzero_topics']}")
        >>> print(f"Most predictive topic index: {results['most_positive_topic']}")
    
    Notes:
        - Model and results are saved to directories specified in config (MODEL_DIR and OUTPUT_DIR)
        - Uses parallel processing for faster hyperparameter tuning
        - The model uses a high max_iter (10000) to ensure convergence
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
    """Plot the validation scores vs alpha values from Lasso regression tuning.
    
    Creates a scatter plot showing how performance varies with different alpha
    regularization parameters. This visualization helps identify regions with
    optimal alpha values and understand the model's sensitivity to regularization.
    
    Args:
        alphas (array): Alpha values tested during hyperparameter tuning.
        scores (array): Corresponding validation scores for each alpha value.
            These are typically negative RMSE scores (higher is better).
        save_plot (bool): Whether to save the plot to disk. If True, saves as
            'lasso_alpha_plot.png' in the directory specified by OUTPUT_DIR.
        
    Returns:
        matplotlib.pyplot.Figure: The plot figure object, which can be further
            customized or displayed.
    
    Examples:
        >>> # Create and display the alpha plot
        >>> fig = plot_lasso_alpha(results['alphas'], results['scores'])
        >>> plt.show()
        >>>
        >>> # Create the plot but don't save to disk
        >>> fig = plot_lasso_alpha(results['alphas'], results['scores'], save_plot=False)
    
    Notes:
        - The plot shows the relationship between regularization strength (alpha)
          and model performance
        - Lower negative RMSE values (higher on the y-axis) indicate better performance
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
    """Train multiple classifiers to predict large positive returns.
    
    Trains and evaluates several classifier models to predict whether a stock will
    have returns exceeding a specified threshold. The function compares LogisticRegression,
    SVM, RandomForest, and GradientBoosting classifiers, and returns the best performing
    model based on macro F1 score.
    
    Args:
        topics (numpy.ndarray): Topic distribution matrix where each row is a document
            and each column represents the weight of a topic.
        returns (pandas.Series): Buy-and-hold abnormal returns (BHAR) - used to create
            binary classification target.
        large_return_threshold (float): Threshold for considering returns as "large" (positive class).
            Returns above this value will be labeled as 1, others as 0.
            Default value is defined in config.
        test_size (float): Proportion of data to use for testing (0.0 to 1.0).
            Default is defined in config.
        n_iter_search (int): Number of iterations for RandomizedSearchCV hyperparameter tuning.
            Higher values explore more hyperparameter combinations but take longer.
            Default is defined in config.
        save_models (bool): Whether to save the models and results to disk. If True,
            saves all models and the best model to separate files.
        
    Returns:
        tuple: A tuple containing two elements:
            - best_model: The best performing classifier model
            - results (dict): Dictionary with results for each classifier model:
                {
                    'model_name': {
                        'best_params': optimal_hyperparameters,
                        'cv_score': cross_validation_score,
                        'test_f1_macro': test_set_f1_score,
                        'classification_report': detailed_classification_metrics
                    },
                    ...
                }
    
    Examples:
        >>> best_model, results = train_classifiers(topic_matrix, returns)
        >>> print(f"Best model test F1: {results[best_model_name]['test_f1_macro']:.4f}")
        >>> print(f"Best parameters: {results[best_model_name]['best_params']}")
    
    Notes:
        - Models compared: LogisticRegression, SVM, RandomForest, GradientBoosting
        - Uses parallel processing for faster hyperparameter tuning
        - Models are selected based on macro-averaged F1 score
        - All models and results are saved if save_models=True and MODEL_DIR/OUTPUT_DIR are defined
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
    """Print and return classification report for a model.
    
    Generates a detailed classification report with precision, recall, F1-score,
    and support metrics for each class. The report is both printed to console
    and returned as a dictionary for programmatic access.
    
    Args:
        model_name (str): Name of the model or identifier for the report.
        y_test (array): True labels for the test data.
        y_pred (array): Predicted labels from the model.
        
    Returns:
        dict: Classification report as a dictionary with the following structure:
            {
                'class_label': {
                    'precision': precision_score,
                    'recall': recall_score,
                    'f1-score': f1_score,
                    'support': number_of_samples
                },
                ...,
                'accuracy': overall_accuracy,
                'macro avg': {metrics_averaged_across_classes},
                'weighted avg': {metrics_weighted_by_support}
            }
    
    Examples:
        >>> y_pred = model.predict(X_test)
        >>> report_dict = print_classification_report('RandomForest', y_test, y_pred)
        >>> # Access programmatically:
        >>> class_1_f1 = report_dict['1']['f1-score']
        >>> overall_precision = report_dict['weighted avg']['precision']
    """
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    
    print(f"Classification Report for {model_name}:")
    print(report_str)
    
    return report_dict

def get_feature_importance(model, feature_names=None):
    """Extract feature importance from a trained model.
    
    Extracts and returns feature importance measures from various model types.
    Works with linear models (using coefficient magnitudes) and tree-based models
    (using feature_importances_).
    
    Args:
        model: Trained model with feature importance attribute. Supported models include:
            - Linear models with 'coef_' attribute (Lasso, LogisticRegression)
            - Tree-based models with 'feature_importances_' attribute (RandomForest, GradientBoosting)
        feature_names (list, optional): Names of features corresponding to the columns
            in the training data. If provided, the returned dictionary will map feature
            names to importance values. If None, features will be labeled "Feature 0", 
            "Feature 1", etc.
        
    Returns:
        dict: Dictionary mapping feature names/indices to importance values, sorted
            in descending order by importance. Returns None if the model doesn't
            have standard feature importance attributes.
    
    Examples:
        >>> # With default feature naming
        >>> importances = get_feature_importance(trained_model)
        >>> for feature, importance in list(importances.items())[:5]:
        ...     print(f"{feature}: {importance:.4f}")
        >>> 
        >>> # With custom feature names
        >>> topic_names = [f"Topic {i}" for i in range(50)]
        >>> importances = get_feature_importance(trained_model, topic_names)
    
    Notes:
        - For linear models, uses the absolute value of coefficients as importance
        - For multi-class classification models, takes the mean of coefficients across classes
        - Returns None for models without coef_ or feature_importances_ attributes
        - The returned dictionary is sorted by importance in descending order
    """
    importance_dict = {}
    
    # Extract feature importance based on model type
    if hasattr(model, 'coef_'):
        # Linear models like Lasso, LogisticRegression
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1:
            # For multi-class models, take the mean across classes
            importance = np.mean(importance, axis=0)
    
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models like RandomForest, GradientBoosting
        importance = model.feature_importances_
    
    else:
        return None  # Model doesn't have standard feature importance

    # Create feature importance dictionary
    if feature_names is not None and len(feature_names) == len(importance):
        for name, imp in zip(feature_names, importance):
            importance_dict[name] = float(imp)
    else:
        for i, imp in enumerate(importance):
            importance_dict[f"Feature {i}"] = float(imp)
    
    # Sort by importance (descending)
    importance_dict = {k: v for k, v in sorted(
        importance_dict.items(), key=lambda item: item[1], reverse=True
    )}
    
    return importance_dict

if __name__ == "__main__":
    """Main execution block when running this module directly.
    
    This script entry point performs a complete model training workflow:
    1. Loads preprocessed data and topic distributions
    2. Trains a Lasso regression model to identify important topics
    3. Plots Lasso regularization parameter tuning results
    4. Trains multiple classifier models to predict large returns
    5. Saves all models and results to disk
    
    Usage:
        python -m src.models.model_trainer
        
    Requirements:
        - Preprocessed data file (task2_data_clean.csv.gz)
        - Topic distributions (from feature extraction)
        - Configuration settings in src.config
        
    Outputs:
        - Trained models saved to MODEL_DIR
        - Results and plots saved to OUTPUT_DIR
    """
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