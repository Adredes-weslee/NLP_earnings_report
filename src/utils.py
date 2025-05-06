# utils.py
# Utility functions for visualization and reporting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR

def plot_topic_words(lda_model, vocab, topics_to_show=None, n_words=10, figsize=(18, 10),
                    save_plot=True):
    """
    Visualize the top words for each topic
    
    Args:
        lda_model: Fitted LDA model
        vocab (list): List of vocabulary terms
        topics_to_show (list, optional): List of topic indices to show. If None, shows all topics
        n_words (int): Number of top words to show per topic
        figsize (tuple): Figure size
        save_plot (bool): Whether to save the plot
        
    Returns:
        matplotlib.pyplot.Figure: The plot figure
    """
    if topics_to_show is None:
        n_topics = len(lda_model.components_)
        topics_to_show = list(range(min(n_topics, 20)))  # Limit to 20 topics max by default
    
    n_topics_to_show = len(topics_to_show)
    cols = min(4, n_topics_to_show)
    rows = (n_topics_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, topic_idx in enumerate(topics_to_show):
        if i >= len(axes):
            break
            
        top_indices = lda_model.components_[topic_idx].argsort()[-(n_words):][::-1]
        top_words = [vocab[j] for j in top_indices]
        top_weights = [lda_model.components_[topic_idx][j] for j in top_indices]
        
        ax = axes[i]
        y_pos = np.arange(len(top_words))
        ax.barh(y_pos, top_weights, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title(f'Topic {topic_idx}')
        ax.set_xlabel('Weight')
    
    # Hide unused subplots
    for i in range(n_topics_to_show, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'topic_words_visualization.png'), 
                   dpi=300, bbox_inches='tight')
    
    return fig

def plot_lasso_coefficients(coefficients, top_n=20, save_plot=True):
    """
    Visualize the most positive and negative coefficients from Lasso regression
    
    Args:
        coefficients (array): Array of coefficient values
        top_n (int): Number of top positive and negative coefficients to show
        save_plot (bool): Whether to save the plot
        
    Returns:
        matplotlib.pyplot.Figure: The plot figure
    """
    coef_df = pd.DataFrame({
        'Topic': range(len(coefficients)),
        'Coefficient': coefficients
    }).sort_values('Coefficient')
    
    # Get top positive and negative coefficients
    top_neg = coef_df.head(top_n).copy()
    top_pos = coef_df.tail(top_n).copy().iloc[::-1]  # Reverse to show highest at the top
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot negative coefficients
    ax1.barh(top_neg['Topic'].astype(str), top_neg['Coefficient'], color='r')
    ax1.set_title(f'Top {top_n} Negative Coefficients')
    ax1.set_xlabel('Coefficient Value')
    ax1.set_ylabel('Topic')
    
    # Plot positive coefficients
    ax2.barh(top_pos['Topic'].astype(str), top_pos['Coefficient'], color='g')
    ax2.set_title(f'Top {top_n} Positive Coefficients')
    ax2.set_xlabel('Coefficient Value')
    
    plt.tight_layout()
    
    if save_plot and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'lasso_coefficients_plot.png'), 
                   dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrices(models_dict, X_test, y_test, save_plot=True):
    """
    Plot confusion matrices for multiple classifier models
    
    Args:
        models_dict (dict): Dictionary mapping model names to fitted model objects
        X_test (array): Test features
        y_test (array): Test labels
        save_plot (bool): Whether to save the plot
        
    Returns:
        matplotlib.pyplot.Figure: The plot figure
    """
    n_models = len(models_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (name, model) in enumerate(models_dict.items()):
        if i >= len(axes):
            break
            
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
    
    return fig

def plot_roc_curves(models_dict, X_test, y_test, save_plot=True):
    """
    Plot ROC curves for multiple classifier models
    
    Args:
        models_dict (dict): Dictionary mapping model names to fitted model objects
        X_test (array): Test features
        y_test (array): Test labels
        save_plot (bool): Whether to save the plot
        
    Returns:
        matplotlib.pyplot.Figure: The plot figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if save_plot and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def generate_summary_report(lda_results, lasso_results, classifier_results, 
                          topics_words, save_report=True):
    """
    Generate a summary report of the analysis results
    
    Args:
        lda_results (dict): Results from LDA topic modeling
        lasso_results (dict): Results from Lasso regression
        classifier_results (dict): Results from classifier training
        topics_words (dict): Dictionary mapping topic indices to lists of top words
        save_report (bool): Whether to save the report
        
    Returns:
        str: The report text
    """
    report = []
    report.append("# Earnings Announcement Analysis Report")
    report.append("\n## 1. Topic Modeling Results")
    report.append(f"- Number of topics: {lda_results.get('n_topics', 'N/A')}")
    report.append(f"- Coherence score: {lda_results.get('coherence_score', 'N/A'):.4f}")
    
    report.append("\n### Top topics and their most representative words:")
    for topic_idx, words in topics_words.items():
        if topic_idx in [lasso_results.get('most_positive_topic'), lasso_results.get('most_negative_topic')]:
            report.append(f"- Topic {topic_idx}: {', '.join(words)} " + 
                        ("(Most positive impact on returns)" if topic_idx == lasso_results.get('most_positive_topic') 
                         else "(Most negative impact on returns)"))
    
    report.append("\n## 2. Lasso Regression Results")
    report.append(f"- Best alpha: {lasso_results.get('best_alpha', 'N/A'):.6f}")
    report.append(f"- Number of topics with non-zero coefficients: {lasso_results.get('nonzero_topics', 'N/A')}")
    report.append(f"- Topic with most positive coefficient: Topic {lasso_results.get('most_positive_topic', 'N/A')} " +
                f"(Coefficient: {lasso_results.get('most_positive_coef', 'N/A'):.4f})")
    report.append(f"- Topic with most negative coefficient: Topic {lasso_results.get('most_negative_topic', 'N/A')} " +
                f"(Coefficient: {lasso_results.get('most_negative_coef', 'N/A'):.4f})")
    
    report.append("\n## 3. Classification Results")
    best_model = max(classifier_results, key=lambda x: classifier_results[x]['test_f1_macro'])
    report.append(f"- Best performing model: {best_model}")
    report.append(f"- Macro F1 score: {classifier_results[best_model]['test_f1_macro']:.4f}")
    
    report.append("\nClassification performance by model:")
    for model_name, results in classifier_results.items():
        report.append(f"- {model_name}: F1 = {results['test_f1_macro']:.4f}")
    
    report.append("\n## 4. Conclusion")
    report.append("The analysis of earnings announcements has revealed that certain topics " +
                "have significant predictive power for stock returns. The Lasso regression " +
                "identified specific language patterns that are associated with positive and " +
                "negative market reactions.")
    
    report.append("\nThe best classifier model is able to predict large positive returns " +
                f"with a macro F1 score of {classifier_results[best_model]['test_f1_macro']:.4f}, " +
                "suggesting that the textual content of earnings announcements contains " +
                "valuable information that can be leveraged for predicting market reactions.")
    
    report_text = "\n".join(report)
    
    if save_report and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, 'analysis_report.md'), 'w') as f:
            f.write(report_text)
    
    return report_text

if __name__ == "__main__":
    print("This module provides utility functions for visualization and reporting.")
    print("It is not meant to be run directly.")