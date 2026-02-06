"""
Utility Functions for ICU Mortality Prediction
===============================================
Helper functions and classes for data processing, modeling, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


def missing_summary(df, threshold=0):
    """
    Display summary of missing values in a DataFrame.
    
    Use: Provides a quick overview of which columns have missing data
    Inputs: 
        - df: pandas DataFrame
        - threshold: int, only show columns with > threshold missing values (default: 0)
    Outputs: 
        - pandas DataFrame with columns: 'count' and 'percent'
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'count': missing,
        'percent': missing_pct
    }).sort_values('percent', ascending=False)
    
    missing_df = missing_df[missing_df['count'] > threshold]
    
    if len(missing_df) == 0:
        print("No missing values found!")
    else:
        print(f"Columns with missing values: {len(missing_df)}")
    
    return missing_df


def plot_class_distribution(y, title='Class Distribution'):
    """
    Plot bar chart and pie chart for binary classification target.
    
    Use: Visualize class imbalance in target variable
    Inputs:
        - y: pandas Series or numpy array, target variable (0/1)
        - title: str, plot title
    Outputs:
        - matplotlib figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count values
    counts = pd.Series(y).value_counts().sort_index()
    
    # Bar plot
    axes[0].bar(range(len(counts)), counts.values, 
                color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(['Survival (0)', 'Death (1)'])
    axes[0].set_ylabel('Count')
    axes[0].set_title(title)
    
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts)*0.01, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=['Survival', 'Death'],
                colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
                startangle=90)
    axes[1].set_title('Class Balance')
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=['Survival', 'Death']):
    """
    Plot confusion matrix with annotations.
    
    Use: Visualize model predictions vs actual values
    Inputs:
        - y_true: array-like, true labels
        - y_pred: array-like, predicted labels
        - labels: list of str, class names
    Outputs:
        - matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve with AUC score.
    
    Use: Evaluate classifier performance across different thresholds
    Inputs:
        - y_true: array-like, true binary labels
        - y_pred_proba: array-like, predicted probabilities for positive class
    Outputs:
        - matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})', color='#3498db')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance from tree-based models.
    
    Use: Identify most important features in prediction
    Inputs:
        - model: trained sklearn model with feature_importances_ attribute
        - feature_names: list of str, feature names
        - top_n: int, number of top features to display
    Outputs:
        - matplotlib figure
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices], color='#3498db', edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Comprehensive model evaluation with metrics and plots.
    
    Use: Generate evaluation report for a trained classifier
    Inputs:
        - model: trained sklearn classifier
        - X_test: test features (DataFrame or array)
        - y_test: true test labels
        - model_name: str, name of model for display
    Outputs:
        - dict with metrics: accuracy, f1, auc, predictions
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"=== {model_name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survival', 'Death']))
    
    # Store results
    results = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results


def compare_models(results_dict):
    """
    Compare multiple models side-by-side.
    
    Use: Visualize performance comparison across different models
    Inputs:
        - results_dict: dict of dicts, {model_name: {metric: value, ...}}
    Outputs:
        - pandas DataFrame with comparison table
    """
    comparison = pd.DataFrame(results_dict).T
    comparison = comparison[['accuracy', 'f1_score', 'roc_auc']].round(4)
    comparison.columns = ['Accuracy', 'F1-Score', 'ROC AUC']
    
    # Highlight best performers
    print("=== Model Comparison ===")
    print(comparison.to_string())
    print("\nBest performers:")
    for col in comparison.columns:
        best_model = comparison[col].idxmax()
        best_value = comparison[col].max()
        print(f"  {col}: {best_model} ({best_value:.4f})")
    
    return comparison


# Import commonly used libraries so they're available when doing: from utility_functions import *
__all__ = [
    'missing_summary',
    'plot_class_distribution',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_importance',
    'evaluate_model',
    'compare_models',
    'np',
    'pd',
    'plt',
    'sns'
]
