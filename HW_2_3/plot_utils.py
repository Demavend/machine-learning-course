import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, f1_score

def plot_custom_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Plot a styled confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    title : str, optional
        Plot title, by default "Confusion Matrix".
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                annot_kws={"size": 14, "color": "black"})
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, title: str = "ROC Curve") -> None:
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_probs : np.ndarray
        Predicted probabilities for the positive class.
    title : str, optional
        Plot title, by default "ROC Curve".
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_main_metrics(y_true, y_proba, y_pred, prefix="", silent=False):
    """
    Print (or return) AUROC and F1-score for classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_proba : array-like
        Predicted probabilities for positive class.
    y_pred : array-like
        Predicted class labels.
    prefix : str
        Prefix for printing.
    silent : bool
        If True, suppress printing.

    Returns
    -------
    dict
        Dictionary with 'auc' and 'f1' scores.
    """
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)

    if not silent:
        print(f"{prefix} AUROC: {auc:.3f}")
        print(f"{prefix} F1 score: {f1:.3f}")

    return {"auc": auc, "f1": f1}

def plot_feature_importances(df: pd.DataFrame, top_n: int = 10, title: str = "Top Feature Importances") -> None:
    """
    Plot top N feature importances as a horizontal bar plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns.
    top_n : int, optional
        Number of top features to show, by default 10.
    title : str, optional
        Plot title, by default "Top Feature Importances".
    """
    top_df = df.sort_values(by="importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    sns.barplot(x="importance", y="feature", data=top_df, palette="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.show()
