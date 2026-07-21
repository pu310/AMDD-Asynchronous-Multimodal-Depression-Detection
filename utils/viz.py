"""Confusion matrix plotting utilities for evaluation results."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(TP, FP, TN, FN, title="Confusion Matrix", filename="confusion_matrix.png"):
    res_dir = "./results/conf_mat/"
    os.makedirs(res_dir, exist_ok=True)
    cm = [[TN, FP], [FN, TP]]

    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=["normal", "depression"],
        yticklabels=["normal", "depression"]
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.savefig(os.path.join(res_dir, filename), dpi=300)
    plt.close()


def plot_confusion_matrix_mean(TP, FP, TN, FN, title="Confusion Matrix", filename="confusion_matrix.png"):
    res_dir = "./results/conf_mat/"
    os.makedirs(res_dir, exist_ok=True)
    cm = [[TN, FP], [FN, TP]]
    cm = np.array(cm).astype('float') / np.array(cm).sum(axis=1)[:, np.newaxis]

    ax = sns.heatmap(
        cm, annot=True, fmt='.3f', cmap='Blues', cbar=False,
        xticklabels=["normal", "depression"],
        yticklabels=["normal", "depression"]
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.savefig(os.path.join(res_dir, filename), dpi=300)
    plt.close()
