import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_model_comparison(results, output_path):
    model_names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(model_names))
    width = 0.8 / len(metric_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("viridis", len(metric_names))

    for i, metric in enumerate(metric_names):
        values = [results[name][metric] for name in model_names]
        ax.bar(x + i * width, values, width=width, label=metric.capitalize(), color=palette[i])

    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
