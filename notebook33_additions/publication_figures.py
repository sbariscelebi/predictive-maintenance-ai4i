"""Publication-oriented plotting helpers used by the revised AI4I notebook.

Plot functions preserve supplied values and class order. They create untitled
figures with black, non-bold labels and save PNG and SVG copies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ("NF", "TWF", "HDF", "PWF", "OSF", "RNF")
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#6A3D9A")


def _style_context():
    return plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _save(fig, output_dir: str, stem: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def plot_roc_curves(
    y_true: Sequence[int],
    y_score: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    output_dir: str = "plots",
    output_stem: str = "roc_curve",
    show: bool = True,
):
    """Plot one-vs-rest ROC curves from the existing test labels and scores."""
    y_score = np.asarray(y_score)
    classes = np.arange(len(class_names))
    if y_score.ndim != 2 or y_score.shape[1] != len(class_names):
        raise ValueError("y_score columns must match the supplied class_names.")
    y_binary = label_binarize(y_true, classes=classes)
    with _style_context():
        fig, ax = plt.subplots(figsize=(14, 12))
        for index, name in enumerate(class_names):
            if np.unique(y_binary[:, index]).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_binary[:, index], y_score[:, index])
            ax.plot(
                fpr,
                tpr,
                color=PALETTE[index % len(PALETTE)],
                linewidth=3.5,
                linestyle=("-", "--", "-.", ":")[index % 4],
                label=f"Class {name} (AUC = {auc(fpr, tpr):.2f})",
            )
        ax.plot((0, 1), (0, 1), color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=34)
        ax.set_ylabel("True Positive Rate", fontsize=34)
        ax.tick_params(axis="both", labelsize=28, colors="black")
        ax.legend(fontsize=21, frameon=True, facecolor="white", edgecolor="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.2)
        fig.tight_layout()
        _save(fig, output_dir, output_stem)
        if show:
            plt.show()
        plt.close(fig)


def plot_normalized_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = CLASS_NAMES,
    output_dir: str = "plots",
    output_stem: str = "normalized_confusion_matrix",
    show: bool = True,
):
    """Plot row-normalized confusion values, retaining the given class order."""
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
    with _style_context():
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            linewidths=0.7,
            linecolor="black",
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 24, "color": "black", "weight": "normal"},
            cbar_kws={"shrink": 1.0},
            ax=ax,
        )
        ax.set_xlabel("Predicted Class", fontsize=30, labelpad=12)
        ax.set_ylabel("True Class", fontsize=30, labelpad=12)
        ax.tick_params(axis="both", labelsize=25, colors="black", rotation=0)
        ax.collections[0].colorbar.ax.tick_params(labelsize=22, colors="black")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.2)
        fig.tight_layout(pad=2)
        _save(fig, output_dir, output_stem)
        if show:
            plt.show()
        plt.close(fig)


def plot_training_history(
    history,
    output_dir: str = "plots",
    show: bool = True,
):
    """Save loss and accuracy as two separate, untitled figures."""
    panels = (("loss", "val_loss", "Loss", "training_loss"),
              ("accuracy", "val_accuracy", "Accuracy", "training_accuracy"))
    with _style_context():
        for train_key, val_key, ylabel, stem in panels:
            if train_key not in history.history or val_key not in history.history:
                continue
            train_values = history.history[train_key]
            val_values = history.history[val_key]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(np.arange(len(train_values)), train_values, color="blue", linestyle="--",
                    marker="o", linewidth=3.2, markersize=8, label="Training")
            ax.plot(np.arange(len(val_values)), val_values, color="red", linestyle="--",
                    marker="o", linewidth=3.2, markersize=8, label="Validation")
            ax.set_xlabel("Epoch", fontsize=32, labelpad=12)
            ax.set_ylabel(ylabel, fontsize=32, labelpad=12)
            ax.tick_params(axis="both", labelsize=28, colors="black")
            ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.7)
            ax.set_axisbelow(True)
            ax.legend(fontsize=25, frameon=True, facecolor="white", edgecolor="black")
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.2)
            fig.tight_layout(pad=1.5)
            _save(fig, output_dir, stem)
            if show:
                plt.show()
            plt.close(fig)


def plot_grid_search_heatmap(
    results: pd.DataFrame,
    output_dir: str = "plots",
    output_stem: str = "grid_search_heatmap",
    show: bool = True,
):
    """Plot mean validation macro-F1 by hidden units and learning rate."""
    required = {"units", "learning_rate", "val_score"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Grid-search results are missing columns: {sorted(missing)}")
    pivot = results.pivot_table(values="val_score", index="units", columns="learning_rate")
    with _style_context():
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            linewidths=1.5,
            linecolor="white",
            annot_kws={"size": 28, "weight": "normal"},
            cbar_kws={"shrink": 1.0, "aspect": 10, "label": "Mean validation macro-F1"},
            ax=ax,
        )
        ax.set_xlabel("Learning Rate", fontsize=34, labelpad=12)
        ax.set_ylabel("Units", fontsize=34, labelpad=12)
        ax.set_xticklabels([f"{value:g}" for value in pivot.columns], fontsize=28, rotation=0)
        ax.set_yticklabels(pivot.index, fontsize=28, rotation=0)
        ax.tick_params(axis="both", length=0, pad=8)
        for label, value in zip(ax.texts, pivot.to_numpy().ravel()):
            label.set_color("white" if ax.collections[0].norm(value) > 0.55 else "black")
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=24, colors="black")
        colorbar.ax.yaxis.label.set_size(25)
        fig.tight_layout(pad=1.5)
        _save(fig, output_dir, output_stem)
        if show:
            plt.show()
        plt.close(fig)


def plot_torque_boxplot(
    data: pd.DataFrame,
    class_names: Sequence[str] = CLASS_NAMES,
    class_column: str = "Machine failure",
    torque_column: str = "Torque [Nm]",
    output_dir: str = "plots",
    output_stem: str = "torque_boxplot",
    show: bool = True,
):
    """Plot torque distributions using the requested class order and no title."""
    missing = {class_column, torque_column}.difference(data.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")
    label_map = {index: name for index, name in enumerate(class_names)}
    frame = data[[class_column, torque_column]].copy()
    frame[class_column] = frame[class_column].replace(label_map)
    counts = frame[class_column].value_counts().to_dict()
    tick_labels = [f"{name}\n(n={counts.get(name, 0):,})" for name in class_names]
    with _style_context():
        fig, ax = plt.subplots(figsize=(15, 9))
        sns.boxplot(
            data=frame,
            x=class_column,
            y=torque_column,
            order=list(class_names),
            palette="muted",
            width=0.65,
            linewidth=1.5,
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "white",
                        "markeredgecolor": "#333333", "markeredgewidth": 0.8},
            ax=ax,
        )
        ax.set_xlabel("Failure Classes", fontsize=36, labelpad=16)
        ax.set_ylabel(torque_column, fontsize=36, labelpad=16)
        ax.set_xticklabels(tick_labels, fontsize=26, weight="normal")
        ax.tick_params(axis="y", labelsize=26, colors="black", length=6, width=1.2)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.3)
        fig.tight_layout(pad=2)
        _save(fig, output_dir, output_stem)
        if show:
            plt.show()
        plt.close(fig)


def plot_correlation_heatmap(
    data: pd.DataFrame,
    output_dir: str = "plots",
    output_stem: str = "correlation_heatmap",
    show: bool = True,
):
    """Plot correlations among numeric columns without changing source data."""
    correlation = data.select_dtypes(include=[np.number]).corr()
    if correlation.empty:
        raise ValueError("No numeric columns are available for the correlation plot.")
    with _style_context():
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f",
                    annot_kws={"size": 16, "color": "black", "weight": "normal"}, ax=ax)
        ax.set_xlabel("Features", fontsize=30, labelpad=12)
        ax.set_ylabel("Features", fontsize=30, labelpad=12)
        ax.tick_params(axis="both", labelsize=22, colors="black")
        fig.tight_layout()
        _save(fig, output_dir, output_stem)
        if show:
            plt.show()
        plt.close(fig)


def plot_shap_summary(
    shap_values,
    X_values: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str] = CLASS_NAMES,
    output_dir: str = "plots",
    output_stem: str = "shap_summary",
    show: bool = True,
):
    """Render already-computed SHAP values; this function does not recompute them."""
    import shap

    with _style_context():
        shap.summary_plot(
            shap_values,
            X_values,
            feature_names=list(feature_names),
            show=False,
            color_bar=True,
            cmap="viridis",
            plot_type="bar",
            class_names=list(class_names),
            plot_size=(16, 12),
        )
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("")
        ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=30, fontweight="normal")
        ax.set_ylabel("Features", fontsize=30, fontweight="normal")
        ax.tick_params(axis="x", labelsize=24, colors="black")
        ax.tick_params(axis="y", labelsize=24, colors="black", length=0)
        ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(24)
        fig.subplots_adjust(left=0.40, right=0.98, bottom=0.14, top=0.98)
        _save(fig, output_dir, output_stem)
        if show:
            plt.show()
        plt.close(fig)

