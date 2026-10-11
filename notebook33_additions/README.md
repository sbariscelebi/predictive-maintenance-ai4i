# Notebook 33 supplementary modules

This folder separates the RNF threshold sensitivity analysis and publication figure helpers used with `muazzez-project33.ipynb` from the notebook's main training cell.

## RNF threshold analysis

After obtaining the unchanged model probabilities and test labels, call:

```python
from notebook33_additions.rnf_threshold_sensitivity import analyze_rnf_thresholds

results, predictions_by_rule = analyze_rnf_thresholds(
    y_true=y_test,
    y_score=y_score,
    thresholds=(0.30, 0.50, 0.70, 0.90),
    rnf_class_id=5,
    normal_class_id=0,
    output_csv="tables/rnf_threshold_sensitivity.csv",
)
```

The function preserves the original argmax result. It applies each threshold only to predictions that argmax assigned to RNF. If the RNF score is below the threshold, the sample is assigned to the highest-scoring non-RNF class. The printed table shows two decimal places. The CSV keeps full precision. This is a sensitivity analysis on supplied data and does not identify an optimal deployment threshold.

## Figure helpers

`publication_figures.py` contains reusable helpers for ROC curves, normalized confusion matrices, separate training and validation loss/accuracy figures, grid-search heatmaps, torque boxplots, correlation heatmaps, and already-computed SHAP values. They preserve the provided data and class order, use untitled figures with readable non-bold labels, and save PNG and SVG files. The SHAP helper only plots supplied SHAP values; it does not run an explanation calculation.

The notebook remains the source for the training workflow and reported experimental results. These helpers do not retrain a model. Install the repository's requirements before importing them. Full model training and result reproduction require the dataset and compute setup described in the main project README.

