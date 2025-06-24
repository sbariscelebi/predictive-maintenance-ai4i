import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import shap

def plot_roc_curves(y_test, y_score, all_classes, class_names, output_dir='plots', output_png='roc_curve.png', output_svg='roc_curve.svg'):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    y_test_bin = label_binarize(y_test, classes=all_classes)
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set2", len(class_names))
    for i, cls in enumerate(class_names):
        if np.sum(y_test_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5, color=colors[i], label=f'Class {cls} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5, alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_png), dpi=300)
    plt.savefig(os.path.join(output_dir, output_svg), format='svg')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, output_path='plots/normalized_confusion_matrix.png'):
    cf_mtx = confusion_matrix(y_true, y_pred)
    cf_mtx_norm = cf_mtx.astype('float') / cf_mtx.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_mtx_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_shap_summary(best_model, X_train_res, X_test, selected_features, window_size, class_names_all):
    def model_predict_2d(x):
        x_3d = x.reshape(-1, window_size, len(selected_features))
        return best_model.predict(x_3d)
    explainer = shap.KernelExplainer(model_predict_2d, X_train_res[:100].reshape(100, -1))
    shap_values = explainer.shap_values(X_test[:10].reshape(10, -1))
    feature_names = [f'{feat}_t{i}' for feat in selected_features for i in range(window_size)]
    shap.summary_plot(shap_values, X_test[:10].reshape(10, -1), feature_names=feature_names, show=False,
                      color_bar=True, cmap='viridis', plot_type="bar", class_names=class_names_all)
    plt.title('Feature Importance by SHAP Values Across Failure Classes')
    plt.savefig('plots/shap_summary_enhanced.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/shap_summary_enhanced.svg', format='svg', bbox_inches='tight')
    plt.show()
