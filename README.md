# Predictive Maintenance on AI4I 2020 Dataset

This repository provides a Transformer-based deep learning framework for multiclass predictive maintenance, evaluated on the AI4I 2020 dataset. It includes feature engineering, ADASYN-based balancing, Random Forest-based feature selection, and explainable AI with SHAP.

## 🚀 Features
- Sliding window time-series feature engineering
- Class imbalance handling with ADASYN
- Random Forest-based feature selection
- Custom weighted categorical crossentropy loss
- Transformer architecture with hyperparameter grid search
- SHAP explainability and detailed visualizations

## 📁 Folder Structure
```
src/
  model.py
  utils.py
notebooks/
  ai4i_analysis.ipynb
README.md
LICENSE
.gitignore
requirements.txt
```

## ⚡ Installation
First, install the requirements in a clean environment:
```bash
pip install -r requirements.txt
```
> **Note:** For best results, use Python 3.10+ and recent versions of TensorFlow and scikit-learn.

## ▶️ Run
```bash
python src/model.py
```
The code will train the Transformer, perform grid search, generate plots, and report metrics.

## 📊 Dataset
- [AI4I 2020 Predictive Maintenance Dataset (UCI)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- Please download `ai4i2020.csv` and place it in the project root.

## 📜 Citation
If you use this work in your research, please cite:

```
@misc{celebi2025transformer,
  title={Transformer-Based Predictive Maintenance Framework Using AI4I Dataset},
  author={Baris Celebi},
  year={2025},
  note={GitHub: https://github.com/sbariscelebi/predictive-maintenance-ai4i}
}
```

## 📄 License
This project is licensed under the MIT License.

---

> For questions or suggestions, please open an issue or contact [Baris Celebi](mailto:your.email@example.com).
