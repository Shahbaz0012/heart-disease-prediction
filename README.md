# Heart Disease Prediction

An end-to-end machine-learning capstone that estimates whether a patient record indicates heart disease. The project cleans the UCI Cleveland Heart Disease data, scales features, compares Logistic Regression and Decision Tree models, and produces accuracy, precision, recall, a confusion matrix, and an ROC curve.

> **Medical disclaimer:** This project is for education only. It is not a diagnostic device and must not guide medical decisions.

## Workflow

1. Download the public UCI Cleveland Heart Disease dataset.
2. Convert the UCI disease score (`num`) into a binary target.
3. Impute missing values with the median and standardize features.
4. Train Logistic Regression and a regularized Decision Tree.
5. Select the model with the highest ROC-AUC on a stratified test set.
6. Save the model and evaluation plots.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "from src.data import download_dataset; print(download_dataset())"
python -m src.train
```

The training command creates `models/heart_disease_model.joblib`, `reports/metrics.json`, `reports/confusion_matrix.png`, and `reports/roc_curve.png`.

## Predict one record

```bash
python -m src.predict --patient '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

## Dataset

The project downloads the Cleveland portion of the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), which follows the commonly used Kaggle Heart Disease UCI format. Raw data and trained models remain outside version control.

## Reproducibility

The pipeline uses random seed 42 and a stratified 75/25 train-test split. Run `pytest -q` to execute the included pipeline tests.
