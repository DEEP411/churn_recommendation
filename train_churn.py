"""Train churn prediction models.

This script generates a synthetic customer churn dataset, trains three
classifiers (logistic regression, random forest, and XGBoost) using
scikit‑learn and XGBoost, evaluates them on a hold‑out test set,
and reports the F1‑score for each model.  It also saves the trained
models and the dataset for future use.

Usage:
    python train_churn.py

Dependencies:
    numpy, pandas, scikit‑learn
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # XGBoost implementation
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 20, seed: int = 42):
    """Generate a synthetic binary classification dataset for churn prediction.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features.
        seed (int): Random seed for reproducibility.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary labels (0 = stay, 1 = churn).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        random_state=seed,
    )
    return X, y


def save_dataset(X: np.ndarray, y: np.ndarray, out_path: str) -> None:
    """Save the synthetic dataset to a CSV file.

    The output CSV will contain numeric columns for each feature and a
    `churn` column for the target variable.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        out_path (str): File path to save the CSV.
    """
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['churn'] = y
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to {out_path}")


def train_models(X: np.ndarray, y: np.ndarray):
    """Train logistic regression and random forest classifiers on the data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary labels.

    Returns:
        results (dict): Mapping of model name to (model_object, f1_score).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    results = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    f1_lr = f1_score(y_test, y_pred_lr)
    results['logistic_regression'] = (log_reg, f1_lr)

    # Random Forest (baseline)
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf)
    results['random_forest'] = (rf, f1_rf)

    # XGBoost classifier (gradient boosted trees)
    # Use a moderate number of estimators and balanced scale_pos_weight to handle class imbalance.
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric='logloss',
        random_state=0
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    results['xgboost'] = (xgb, f1_xgb)

    return results


def save_model(model, model_path: str) -> None:
    """Persist the trained model to disk using pickle.

    Args:
        model: Trained model object.
        model_path (str): File path to save the model.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def main():
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=2000, n_features=20, seed=42)

    # Save dataset
    data_file = os.path.join('data', 'churn_data.csv')
    save_dataset(X, y, data_file)

    # Train models
    results = train_models(X, y)

    # Save models and print F1 scores
    metrics = []
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)
    for name, (model, f1) in results.items():
        model_file = os.path.join(model_dir, f'{name}_model.pkl')
        save_model(model, model_file)
        metrics.append((name, f1))
    
    # Display metrics
    print("\nModel performance (F1‑score):")
    for name, f1 in metrics:
        print(f"  {name}: {f1:.3f}")

    # Save metrics to CSV for dashboarding
    metrics_df = pd.DataFrame(metrics, columns=['model', 'f1_score'])
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("Metrics saved to model_metrics.csv")


if __name__ == '__main__':
    main()