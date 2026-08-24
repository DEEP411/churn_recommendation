"""Train churn prediction models on the IBM Telco Customer Churn dataset.

This script loads the public IBM Telco Customer Churn dataset (7,043 real
customers, data/Telco-Customer-Churn.csv), encodes categorical features,
trains three classifiers (logistic regression, random forest, and XGBoost)
using scikit-learn and XGBoost, evaluates them on a hold-out test set, and
reports the F1-score and ROC-AUC for each model. It also saves the trained
models and per-model metrics for future use.

Dataset source:
    https://github.com/IBM/telco-customer-churn-on-icp4d
    (IBM sample telecom dataset, released for public analytics use)

Usage:
    python train_churn.py

Dependencies:
    numpy, pandas, scikit-learn, xgboost
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join('data', 'Telco-Customer-Churn.csv')


def load_real_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and clean the real Telco Customer Churn dataset.

    Args:
        path (str): Path to the Telco-Customer-Churn.csv file.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for feature encoding.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=['customerID'])
    # TotalCharges has a handful of blank strings for brand-new customers; treat as 0 tenure spend
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df


def encode_features(df: pd.DataFrame):
    """One-hot encode categorical columns and scale numeric columns.

    Args:
        df (pd.DataFrame): Cleaned Telco dataframe including the `Churn` target.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary churn labels.
        feature_names (list): Column names in X, in order.
    """
    y = df['Churn'].values
    X_df = df.drop(columns=['Churn'])

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
    X_encoded[numeric_cols] = StandardScaler().fit_transform(X_encoded[numeric_cols])

    return X_encoded.values.astype(float), y, list(X_encoded.columns)


def train_models(X: np.ndarray, y: np.ndarray):
    """Train logistic regression, random forest, and XGBoost classifiers.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary labels.

    Returns:
        results (dict): Mapping of model name to (model_object, f1_score, roc_auc).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    results = {}

    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
    results['logistic_regression'] = (log_reg, f1_score(y_test, y_pred_lr), roc_auc_score(y_test, y_prob_lr))

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    results['random_forest'] = (rf, f1_score(y_test, y_pred_rf), roc_auc_score(y_test, y_prob_rf))

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=0,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    results['xgboost'] = (xgb, f1_score(y_test, y_pred_xgb), roc_auc_score(y_test, y_prob_xgb))

    return results


def save_model(model, model_path: str) -> None:
    """Persist a trained model to disk using pickle."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def main():
    df = load_real_data()
    X, y, feature_names = encode_features(df)
    print(f"Loaded {len(df)} real customers, {y.sum()} churned ({y.mean():.1%} churn rate)")

    results = train_models(X, y)

    metrics = []
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)
    for name, (model, f1, auc) in results.items():
        model_file = os.path.join(model_dir, f'{name}_model.pkl')
        save_model(model, model_file)
        metrics.append((name, f1, auc))

    print("\nModel performance on the real Telco Customer Churn test set:")
    for name, f1, auc in metrics:
        print(f"  {name:<20} F1={f1:.3f}  ROC-AUC={auc:.3f}")

    metrics_df = pd.DataFrame(metrics, columns=['model', 'f1_score', 'roc_auc'])
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("Metrics saved to model_metrics.csv")


if __name__ == '__main__':
    main()
