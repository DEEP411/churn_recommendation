# Customer Churn Prediction & Recommendation System

This project implements a simple end‑to‑end pipeline for predicting customer churn and recommending personalized services.  
It demonstrates two key capabilities:

* **Churn Prediction** – training machine‑learning models (logistic regression, random forest, and XGBoost) to predict which customers are likely to leave a service.
* **Recommendation Engine** – building a collaborative‑filtering model that suggests new products or services to existing customers.

The code is written entirely in Python using scikit‑learn for the predictive models.  It runs locally on a laptop without requiring big‑data infrastructure.  However, the scripts are structured so that they can be ported to distributed environments such as Apache Spark on AWS EMR if needed.

## Requirements

The project uses only a few common Python packages:

```
numpy
pandas
scikit‑learn
xgboost
```

If you don’t already have them installed, install with:

```
pip install numpy pandas scikit‑learn
```

## Project Structure

```
churn_recommendation/
├── data/
│   └── churn_data.csv            # synthetic customer dataset
├── train_churn.py               # trains churn models (logistic regression & random forest)
├── recommendation.py            # builds simple collaborative‑filtering recommendations
├── requirements.txt             # pinned dependencies
└── README.md                    # this file
```

## Usage

1. **Generate data and train churn models**

   Run `train_churn.py` to generate a synthetic dataset, train logistic regression, random forest, and XGBoost models, and evaluate them.  It prints the F1‑scores for each model and saves the models to disk.

   ```bash
   cd churn_recommendation
   python train_churn.py
   ```

2. **Build recommendation engine**

   Run `recommendation.py` to create a synthetic user‑item interaction matrix, compute user‑based collaborative filtering recommendations, and print example recommendations for a sample user.  It also reports a simple engagement metric comparing recommended versus baseline interactions.

   ```bash
   python recommendation.py
   ```

## Notes

* The code uses synthetic data for demonstration; real churn prediction would require real customer data, feature engineering, and careful validation.
* The recommendation engine is a simple user‑based collaborative filter using cosine similarity; more advanced techniques (matrix factorization, implicit feedback models) can be substituted.
* To integrate with a business intelligence tool like Tableau, export the `metrics_df` DataFrame from `train_churn.py` or the recommendation outputs from `recommendation.py` as CSV files and load them into Tableau.
* Although the project references Spark and AWS EMR in the resume, this implementation runs locally with scikit‑learn.  To port to Spark, consider rewriting the training pipeline using PySpark MLlib and running on a Spark cluster.