# Customer Churn Prediction & Recommendation System

This project implements an end-to-end pipeline for predicting customer churn and recommending personalized services. It demonstrates two capabilities:

* **Churn Prediction** – training machine-learning models (logistic regression, random forest, and XGBoost) on the real **IBM Telco Customer Churn** dataset (7,043 customers) to predict which customers are likely to leave a service.
* **Recommendation Engine** – building a collaborative-filtering model that suggests new services to existing customers, evaluated with a real precision@k / recall@k hold-out test.

The code is written entirely in Python using scikit-learn for the predictive models. It runs locally on a laptop without requiring big-data infrastructure. The scripts are structured so that they can be ported to distributed environments such as Apache Spark on AWS EMR if needed — that part hasn't been done, so treat it as a known extension, not a claim.

## Results

**Churn prediction** (real Telco data, 20% held-out test set, 26.5% actual churn rate):

| Model | F1-score | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.634 | 0.851 |
| Random Forest | **0.643** | 0.851 |
| XGBoost | 0.623 | 0.844 |

Random Forest edges out the others on F1; all three land around 0.85 ROC-AUC, a reasonable result on this dataset without heavy feature engineering.

**Recommendation engine** (synthetic interaction matrix, 200 users, precision@3 / recall@3 against held-out interactions):

```
Precision@3: 0.092   Recall@3: 0.276
Random-guess baseline precision: 0.304
Lift over random baseline: 0.30x
```

Worth being upfront about: on this synthetic matrix, the collaborative filter performs **below** the random baseline. That's the honest, expected result — user-based collaborative filtering only beats random guessing when there's real behavioral correlation between users, and a randomly generated interaction matrix has none by construction. The recommendation half of this project stays synthetic because there's no public dataset pairing individual Telco customers with which add-on services they bought — there's nothing real to plug in. The evaluation code (`precision_recall_at_k` in `recommendation.py`) is the reusable part; point it at real interaction data and the numbers become meaningful.

## Requirements

```
numpy
pandas
scikit-learn
xgboost
```

Install with:

```bash
pip install numpy pandas scikit-learn xgboost
```

## Project Structure

```
churn_recommendation/
├── data/
│   └── Telco-Customer-Churn.csv   # real IBM Telco Customer Churn dataset (7,043 rows)
├── train_churn.py                 # loads real data, trains 3 churn models, reports F1/ROC-AUC
├── recommendation.py              # collaborative filtering + real precision@k/recall@k evaluation
├── requirements.txt               # pinned dependencies
└── README.md                      # this file
```

Running either script also creates `models/` (pickled trained models) and `model_metrics.csv` — both are generated output, not checked in.

## Usage

1. **Train churn models**

   ```bash
   cd churn_recommendation
   python train_churn.py
   ```

   Loads `data/Telco-Customer-Churn.csv`, one-hot encodes categorical fields, scales numeric ones, trains all three models, and prints F1/ROC-AUC for each.

2. **Run the recommendation engine**

   ```bash
   python recommendation.py
   ```

   Generates a synthetic user-service interaction matrix, holds out 20% of each user's known interactions, computes user-based cosine-similarity recommendations, and reports precision@3/recall@3 against what was held out.

## Notes for anyone reviewing this

* Churn prediction runs on **real, public data** — the numbers above are real, not illustrative.
* The recommendation engine's *input data* is synthetic (explained above), but the *evaluation methodology* — mask known positives, see if the model recovers them — is a standard, real way to score a recommender, not a made-up proxy metric.
* To integrate with a BI tool like Tableau, export `model_metrics.csv` (from `train_churn.py`) or the recommendation output into a CSV and load it there.
