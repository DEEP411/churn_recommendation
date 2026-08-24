"""User-based collaborative filtering recommendation engine, with a real evaluation.

This script generates a synthetic user-service interaction matrix (there is
no public per-user "which telecom add-ons did they buy" dataset paired with
the Telco churn data, so a synthetic interaction matrix is the honest choice
here), computes cosine similarities between users, and recommends items a
user hasn't interacted with yet.

To measure whether the recommendations are actually any good, the script
holds out 20% of each user's known positive interactions before training,
then checks whether the top-N recommendations recover those held-out items.
That gives real precision@k / recall@k numbers instead of an unlabeled proxy
metric.

Usage:
    python recommendation.py

Dependencies:
    numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def generate_interactions(num_users: int = 200, num_items: int = 15, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic user-item interaction matrix.

    Args:
        num_users (int): Number of users.
        num_items (int): Number of items/services.
        seed (int): Random seed for reproducibility.

    Returns:
        interactions (pd.DataFrame): DataFrame with shape (num_users, num_items).
    """
    rng = np.random.default_rng(seed)
    interactions = rng.random((num_users, num_items)) < 0.3
    df = pd.DataFrame(interactions.astype(int), columns=[f'service_{i}' for i in range(num_items)])
    df.index.name = 'user_id'
    return df


def holdout_split(interactions: pd.DataFrame, holdout_frac: float = 0.2, seed: int = 0):
    """Mask a fraction of each user's positive interactions for evaluation.

    Args:
        interactions (pd.DataFrame): Full binary user-item interaction matrix.
        holdout_frac (float): Fraction of each user's positive items to hide.
        seed (int): Random seed.

    Returns:
        train (pd.DataFrame): Interaction matrix with held-out items zeroed out.
        held_out (dict): user_id -> list of item columns that were hidden.
    """
    rng = np.random.default_rng(seed)
    train = interactions.copy()
    held_out = {}
    for user in interactions.index:
        positives = interactions.columns[interactions.loc[user] == 1].tolist()
        n_hide = max(1, int(len(positives) * holdout_frac)) if positives else 0
        if n_hide == 0:
            held_out[user] = []
            continue
        hidden = list(rng.choice(positives, size=min(n_hide, len(positives)), replace=False))
        train.loc[user, hidden] = 0
        held_out[user] = hidden
    return train, held_out


def compute_user_similarity(interactions: pd.DataFrame) -> np.ndarray:
    """Compute cosine similarity between users based on their item interactions."""
    return cosine_similarity(interactions)


def predict_scores(interactions: pd.DataFrame, similarity: np.ndarray) -> pd.DataFrame:
    """Predict user-item scores for items not yet interacted with."""
    np.fill_diagonal(similarity, 0)
    preds = similarity.dot(interactions.values) / np.maximum(similarity.sum(axis=1)[:, None], 1e-9)
    return pd.DataFrame(preds, index=interactions.index, columns=interactions.columns)


def recommend_top_n(predictions: pd.DataFrame, interactions: pd.DataFrame, n: int = 3) -> dict:
    """Recommend top-N items for each user based on predicted scores."""
    recommendations = {}
    for user in predictions.index:
        uninteracted = interactions.loc[user] == 0
        scores = predictions.loc[user][uninteracted]
        top_items = scores.sort_values(ascending=False).head(n).index.tolist()
        recommendations[user] = top_items
    return recommendations


def precision_recall_at_k(recommendations: dict, held_out: dict, k: int) -> tuple:
    """Compute precision@k and recall@k against the held-out positives.

    Args:
        recommendations (dict): user_id -> ranked list of recommended items (already length k).
        held_out (dict): user_id -> list of held-out true-positive items.
        k (int): Number of recommendations evaluated per user.

    Returns:
        (float, float): (mean precision@k, mean recall@k) across users with at least one held-out item.
    """
    precisions, recalls = [], []
    for user, hidden in held_out.items():
        if not hidden:
            continue
        recs = set(recommendations.get(user, []))
        hits = len(recs & set(hidden))
        precisions.append(hits / k)
        recalls.append(hits / len(hidden))
    return float(np.mean(precisions)), float(np.mean(recalls))


def main():
    interactions = generate_interactions(num_users=200, num_items=15)
    train, held_out = holdout_split(interactions, holdout_frac=0.2)

    similarity = compute_user_similarity(train)
    predictions = predict_scores(train, similarity)

    k = 3
    recommendations = recommend_top_n(predictions, train, n=k)

    sample_user = list(recommendations.keys())[0]
    print(f"Top-{k} recommendations for user {sample_user}: {recommendations[sample_user]}")
    print(f"Items actually held out for user {sample_user}: {held_out[sample_user]}")

    precision, recall = precision_recall_at_k(recommendations, held_out, k=k)
    n_random_positive = interactions.values.mean()  # baseline: item's overall popularity rate
    print(f"\nEvaluated on {len(interactions)} users, {sum(len(v) for v in held_out.values())} held-out interactions")
    print(f"Precision@{k}: {precision:.3f}  Recall@{k}: {recall:.3f}")
    print(f"Random-guess baseline precision (overall item density): {n_random_positive:.3f}")
    print(f"Lift over random baseline: {precision / n_random_positive:.2f}x" if n_random_positive > 0 else "")


if __name__ == '__main__':
    main()
