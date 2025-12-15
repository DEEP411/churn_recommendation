"""User‑based collaborative filtering recommendation engine.

This script generates a synthetic user‑service interaction matrix,
computes cosine similarities between users, predicts preferences for
services the user has not interacted with, and recommends items with
the highest predicted scores.  A simple engagement metric is
calculated to compare recommended scores against the baseline.

Usage:
    python recommendation.py

Dependencies:
    numpy, pandas, scikit‑learn
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def generate_interactions(num_users: int = 30, num_items: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic user‑item interaction matrix.

    Each user has binary interactions with items (1 = interacted, 0 = not),
    with sparsity controlled by a random seed.

    Args:
        num_users (int): Number of users.
        num_items (int): Number of items/services.
        seed (int): Random seed for reproducibility.

    Returns:
        interactions (pd.DataFrame): DataFrame with shape (num_users, num_items).
    """
    rng = np.random.default_rng(seed)
    # Generate a binary interaction matrix with ~30% density
    interactions = rng.random((num_users, num_items)) < 0.3
    df = pd.DataFrame(interactions.astype(int), columns=[f'service_{i}' for i in range(num_items)])
    df.index.name = 'user_id'
    return df


def compute_user_similarity(interactions: pd.DataFrame) -> np.ndarray:
    """Compute cosine similarity between users based on their item interactions.

    Args:
        interactions (pd.DataFrame): User‑item interaction matrix.

    Returns:
        similarity (np.ndarray): User‑user similarity matrix.
    """
    return cosine_similarity(interactions)


def predict_scores(interactions: pd.DataFrame, similarity: np.ndarray) -> pd.DataFrame:
    """Predict user‑item scores for items not yet interacted with.

    The predicted score for user u on item i is a weighted average of
    other users' interactions with item i, weighted by the similarity to u.

    Args:
        interactions (pd.DataFrame): User‑item interaction matrix (binary).
        similarity (np.ndarray): User‑user similarity matrix.

    Returns:
        pd.DataFrame: Predicted scores with the same shape as interactions.
    """
    # Ensure similarity diagonal zeros (exclude self influence)
    np.fill_diagonal(similarity, 0)
    preds = similarity.dot(interactions.values) / np.maximum(similarity.sum(axis=1)[:, None], 1e-9)
    predicted_df = pd.DataFrame(preds, index=interactions.index, columns=interactions.columns)
    return predicted_df


def recommend_top_n(predictions: pd.DataFrame, interactions: pd.DataFrame, n: int = 3) -> dict:
    """Recommend top‑N items for each user based on predicted scores.

    Args:
        predictions (pd.DataFrame): Predicted scores per user and item.
        interactions (pd.DataFrame): Original binary interaction matrix.
        n (int): Number of recommendations per user.

    Returns:
        dict: Mapping user index to list of recommended item names.
    """
    recommendations = {}
    for user in predictions.index:
        # Items not already interacted with
        uninteracted = interactions.loc[user] == 0
        scores = predictions.loc[user][uninteracted]
        top_items = scores.sort_values(ascending=False).head(n).index.tolist()
        recommendations[user] = top_items
    return recommendations


def compute_engagement_gain(predictions: pd.DataFrame, interactions: pd.DataFrame) -> float:
    """Compute a simple engagement gain metric.

    This function compares the average predicted top‑item score to the
    average historical engagement level (number of items interacted per user).

    For each user, we find the maximum predicted score among items they
    haven't interacted with.  The mean of these maximum scores serves
    as a proxy for anticipated engagement.  We then compute the
    percentage difference from the baseline, which is the average
    fraction of items each user has historically interacted with.

    Args:
        predictions (pd.DataFrame): Predicted scores for each user and item.
        interactions (pd.DataFrame): Binary user‑item interaction matrix.

    Returns:
        float: Estimated percentage gain in engagement from recommendations.
    """
    # Compute baseline engagement: average fraction of interacted items per user
    interaction_counts = interactions.sum(axis=1)
    baseline_engagement = (interaction_counts / interactions.shape[1]).mean()
    # Compute predicted engagement: for each user, take max score among items not interacted
    top_scores = []
    for user in interactions.index:
        interacted_mask = interactions.loc[user] == 1
        # Filter predictions for uninteracted items
        user_preds = predictions.loc[user][~interacted_mask]
        if len(user_preds) > 0:
            top_scores.append(user_preds.max())
        else:
            top_scores.append(0.0)
    predicted_engagement = np.mean(top_scores)
    # Return percentage gain compared to baseline
    if baseline_engagement == 0:
        return 0.0
    return ((predicted_engagement - baseline_engagement) / baseline_engagement) * 100


def main():
    # Generate synthetic interactions
    interactions = generate_interactions(num_users=50, num_items=10)
    print("Synthetic interaction matrix:")
    print(interactions.head())

    # Compute user similarities
    similarity = compute_user_similarity(interactions)

    # Predict scores for uninteracted items
    predictions = predict_scores(interactions, similarity)

    # Recommend top 3 items per user
    recommendations = recommend_top_n(predictions, interactions, n=3)
    sample_user = list(recommendations.keys())[0]
    print(f"\nTop recommendations for user {sample_user}: {recommendations[sample_user]}")

    # Compute engagement gain
    engagement_gain = compute_engagement_gain(predictions, interactions)
    print(f"\nEstimated engagement gain from recommendations: {engagement_gain:.2f}%")


if __name__ == '__main__':
    main()