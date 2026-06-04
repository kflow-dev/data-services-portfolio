"""LinUCB Bandit for exploration-exploitation in recommendations.

Linear Upper Confidence Bound algorithm for balancing exploration vs. exploitation
in contextual bandit settings.

Algorithm:
- For each drink k, maintain linear model u_k = A_k^(-1) b_k
- Confidence interval: c_k = sqrt(x_k^T A_k^(-1) x_k)
- UCB score: s_k = u_k^T x_k + alpha * c_k
- Select drink with highest UCB score

References:
- Li et al. "A Contextual-Bandit Approach to Personalized News Article Recommendation" (2010)
- Abbasi-Yadkori et al. "Improved Algorithms for Linear Contextual Bandits" (2011)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class LinUCBAlgorithm:
    """LinUCB contextual bandit algorithm."""

    def __init__(
        self,
        n_drinks: int,
        d_features: int,
        alpha: float = 1.0,
        reg_param: float = 1.0,
        seed: int = 42
    ):
        """Initialize LinUCB algorithm.

        Args:
            n_drinks: Number of drinks (arms)
            d_features: Feature dimension
            alpha: Exploration parameter (higher = more exploration)
            reg_param: Regularization parameter for A matrix
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        self.n_drinks = n_drinks
        self.d_features = d_features
        self.alpha = alpha
        self.reg_param = reg_param

        # For each drink (arm), maintain:
        # A_k: d x d matrix (information matrix)
        # b_k: d vector (sufficient statistic)
        # u_k: d vector (linear model weights)

        self.A = {k: np.eye(d_features) * reg_param for k in range(n_drinks)}
        self.b = {k: np.zeros(d_features) for k in range(n_drinks)}

        # Track statistics
        self.n_pulls = {k: 0 for k in range(n_drinks)}
        self.total_reward = {k: 0.0 for k in range(n_drinks)}

        # Track feature norms for analysis
        self.feature_norms = []

    def _compute_UCB_score(self, x: np.ndarray, k: int) -> float:
        """Compute UCB score for drink k with feature vector x.

        Args:
            x: Feature vector [d_features]
            k: Drink index

        Returns:
            UCB score
        """
        # Solve for u_k = A_k^(-1) b_k using Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.A[k])
            u_k = np.linalg.solve(L.T, np.linalg.solve(L, self.b[k]))

            # Compute confidence width: sqrt(x^T A_k^(-1) x)
            A_inv_x = np.linalg.solve(self.A[k], x)
            confidence = np.sqrt(np.dot(x, A_inv_x))

            # UCB score
            score = np.dot(u_k, x) + self.alpha * confidence

            return score, confidence

        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            return 0.0, 0.0

    def select_arm(self, x: np.ndarray) -> int:
        """Select arm with highest UCB score.

        Args:
            x: Context feature vector

        Returns:
            Selected drink index
        """
        scores = {}

        for k in range(self.n_drinks):
            score, _ = self._compute_UCB_score(x, k)
            scores[k] = score

        # Select drink with highest score
        selected_k = max(scores, key=scores.get)
        return selected_k

    def update(self, k: int, x: np.ndarray, reward: float):
        """Update LinUCB model after observing reward.

        Args:
            k: Selected drink index
            x: Feature vector at selection time
            reward: Observed reward
        """
        # A_k <- A_k + x x^T
        self.A[k] += np.outer(x, x)

        # b_k <- b_k + r * x
        self.b[k] += reward * x

        # Update statistics
        self.n_pulls[k] += 1
        self.total_reward[k] += reward

        self.feature_norms.append(np.linalg.norm(x))

    def get_uncertainty(self, x: np.ndarray, k: int) -> float:
        """Get uncertainty (confidence width) for drink k.

        Args:
            x: Feature vector
            k: Drink index

        Returns:
            Uncertainty value
        """
        _, confidence = self._compute_UCB_score(x, k)
        return confidence

    def get_model_weights(self, k: int) -> Optional[np.ndarray]:
        """Get learned model weights for drink k.

        Args:
            k: Drink index

        Returns:
            Model weight vector or None if not enough data
        """
        try:
            L = np.linalg.cholesky(self.A[k])
            return np.linalg.solve(L.T, np.linalg.solve(L, self.b[k]))
        except np.linalg.LinAlgError:
            return None

    def get_statistics(self) -> Dict[int, Dict]:
        """Get statistics for all drinks.

        Returns:
            Dictionary with per-drink statistics
        """
        stats = {}

        for k in range(self.n_drinks):
            mean_reward = (
                self.total_reward[k] / self.n_pulls[k]
                if self.n_pulls[k] > 0 else 0.0
            )

            stats[k] = {
                "n_pulls": self.n_pulls[k],
                "total_reward": self.total_reward[k],
                "mean_reward": mean_reward,
                "uncertainty": self.get_uncertainty(np.ones(self.d_features), k)
            }

        return stats


class LinUCBRecommender:
    """LinUCB-based recommender with drink-level bandits."""

    def __init__(
        self,
        drink_ids: List[str],
        d_features: int,
        alpha: float = 1.0,
        reg_param: float = 1.0,
        seed: int = 42
    ):
        """Initialize LinUCB recommender.

        Args:
            drink_ids: List of drink IDs
            d_features: Feature dimension
            alpha: Exploration parameter
            reg_param: Regularization parameter
            seed: Random seed
        """
        self.drink_ids = drink_ids
        self.drink_to_idx = {drink_id: idx for idx, drink_id in enumerate(drink_ids)}
        self.idx_to_drink = {idx: drink_id for idx, drink_id in enumerate(drink_ids)}
        self.n_drinks = len(drink_ids)

        # Per-drink LinUCB models
        self.algorithms = {
            drink_id: LinUCBAlgorithm(
                n_drinks=1,
                d_features=d_features,
                alpha=alpha,
                reg_param=reg_param,
                seed=seed + idx
            )
            for idx, drink_id in enumerate(drink_ids)
        }

        # User-specific models
        self.user_models: Dict[str, Dict[str, LinUCBAlgorithm]] = defaultdict(dict)

    def _encode_context(
        self,
        weather: str,
        time_period: str,
        occasion: str,
        bitterness_pref: float = 0.5,
        sweetness_pref: float = 0.5,
        strength_pref: float = 0.5
    ) -> np.ndarray:
        """Encode context into feature vector.

        Args:
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type
            bitterness_pref: User bitterness preference [0, 1]
            sweetness_pref: User sweetness preference [0, 1]
            strength_pref: User strength preference [0, 1]

        Returns:
            Feature vector
        """
        # One-hot encode weather
        weather_vocab = ["sunny", "rainy", "cloudy", "snowy", "stormy"]
        weather_vec = np.zeros(len(weather_vocab))
        if weather in weather_vocab:
            weather_vec[weather_vocab.index(weather)] = 1

        # One-hot encode time
        time_vocab = ["morning", "afternoon", "evening"]
        time_vec = np.zeros(len(time_vocab))
        if time_period in time_vocab:
            time_vec[time_vocab.index(time_period)] = 1

        # One-hot encode occasion
        occasion_vocab = ["casual", "celebration", "pairing", "recovery", "social", "business"]
        occasion_vec = np.zeros(len(occasion_vocab))
        if occasion in occasion_vocab:
            occasion_vec[occasion_vocab.index(occasion)] = 1

        # Concatenate features
        x = np.concatenate([
            weather_vec,
            time_vec,
            occasion_vec,
            [bitterness_pref, sweetness_pref, strength_pref]
        ])

        return x

    def _get_user_model(self, user_id: str) -> Dict[str, LinUCBAlgorithm]:
        """Get or create user-specific bandit models.

        Args:
            user_id: User identifier

        Returns:
            Dict mapping drink_id to LinUCB algorithm
        """
        if user_id not in self.user_models:
            # Initialize with pre-trained global models
            for drink_id in self.drink_ids:
                self.user_models[user_id][drink_id] = self.algorithms[drink_id]

        return self.user_models[user_id]

    def select_drinks(
        self,
        user_id: str,
        weather: str,
        time_period: str,
        occasion: str,
        bitterness_pref: float = 0.5,
        sweetness_pref: float = 0.5,
        strength_pref: float = 0.5,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Select top-k drinks using LinUCB.

        Args:
            user_id: User identifier
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type
            bitterness_pref: User bitterness preference
            sweetness_pref: User sweetness preference
            strength_pref: User strength preference
            top_k: Number of drinks to return

        Returns:
            List of (drink_id, score) tuples
        """
        x = self._encode_context(
            weather, time_period, occasion,
            bitterness_pref, sweetness_pref, strength_pref
        )

        user_model = self._get_user_model(user_id)

        scores = {}

        for drink_id in self.drink_ids:
            algorithm = user_model.get(drink_id)
            score, _ = algorithm._compute_UCB_score(x, 0)
            scores[drink_id] = score

        # Sort by score and return top-k
        sorted_drinks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_drinks[:top_k]

    def observe(
        self,
        user_id: str,
        drink_id: str,
        x: np.ndarray,
        reward: float
    ):
        """Observe reward for selected drink.

        Args:
            user_id: User identifier
            drink_id: Selected drink
            x: Feature vector at selection time
            reward: Observed reward
        """
        user_model = self._get_user_model(user_id)

        if drink_id in user_model:
            user_model[drink_id].update(0, x, reward)

    def train_from_interactions(
        self,
        interactions_df: pd.DataFrame,
        d_features: int,
        n_epochs: int = 5
    ):
        """Train LinUCB models from interaction logs.

        Args:
            interactions_df: Interaction logs
            d_features: Feature dimension
            n_epochs: Number of training epochs
        """
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_samples = 0

            for _, row in interactions_df.iterrows():
                x = self._encode_context(
                    row["weather"],
                    row["time_period"],
                    row["occasion"],
                    row.get("bitterness_pref", 0.5),
                    row.get("sweetness_pref", 0.5),
                    row.get("strength_pref", 0.5)
                )

                drink_id = row["drink_id"]

                if drink_id in self.algorithms:
                    # Get predicted reward
                    algorithm = self.algorithms[drink_id]
                    pred_reward, _ = algorithm._compute_UCB_score(x, 0)

                    # Actual reward
                    actual_reward = row.get("value", 0)

                    # Update model
                    algorithm.update(0, x, actual_reward)

                    # Track loss
                    total_loss += (pred_reward - actual_reward) ** 2
                    n_samples += 1

            avg_loss = total_loss / n_samples if n_samples > 0 else 0
            print(f"Epoch {epoch+1}/{n_epochs}: LinUCB train_loss={avg_loss:.4f}")

    def get_recommendations_with_exploration(
        self,
        user_id: str,
        weather: str,
        time_period: str,
        occasion: str,
        bitterness_pref: float = 0.5,
        sweetness_pref: float = 0.5,
        strength_pref: float = 0.5,
        top_k: int = 10
    ) -> List[Dict]:
        """Get recommendations with exploration bonus.

        Args:
            user_id: User identifier
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type
            bitterness_pref: User bitterness preference
            sweetness_pref: User sweetness preference
            strength_pref: User strength preference
            top_k: Number of recommendations

        Returns:
            List of recommendation dictionaries with scores and uncertainty
        """
        x = self._encode_context(
            weather, time_period, occasion,
            bitterness_pref, sweetness_pref, strength_pref
        )

        user_model = self._get_user_model(user_id)

        recommendations = []

        for drink_id in self.drink_ids:
            algorithm = user_model.get(drink_id)

            score, uncertainty = algorithm._compute_UCB_score(x, 0)

            recommendations.append({
                "drink_id": drink_id,
                "expected_reward": float(score),
                "uncertainty": float(uncertainty),
                "n_interactions": algorithm.n_pulls[0],
                "mean_reward": (
                    algorithm.total_reward[0] / algorithm.n_pulls[0]
                    if algorithm.n_pulls[0] > 0 else 0.0
                )
            })

        # Sort by expected reward
        recommendations.sort(key=lambda x: x["expected_reward"], reverse=True)

        return recommendations[:top_k]

    def cold_start_scores(
        self,
        x: np.ndarray,
        n_samples: int = 100
    ) -> List[Tuple[int, float]]:
        """Get cold-start scores based on feature similarity.

        For drinks with no interaction history, use feature-based scoring.

        Args:
            x: Feature vector
            n_samples: Number of Monte Carlo samples

        Returns:
            List of (drink_idx, score) tuples
        """
        scores = {}

        for k in range(self.n_drinks):
            algorithm = self.algorithms[self.drink_ids[k]]

            if algorithm.n_pulls[0] == 0:
                # Cold start: use feature norm as proxy for preference match
                feature_norm = np.linalg.norm(x)
                # Normalize by expected feature norm
                score = 1.0 / (1.0 + feature_norm)
            else:
                # Use learned model
                score, _ = algorithm._compute_UCB_score(x, 0)

            scores[k] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
