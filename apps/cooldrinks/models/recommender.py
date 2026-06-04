"""Hybrid Recommender Engine for CoolDrinks.

Combines SASRec sequential signals, multi-modal fusion content+context,
and LinUCB exploration-exploitation into unified recommendation engine.

Architecture:
- SASRec: Sequential preference modeling from user history
- Fusion: Multi-modal context-aware scoring
- LinUCB: Exploration-exploitation balancing
- Cold-start: Hybrid content + popularity fallback
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_layer import (
    load_drink_data,
    load_interaction_data,
    get_drink_by_id,
    get_drink_stats,
    get_context_stats
)
from models.sasrec import SASRecModel, build_sessions_from_interactions, create_user_history_mapping
from models.fusion import MultiModalFusionModel
from models.linucb import LinUCBRecommender


class HybridRecommenderEngine:
    """Unified hybrid recommender combining SASRec + Fusion + LinUCB."""

    def __init__(
        self,
        drink_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        d_model: int = 64,
        alpha: float = 1.0,
        sasrec_weights: float = 0.3,
        fusion_weights: float = 0.5,
        linucb_weights: float = 0.2,
        seed: int = 42
    ):
        """Initialize hybrid recommender engine.

        Args:
            drink_df: Drink catalog DataFrame
            interactions_df: Interaction logs DataFrame
            d_model: Embedding dimension
            alpha: LinUCB exploration parameter
            sasrec_weights: Weight for SASRec component
            fusion_weights: Weight for Fusion component
            linucb_weights: Weight for LinUCB component
            seed: Random seed
        """
        self.drink_df = drink_df
        self.interactions_df = interactions_df
        self.d_model = d_model
        self.alpha = alpha
        self.seed = seed

        # Component weights
        self.sasrec_weights = sasrec_weights
        self.fusion_weights = fusion_weights
        self.linucb_weights = linucb_weights

        # Build item mappings
        self.drinks = drink_df["drink_id"].tolist()
        self.drink_to_idx = {drink: idx for idx, drink in enumerate(self.drinks)}
        self.idx_to_drink = {idx: drink for idx, drink in enumerate(self.drinks)}
        self.n_drinks = len(self.drinks)

        # Initialize models
        self.sasrec = SASRecModel(
            n_items=self.n_drinks,
            d_model=d_model,
            seed=seed
        )

        self.fusion = MultiModalFusionModel(
            n_drinks=self.n_drinks,
            d_model=d_model,
            seed=seed + 1
        )

        self.linucb = LinUCBRecommender(
            drink_ids=self.drinks,
            d_features=17,  # weather(5) + time(3) + occasion(6) + prefs(3)
            alpha=alpha,
            seed=seed + 2
        )

        # Build training data
        self.sessions, self.targets, self.item_to_idx = self._build_training_data()

        # Build user history
        self.user_history = create_user_history_mapping(
            interactions_df,
            self.item_to_idx
        )

        # Train models
        print("Training SASRec model...")
        self.sasrec.fit(
            self.sessions,
            self.targets,
            n_epochs=5,
            batch_size=32
        )

        print("Training Fusion model...")
        self.fusion.fit(
            drink_df,
            interactions_df,
            n_epochs=5,
            batch_size=32
        )

        print("Training LinUCB model...")
        self.linucb.train_from_interactions(
            interactions_df,
            d_features=16,
            n_epochs=5
        )

    def _build_training_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Build training data from interactions.

        Returns:
            Tuple of (sessions, targets, item_to_idx)
        """
        # Create item to index mapping
        unique_drinks = sorted(self.interactions_df["drink_id"].unique())
        item_to_idx = {drink: idx for idx, drink in enumerate(unique_drinks)}

        # Build sessions with consistent length (pad to max length)
        sessions = []
        targets = []
        max_session_len = 10

        for user_id, user_interactions in self.interactions_df.groupby("user_id"):
            user_drinks = user_interactions["drink_id"].unique().tolist()

            for i in range(2, len(user_drinks)):
                # Use fixed session length
                seq_len = min(5, i)
                session = user_drinks[i - seq_len:i]
                target = user_drinks[i]

                session_indices = [item_to_idx.get(d, 0) for d in session]

                # Pad session to max_session_len
                padded_session = session_indices + [0] * (max_session_len - len(session_indices))
                sessions.append(padded_session)
                targets.append(item_to_idx.get(target, 0))

        return np.array(sessions, dtype=np.int32), np.array(targets, dtype=np.int32), item_to_idx

    def encode_context(
        self,
        weather: str,
        time_period: str,
        occasion: str,
        bitterness_pref: float = 0.5,
        sweetness_pref: float = 0.5,
        strength_pref: float = 0.5
    ) -> Dict[str, Any]:
        """Encode context into unified representation.

        Args:
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type
            bitterness_pref: User bitterness preference [0, 1]
            sweetness_pref: User sweetness preference [0, 1]
            strength_pref: User strength preference [0, 1]

        Returns:
            Context representation dictionary
        """
        return {
            "weather": weather,
            "time_period": time_period,
            "occasion": occasion,
            "bitterness_pref": bitterness_pref,
            "sweetness_pref": sweetness_pref,
            "strength_pref": strength_pref,
        }

    def _sasrec_scores(self, user_id: str) -> Dict[str, float]:
        """Get SASRec-based scores from user history.

        Args:
            user_id: User identifier

        Returns:
            Dict mapping drink_id to score
        """
        history = self.user_history.get(user_id, [])

        if len(history) < 2:
            return {drink: 0.0 for drink in self.drinks}

        # Get last few items as session
        session = history[-5:]
        scores = self.sasrec.predict_next(np.array(session))

        return {
            self.idx_to_drink[idx]: float(score)
            for idx, score in enumerate(scores)
        }

    def _fusion_scores(
        self,
        weather: str,
        time_period: str,
        occasion: str
    ) -> Dict[str, float]:
        """Get Fusion-based scores for context.

        Args:
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type

        Returns:
            Dict mapping drink_id to score
        """
        recommendations = self.fusion.get_recommendations(
            self.drink_df,
            weather,
            time_period,
            occasion,
            top_k=self.n_drinks
        )

        return {drink_id: score for drink_id, score in recommendations}

    def _linucb_scores(
        self,
        user_id: str,
        weather: str,
        time_period: str,
        occasion: str,
        bitterness_pref: float,
        sweetness_pref: float,
        strength_pref: float
    ) -> Dict[str, float]:
        """Get LinUCB-based scores.

        Args:
            user_id: User identifier
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type
            bitterness_pref: User bitterness preference
            sweetness_pref: User sweetness preference
            strength_pref: User strength preference

        Returns:
            Dict mapping drink_id to score
        """
        recommendations = self.linucb.get_recommendations_with_exploration(
            user_id,
            weather,
            time_period,
            occasion,
            bitterness_pref,
            sweetness_pref,
            strength_pref,
            top_k=self.n_drinks
        )

        return {
            rec["drink_id"]: rec["expected_reward"]
            for rec in recommendations
        }

    def _content_popularity_scores(self) -> Dict[str, float]:
        """Get content-based popularity fallback scores.

        For cold-start drinks with no interaction history.

        Returns:
            Dict mapping drink_id to popularity score
        """
        # Calculate popularity from interactions
        interaction_counts = self.interactions_df["drink_id"].value_counts()

        # Normalize to [0, 1]
        max_count = interaction_counts.max()
        if max_count > 0:
            popularity = interaction_counts / max_count
        else:
            popularity = pd.Series(dtype=float)

        return {
            drink: float(popularity.get(drink, 0.0))
            for drink in self.drinks
        }

    def _cold_start_scores(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get cold-start scores for new drinks.

        Hybrid content + popularity fallback.

        Args:
            context: Context representation

        Returns:
            Dict mapping drink_id to score
        """
        # Content similarity based on context matching
        weather = context["weather"]
        time_period = context["time_period"]
        occasion = context["occasion"]

        content_scores = {}

        for _, drink in self.drink_df.iterrows():
            score = 0.0

            # Weather matching
            seasonality = drink["seasonality"]
            if seasonality == "any":
                score += 0.3
            elif seasonality == "summer" and weather in ["sunny", "hot"]:
                score += 0.4
            elif seasonality == "winter" and weather in ["snowy", "cold"]:
                score += 0.4

            # Time-based matching
            hour = 12 if time_period == "afternoon" else (9 if time_period == "morning" else 20)
            if time_period == "morning" and drink["type"] in ["coffee", "tea"]:
                score += 0.3
            elif time_period == "evening" and drink["type"] in ["beer", "wine", "cocktail"]:
                score += 0.3

            # Occasion matching
            if occasion == "celebration" and drink["type"] in ["wine", "cocktail"]:
                score += 0.2
            elif occasion == "recovery" and drink["type"] in ["non-alcoholic", "tea"]:
                score += 0.2

            content_scores[drink["drink_id"]] = score

        # Combine with popularity
        popularity_scores = self._content_popularity_scores()

        hybrid_scores = {
            drink: 0.6 * content_scores[drink] + 0.4 * popularity_scores[drink]
            for drink in self.drinks
        }

        return hybrid_scores

    def recommend(
        self,
        user_id: str,
        weather: str,
        time_period: str,
        occasion: str,
        bitterness_pref: float = 0.5,
        sweetness_pref: float = 0.5,
        strength_pref: float = 0.5,
        top_k: int = 10,
        excluded_items: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get top-k recommendations for user and context.

        Hybrid scoring:
        Score = w_sasrec * sasrec + w_fusion * fusion + w_linucb * linucb

        Args:
            user_id: User identifier
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type
            bitterness_pref: User bitterness preference [0, 1]
            sweetness_pref: User sweetness preference [0, 1]
            strength_pref: User strength preference [0, 1]
            top_k: Number of recommendations
            excluded_items: Items to exclude

        Returns:
            List of recommendation dictionaries
        """
        # Encode context
        context = self.encode_context(
            weather, time_period, occasion,
            bitterness_pref, sweetness_pref, strength_pref
        )

        # Get scores from each component
        sasrec_scores = self._sasrec_scores(user_id)
        fusion_scores = self._fusion_scores(weather, time_period, occasion)
        linucb_scores = self._linucb_scores(
            user_id, weather, time_period, occasion,
            bitterness_pref, sweetness_pref, strength_pref
        )

        # Combine scores
        final_scores = {}

        for drink in self.drinks:
            if excluded_items and drink in excluded_items:
                continue

            score = (
                self.sasrec_weights * sasrec_scores.get(drink, 0.0) +
                self.fusion_weights * fusion_scores.get(drink, 0.0) +
                self.linucb_weights * linucb_scores.get(drink, 0.0)
            )

            final_scores[drink] = score

        # Sort by final score
        sorted_drinks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result
        recommendations = []
        for drink_id, score in sorted_drinks[:top_k]:
            drink = get_drink_by_id(drink_id, self.drink_df)

            # Get uncertainty from LinUCB
            linucb_rec = next(
                (r for r in linucb_scores.items() if r[0] == drink_id),
                None
            )

            recommendations.append({
                "drink_id": drink_id,
                "name": drink["name"],
                "type": drink["type"],
                "style": drink["style"],
                "abv": drink["abv"],
                "bitterness": drink["bitterness"],
                "sweetness": drink["sweetness"],
                "carbonation": drink["carbonation"],
                "seasonality": drink["seasonality"],
                "overall_score": round(score, 4),
                "sasrec_score": round(sasrec_scores.get(drink_id, 0.0), 4),
                "fusion_score": round(fusion_scores.get(drink_id, 0.0), 4),
                "linucb_score": round(linucb_scores.get(drink_id, 0.0), 4),
            })

        return recommendations

    def explain_recommendation(
        self,
        drink_id: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate explanation for recommendation.

        Args:
            drink_id: Drink identifier
            context: Context representation

        Returns:
            Explanation string
        """
        drink = get_drink_by_id(drink_id, self.drink_df)

        explanations = []

        # Weather-based explanation
        weather = context["weather"]
        if drink["seasonality"] == "any":
            explanations.append("suitable for any weather")
        elif drink["seasonality"] == "summer" and weather in ["sunny", "rainy"]:
            explanations.append("refreshing for summer weather")
        elif drink["seasonality"] == "winter" and weather in ["cloudy", "snowy"]:
            explanations.append("cozy for winter days")

        # Time-based explanation
        time_period = context["time_period"]
        if time_period == "morning" and drink["type"] == "coffee":
            explanations.append("perfect morning pick-me-up")
        elif time_period == "afternoon" and drink["type"] == "tea":
            explanations.append("great afternoon refresher")
        elif time_period == "evening" and drink["type"] == "beer":
            explanations.append("ideal evening relaxation")

        # Occasion-based explanation
        occasion = context["occasion"]
        if occasion == "casual":
            explanations.append("casual and accessible")
        elif occasion == "celebration" and drink["type"] == "wine":
            explanations.append("elegant for celebrations")
        elif occasion == "pairing" and drink["type"] == "beer":
            explanations.append("versatile for food pairing")

        # Taste-based explanation
        bitterness_pref = context["bitterness_pref"]
        sweetness_pref = context["sweetness_pref"]

        if bitterness_pref > 0.6 and drink["bitterness"] > 50:
            explanations.append("bold bitter profile")
        elif sweetness_pref > 0.6 and drink["sweetness"] > 50:
            explanations.append("naturally sweet finish")

        return "; ".join(explanations) if explanations else "recommended based on your preferences"

    def get_drink_details(self, drink_id: str) -> Optional[Dict[str, Any]]:
        """Get full details for a drink.

        Args:
            drink_id: Drink identifier

        Returns:
            Drink details dictionary or None
        """
        return get_drink_by_id(drink_id, self.drink_df)

    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get statistics about the drink catalog.

        Returns:
            Catalog statistics
        """
        return get_drink_stats(self.drink_df)

    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get statistics about interaction data.

        Returns:
            Interaction statistics
        """
        return get_context_stats(self.interactions_df)
