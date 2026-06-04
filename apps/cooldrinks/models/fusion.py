"""Multi-modal fusion model for context-aware recommendation.

Combines drink content embeddings with context embeddings (weather, time, occasion)
using cross-attention to model context-drink interactions.

Architecture:
- Drink embedding: content_features -> dense(64) -> tanh
- Context embedding: one_hot(weather, time, occasion) -> dense(64)
- Cross-attention: models interaction between drink and context
- Output: ranking scores for all drinks
- Loss: Pairwise ranking loss (BPR)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class MultiModalFusionModel:
    """Multi-modal fusion model combining drink content and context."""

    def __init__(
        self,
        n_drinks: int,
        d_model: int = 64,
        d_context: int = 32,
        dropout_rate: float = 0.1,
        lr: float = 0.001,
        seed: int = 42
    ):
        """Initialize multi-modal fusion model.

        Args:
            n_drinks: Number of drinks in catalog
            d_model: Embedding dimension for drinks
            d_context: Context embedding dimension
            dropout_rate: Dropout rate
            lr: Learning rate
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        self.n_drinks = n_drinks
        self.d_model = d_model
        self.d_context = d_context
        self.dropout_rate = dropout_rate
        self.lr = lr

        # Context features: [weather, time_period, occasion] (one-hot encoded)
        self.weather_vocab = ["sunny", "rainy", "cloudy", "snowy", "stormy"]
        self.time_vocab = ["morning", "afternoon", "evening"]
        self.occasion_vocab = ["casual", "celebration", "pairing", "recovery", "social", "business"]

        self.n_context_features = (
            len(self.weather_vocab) + len(self.time_vocab) + len(self.occasion_vocab)
        )

        # Context encoder
        self.context_encoder = self._init_weights(self.n_context_features, d_context, seed + 1)
        self.context_bias = np.zeros(d_context)

        # Cross-attention weights
        self.q_w = self._init_weights(d_model, d_model, seed + 2)  # Query from drink
        self.k_w = self._init_weights(d_context, d_model, seed + 3)  # Key from context
        self.v_w = self._init_weights(d_context, d_model, seed + 4)  # Value from context

       # Output projection
        self.output_w = self._init_weights(d_model, 1, seed + 5)
        self.output_bias = np.zeros(1)

        # Content encoder (initialized in fit() with correct feature dimension)
        self.content_encoder = None
        self.content_bias = None

        # Training state
        self.training = False

         # Store styles for consistent encoding (set when fit is called)
        self.styles = None
        self.n_types = 6  # beer, wine, coffee, tea, cocktail, non-alcoholic

    def _init_weights(self, in_dim: int, out_dim: int, seed: int) -> np.ndarray:
        """Initialize weights with Xavier initialization."""
        np.random.seed(seed)
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim) * scale

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation."""
        return np.tanh(x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def encode_drink_content(self, drink_features: np.ndarray) -> np.ndarray:
        """Encode drink content features into embedding.

        Args:
            drink_features: Array of [abv, bitterness, sweetness, carbonation, strength]
                           Plus one-hot encoded type and style

        Returns:
            Drink embedding vector
        """
        return self._tanh(np.matmul(drink_features, self.content_encoder) + self.content_bias)

    def encode_context(self, context_features: np.ndarray) -> np.ndarray:
        """Encode context features into embedding.

        Args:
            context_features: One-hot encoded [weather, time_period, occasion]

        Returns:
            Context embedding vector
        """
        return self._tanh(np.matmul(context_features, self.context_encoder) + self.context_bias)

    def _one_hot_encode(self, value: str, vocabulary: List[str]) -> np.ndarray:
        """One-hot encode a categorical value."""
        vector = np.zeros(len(vocabulary))
        if value in vocabulary:
            vector[vocabulary.index(value)] = 1
        return vector

    def encode_context_full(self,
                           weather: str,
                           time_period: str,
                           occasion: str) -> np.ndarray:
        """Encode full context into embedding.

        Args:
            weather: Weather condition
            time_period: Time period (morning/afternoon/evening)
            occasion: Occasion type

        Returns:
            Context embedding vector
        """
        weather_vec = self._one_hot_encode(weather, self.weather_vocab)
        time_vec = self._one_hot_encode(time_period, self.time_vocab)
        occasion_vec = self._one_hot_encode(occasion, self.occasion_vocab)

        context_vec = np.concatenate([weather_vec, time_vec, occasion_vec])
        return self.encode_context(context_vec)

    def cross_attention(self,
                       drink_emb: np.ndarray,
                       context_emb: np.ndarray) -> np.ndarray:
        """Apply cross-attention between drink and context.

        Args:
            drink_emb: Drink embedding [d_model]
            context_emb: Context embedding [d_context]

        Returns:
            Attended drink representation
        """
        # Query from drink, Key/Value from context
        q = np.matmul(drink_emb.reshape(1, -1), self.q_w)  # [1, d_model]
        k = np.matmul(context_emb.reshape(1, -1), self.k_w)  # [1, d_model]
        v = np.matmul(context_emb.reshape(1, -1), self.v_w)  # [1, d_model]

        # Compute attention scores
        scores = np.matmul(q, k.T) / np.sqrt(self.d_model)  # [1, 1]
        attention_weights = self._softmax(scores)  # [1, 1]

        # Weighted sum of values
        attended = np.matmul(attention_weights, v)  # [1, d_model]

        return attended[0]

    def compute_score(self,
                     drink_emb: np.ndarray,
                     context_emb: np.ndarray) -> float:
        """Compute drink-context compatibility score.

        Args:
            drink_emb: Drink embedding
            context_emb: Context embedding

        Returns:
            Compatibility score
        """
        attended = self.cross_attention(drink_emb, context_emb)
        score = np.matmul(attended.reshape(1, -1), self.output_w) + self.output_bias
        return float(self._sigmoid(score)[0, 0])

    def get_drink_features(self, drink_df: pd.DataFrame, drink_id: str) -> np.ndarray:
        """Get feature vector for a drink.

        Args:
            drink_df: Drink catalog DataFrame
            drink_id: Drink identifier

        Returns:
            Feature vector [type_onehot, style_onehot, abv, bitterness, sweetness, carbonation, strength]
        """
        drink = drink_df[drink_df["drink_id"] == drink_id].iloc[0]

        # One-hot encode type
        type_vec = self._one_hot_encode(drink["type"], ["beer", "wine", "coffee", "tea", "cocktail", "non-alcoholic"])

        # One-hot encode style using stored styles
        style_vec = self._one_hot_encode(drink["style"], self.styles)

        # Normalize numeric features
        abv = drink["abv"] / 40  # Normalize to [0, 1]
        bitterness = drink["bitterness"] / 100
        sweetness = drink["sweetness"] / 100
        carbonation = drink["carbonation"] / 5
        strength = drink["strength"]

        return np.concatenate([type_vec, style_vec, [abv, bitterness, sweetness, carbonation, strength]])

    def predict_all(
        self,
        drink_df: pd.DataFrame,
        weather: str,
        time_period: str,
        occasion: str
    ) -> List[Tuple[str, float]]:
        """Predict scores for all drinks given context.

        Args:
            drink_df: Drink catalog DataFrame
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type

        Returns:
            List of (drink_id, score) tuples sorted by score
        """
        context_emb = self.encode_context_full(weather, time_period, occasion)

        scores = []
        for _, drink in drink_df.iterrows():
            drink_features = self.get_drink_features(drink_df, drink["drink_id"])
            drink_emb = self.encode_drink_content(drink_features)
            score = self.compute_score(drink_emb, context_emb)
            scores.append((drink["drink_id"], score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_recommendations(
        self,
        drink_df: pd.DataFrame,
        weather: str,
        time_period: str,
        occasion: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top-k recommendations for context.

        Args:
            drink_df: Drink catalog DataFrame
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type
            top_k: Number of recommendations

        Returns:
            List of (drink_id, score) tuples
        """
        all_scores = self.predict_all(drink_df, weather, time_period, occasion)
        return all_scores[:top_k]

    def compute_loss_pairwise(
        self,
        drink_df: pd.DataFrame,
        pos_drink_id: str,
        neg_drink_id: str,
        weather: str,
        time_period: str,
        occasion: str
    ) -> float:
        """Compute BPR pairwise ranking loss.

        Args:
            drink_df: Drink catalog DataFrame
            pos_drink_id: Positive (liked) drink
            neg_drink_id: Negative (disliked) drink
            weather: Weather condition
            time_period: Time period
            occasion: Occasion type

        Returns:
            BPR loss value
        """
        context_emb = self.encode_context_full(weather, time_period, occasion)

        pos_features = self.get_drink_features(drink_df, pos_drink_id)
        neg_features = self.get_drink_features(drink_df, neg_drink_id)

        pos_emb = self.encode_drink_content(pos_features)
        neg_emb = self.encode_drink_content(neg_features)

        pos_score = self.compute_score(pos_emb, context_emb)
        neg_score = self.compute_score(neg_emb, context_emb)

        # BPR loss: -log(sigmoid(pos - neg))
        diff = pos_score - neg_score
        loss = -np.log(self._sigmoid(diff) + 1e-10)

        return float(loss)

    def train_step(
        self,
        drink_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        batch_size: int = 32
    ) -> float:
        """Perform one training step on a batch of interactions.

        Args:
            drink_df: Drink catalog DataFrame
            interactions_df: Interaction logs
            batch_size: Batch size

        Returns:
            Average loss
        """
        # Sample positive-negative pairs
        sample_interactions = interactions_df.sample(n=min(batch_size, len(interactions_df)))

        total_loss = 0
        for _, row in sample_interactions.iterrows():
            # Get positive drink
            pos_drink = row["drink_id"]

            # Sample negative drink (different drink)
            neg_drink_options = drink_df[drink_df["drink_id"] != pos_drink]["drink_id"].tolist()
            neg_drink = np.random.choice(neg_drink_options) if neg_drink_options else drink_df.iloc[0]["drink_id"]

            # Compute loss
            loss = self.compute_loss_pairwise(
                drink_df, pos_drink, neg_drink,
                row["weather"], row["time_period"], row["occasion"]
            )
            total_loss += loss

        avg_loss = total_loss / len(sample_interactions)

        # Simple gradient update (simplified for demo)
        # In practice, use backpropagation
        lr = self.lr
        if avg_loss > 0:
            # Update content encoder weights based on loss gradient
            grad_scale = lr * avg_loss * 0.01
            self.content_encoder -= grad_scale * np.random.randn(*self.content_encoder.shape)

        return avg_loss

    def fit(
        self,
        drink_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        n_epochs: int = 10,
        batch_size: int = 32,
        val_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            drink_df: Drink catalog DataFrame
            interactions_df: Interaction logs
            n_epochs: Number of epochs
            batch_size: Batch size
            val_split: Validation split

        Returns:
            Training history
        """
        # Initialize content encoder with correct feature dimension
        # Feature vector: [type_onehot (6), style_onehot (n_styles), abv, bitterness, sweetness, carbonation, strength]
        self.styles = sorted(drink_df["style"].unique())
        n_styles = len(self.styles)
        self.n_content_features = self.n_types + n_styles + 5  # 5 numeric features

        self.content_encoder = self._init_weights(self.n_content_features, self.d_model, 42)
        self.content_bias = np.zeros(self.d_model)

        train_losses = []
        val_losses = []

        n_samples = len(interactions_df)
        n_train = int(n_samples * (1 - val_split))

        train_interactions = interactions_df.iloc[:n_train]
        val_interactions = interactions_df.iloc[n_train:]

        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_step(drink_df, train_interactions, batch_size)
            train_losses.append(train_loss)

            # Validation
            val_loss = self.train_step(drink_df, val_interactions, batch_size)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        return {"train_losses": train_losses, "val_losses": val_losses}

    def save(self, path: str):
        """Save model weights."""
        np.savez(
            path,
            content_encoder=self.content_encoder,
            content_bias=self.content_bias,
            context_encoder=self.context_encoder,
            context_bias=self.context_bias,
            q_w=self.q_w,
            k_w=self.k_w,
            v_w=self.v_w,
            output_w=self.output_w,
            output_bias=self.output_bias,
            weather_vocab=np.array(self.weather_vocab),
            time_vocab=np.array(self.time_vocab),
            occasion_vocab=np.array(self.occasion_vocab),
        )

    def load(self, path: str):
        """Load model weights."""
        data = np.load(path, allow_pickle=True)
        self.content_encoder = data["content_encoder"]
        self.content_bias = data["content_bias"]
        self.context_encoder = data["context_encoder"]
        self.context_bias = data["context_bias"]
        self.q_w = data["q_w"]
        self.k_w = data["k_w"]
        self.v_w = data["v_w"]
        self.output_w = data["output_w"]
        self.output_bias = data["output_bias"]
        self.weather_vocab = data["weather_vocab"].tolist()
        self.time_vocab = data["time_vocab"].tolist()
        self.occasion_vocab = data["occasion_vocab"].tolist()


# ============================================================================
# CONTEXT ENCODING HELPERS
# ============================================================================


def build_context_features(
    weather: str,
    time_period: str,
    occasion: str
) -> Dict[str, str]:
    """Build context feature dictionary with validation.

    Args:
        weather: Weather condition
        time_period: Time period
        occasion: Occasion type

    Returns:
        Dictionary with validated context features
    """
    valid_weather = ["sunny", "rainy", "cloudy", "snowy", "stormy"]
    valid_time = ["morning", "afternoon", "evening"]
    valid_occasion = ["casual", "celebration", "pairing", "recovery", "social", "business"]

    return {
        "weather": weather if weather in valid_weather else "sunny",
        "time_period": time_period if time_period in valid_time else "afternoon",
        "occasion": occasion if occasion in valid_occasion else "casual"
    }


def context_to_features(
    context: Dict[str, str],
    model: MultiModalFusionModel
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert context dict to encoded features.

    Args:
        context: Context dictionary
        model: Fusion model (for vocabulary)

    Returns:
        Tuple of (drink_features, context_features)
    """
    context_encoded = model.encode_context_full(
        context["weather"],
        context["time_period"],
        context["occasion"]
    )
    return context_encoded
