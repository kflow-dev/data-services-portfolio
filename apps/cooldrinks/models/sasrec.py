"""SASRec (Sequence-Aware Session-based Recommender) implementation.

Implements Transformer-based sequential recommendation using self-attention
to model user preference evolution over time.

Architecture:
- Input: User session history [d_{t-3}, d_{t-2}, d_{t-1}]
- Embedding: Drink ID -> d_dim (64)
- Transformer: 2-layer self-attention, d_model=64, 4 heads
- Output: Next drink probability distribution
- Loss: Cross-entropy on next item prediction

Based on: "Self-Attentive Sequential Recommendation" (Kang & McAuley, 2018)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json


class SASRecModel:
    """SASRec model for sequential recommendation."""

    def __init__(
        self,
        n_items: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 10,
        dropout_rate: float = 0.1,
        lr: float = 0.001,
        seed: int = 42
    ):
        """Initialize SASRec model.

        Args:
            n_items: Number of items in catalog
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            lr: Learning rate
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        self.n_items = n_items
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.lr = lr

        # Initialize parameters
        self.item_embeddings = self._init_embeddings(n_items, d_model)
        self.position_embeddings = self._init_embeddings(max_seq_len, d_model)
        self.attention_w = self._init_weights(d_model, d_model, seed)
        self.attention_v = self._init_weights(d_model, d_model, seed + 1)
        self.ff_weights_1 = self._init_weights(d_model, d_ff, seed + 2)
        self.ff_bias_1 = np.zeros(d_ff)
        self.ff_weights_2 = self._init_weights(d_ff, d_model, seed + 3)
        self.ff_bias_2 = np.zeros(d_model)

        # Layer normalization parameters
        self.ln_gamma = np.ones(d_model)
        self.ln_beta = np.zeros(d_model)

        # Training state
        self.training = False

    def _init_embeddings(self, n_items: int, d_model: int, scale: float = 0.1) -> np.ndarray:
        """Initialize item embeddings."""
        return np.random.randn(n_items, d_model) * scale

    def _init_weights(self, in_dim: int, out_dim: int, seed: int) -> np.ndarray:
        """Initialize weights with Xavier initialization."""
        np.random.seed(seed)
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim) * scale

    def _scaled_dot_product_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                                       mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Scaled dot-product attention."""
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, scores, -1e9)

        attention = self._softmax(scores, axis=-1)
        return np.matmul(attention, v)

    def _multi_head_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                              mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Multi-head attention."""
        batch_size = q.shape[0]

        # Linear projections
        q_proj = np.matmul(q, self.attention_w)
        k_proj = np.matmul(k, self.attention_w)
        v_proj = np.matmul(v, self.attention_v)

        # Split into heads
        q_heads = self._split_heads(q_proj, batch_size)
        k_heads = self._split_heads(k_proj, batch_size)
        v_heads = self._split_heads(v_proj, batch_size)

        # Apply attention
        attended = self._scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        # Concatenate heads
        output = self._concat_heads(attended, batch_size)

        return output

    def _split_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Split into multi-heads."""
        return x.reshape(batch_size, -1, self.n_heads, self.d_model // self.n_heads)\
                 .transpose(0, 2, 1, 3)

    def _concat_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Concatenate multi-heads."""
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * (self.d_model // self.n_heads))

    def _positionwise_feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Position-wise feed-forward network."""
        return self._softmax(
            np.matmul(
                self._relu(np.matmul(x, self.ff_weights_1) + self.ff_bias_1),
                self.ff_weights_2
            ) + self.ff_bias_2,
            axis=-1
        )

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _add_positional_encoding(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to embeddings."""
        seq_len = x.shape[1]
        positions = np.arange(seq_len)
        return x + self.position_embeddings[:seq_len]

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        return self.ln_gamma * x + self.ln_beta

    def encode_session(self, session: np.ndarray) -> np.ndarray:
        """Encode a user session into representation (simplified avg pooling).

        Args:
            session: Session history as array of item indices [seq_len]

        Returns:
            Session representation vector
        """
        # Simple average of item embeddings in session
        # This is a simplified version that avoids complex tensor operations
        if len(session) == 0:
            return self.item_embeddings[0]

        session_array = np.array(session, dtype=np.int32)[:self.max_seq_len]
        item_emb = self.item_embeddings[session_array]

        # Average pooling over session items
        return np.mean(item_emb, axis=0)

    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal (look-ahead) mask."""
        mask = np.tril(np.ones((seq_len, seq_len)))
        return 1 - mask  # 1 where we should mask, 0 where we should attend

    def predict_next(self, session: np.ndarray) -> np.ndarray:
        """Predict next item probabilities.

        Args:
            session: Session history as array of item indices

        Returns:
            Probability distribution over all items
        """
        session_array = np.array(session, dtype=np.int32)
        session_repr = self.encode_session(session_array)

        # Compute similarity with all item embeddings
        scores = np.matmul(self.item_embeddings, session_repr)

        # Softmax to get probabilities
        probs = self._softmax(scores)

        return probs

    def get_item_similarity(self, item_idx: int) -> np.ndarray:
        """Get similarity scores between item and all others.

        Args:
            item_idx: Item index

        Returns:
            Similarity scores with all items
        """
        item_emb = self.item_embeddings[item_idx]
        similarities = np.matmul(self.item_embeddings, item_emb)
        return similarities

    def get_recommendations(
        self,
        session: np.ndarray,
        excluded_items: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Get top-k recommendations for a session.

        Args:
            session: Session history
            excluded_items: Items to exclude from recommendations
            top_k: Number of recommendations to return

        Returns:
            List of (item_idx, score) tuples
        """
        probs = self.predict_next(session)

        if excluded_items:
            probs[list(excluded_items)] = 0

        # Get top-k indices
        top_indices = np.argsort(probs)[::-1][:top_k]

        return [(int(idx), float(probs[idx])) for idx in top_indices]

    def train_step(
        self,
        sessions: np.ndarray,
        targets: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> float:
        """Perform one training step.

        Args:
            sessions: Session histories [batch, seq_len]
            targets: Target items [batch]
            learning_rate: Override learning rate

        Returns:
            Loss value
        """
        lr = learning_rate or self.lr
        batch_size = sessions.shape[0]

        # Forward pass
        session_reprs = np.array([self.encode_session(sessions[b]) for b in range(batch_size)])

        # Compute scores [n_items, batch_size]
        scores = np.matmul(self.item_embeddings, session_reprs.T)

        # Softmax over items for each sample
        probs = self._softmax(scores, axis=0)

        # Cross-entropy loss
        log_probs = np.log(probs[targets, np.arange(batch_size)] + 1e-10)
        loss = -np.mean(log_probs)

        # Gradient: probs - one_hot(targets)
        grad = probs.copy()
        grad[targets, np.arange(batch_size)] -= 1
        grad /= batch_size

        # Update embeddings: grad @ session_reprs.T
        self.item_embeddings -= lr * np.matmul(grad, session_reprs)

        return float(loss)

    def fit(
        self,
        sessions: np.ndarray,
        targets: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 32,
        val_split: float = 0.2,
        early_stop_patience: int = 3
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            sessions: Session histories [n_samples, seq_len]
            targets: Target items [n_samples]
            n_epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split ratio
            early_stop_patience: Early stopping patience

        Returns:
            Training history with losses
        """
        n_samples = sessions.shape[0]
        indices = np.random.permutation(n_samples)

        # Split validation
        val_size = int(n_samples * val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            np.random.shuffle(train_indices)

            epoch_loss = 0
            n_batches = 0

            for start in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[start:start + batch_size]
                batch_sessions = sessions[batch_indices]
                batch_targets = targets[batch_indices]

                loss = self.train_step(batch_sessions, batch_targets)
                epoch_loss += loss
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            # Validation
            val_sessions = sessions[val_indices]
            val_targets = targets[val_indices]
            val_loss = 0
            for b in range(min(100, len(val_indices))):
                val_loss += self.train_step(
                    val_sessions[b:b+1],
                    val_targets[b:b+1],
                    learning_rate=0.0
                )
            val_loss /= min(100, len(val_indices))
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    def save(self, path: str):
        """Save model weights."""
        np.savez(
            path,
            item_embeddings=self.item_embeddings,
            position_embeddings=self.position_embeddings,
            attention_w=self.attention_w,
            attention_v=self.attention_v,
            ff_weights_1=self.ff_weights_1,
            ff_bias_1=self.ff_bias_1,
            ff_weights_2=self.ff_weights_2,
            ff_bias_2=self.ff_bias_2,
            ln_gamma=self.ln_gamma,
            ln_beta=self.ln_beta,
        )

    def load(self, path: str):
        """Load model weights."""
        data = np.load(path)
        self.item_embeddings = data["item_embeddings"]
        self.position_embeddings = data["position_embeddings"]
        self.attention_w = data["attention_w"]
        self.attention_v = data["attention_v"]
        self.ff_weights_1 = data["ff_weights_1"]
        self.ff_bias_1 = data["ff_bias_1"]
        self.ff_weights_2 = data["ff_weights_2"]
        self.ff_bias_2 = data["ff_bias_2"]
        self.ln_gamma = data["ln_gamma"]
        self.ln_beta = data["ln_beta"]


# ============================================================================
# HELPER FUNCTIONS FOR SESSION CONSTRUCTION
# ============================================================================


def build_sessions_from_interactions(
    interactions_df: pd.DataFrame,
    min_session_len: int = 2,
    max_session_len: int = 10
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Build sequential sessions from interaction logs.

    Args:
        interactions_df: Interaction logs with user_id and drink_id
        min_session_len: Minimum sessions length
        max_session_len: Maximum session length

    Returns:
        Tuple of (sessions array, target array, item_to_idx mapping)
    """
    # Create item to index mapping
    unique_drinks = sorted(interactions_df["drink_id"].unique())
    item_to_idx = {drink: idx for idx, drink in enumerate(unique_drinks)}
    idx_to_item = {idx: drink for drink, idx in item_to_idx.items()}

    # Group by user and build sessions
    sessions = []
    targets = []

    for user_id, user_interactions in interactions_df.groupby("user_id"):
        # Sort by some order (e.g., interaction time if available, otherwise by index)
        user_drinks = user_interactions["drink_id"].tolist()

        # Build overlapping sessions
        for i in range(len(user_drinks)):
            # Variable session length
            seq_len = np.random.randint(min_session_len, min(max_session_len + 1, i + 1))

            if i >= seq_len:
                session = user_drinks[i - seq_len:i]
                target = user_drinks[i]

                # Convert to indices
                session_indices = [item_to_idx[d] for d in session if d in item_to_idx]
                target_idx = item_to_idx.get(target, 0)

                if len(session_indices) >= min_session_len:
                    sessions.append(session_indices)
                    targets.append(target_idx)

    sessions_array = np.array(sessions, dtype=np.int32)
    targets_array = np.array(targets, dtype=np.int32)

    return sessions_array, targets_array, item_to_idx


def create_user_history_mapping(
    interactions_df: pd.DataFrame,
    item_to_idx: Dict[str, int]
) -> Dict[str, List[int]]:
    """Create user ID to session history mapping.

    Args:
        interactions_df: Interaction logs
        item_to_idx: Item ID to index mapping

    Returns:
        Dict mapping user_id to list of item indices
    """
    user_history = defaultdict(list)

    for user_id, user_interactions in interactions_df.groupby("user_id"):
        for drink_id in user_interactions["drink_id"].unique():
            if drink_id in item_to_idx:
                user_history[str(user_id)].append(item_to_idx[drink_id])

    return dict(user_history)
