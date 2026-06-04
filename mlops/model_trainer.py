"""Model trainer module with reusable ML training logic."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ModelTrainer:
    """Train and configure ML models with consistent interface."""

    def __init__(
        self,
        model_type: str = "auto",
        random_state: int = 42,
        test_size: float = 0.2,
        **model_kwargs,
    ):
        """Initialize model trainer.

        Args:
            model_type: Type of model ('classifier', 'regressor', 'auto')
            random_state: Random seed for reproducibility
            test_size: Test set proportion
            **model_kwargs: Additional model hyperparameters
        """
        self.model_type = model_type
        self.random_state = random_state
        self.test_size = test_size
        self.model_kwargs = model_kwargs
        self._model: Optional[BaseEstimator] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_names: Optional[List[str]] = None

    def _create_model(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Create model instance based on data type."""
        is_classification = isinstance(y, pd.Series) and y.dtype.name == "category"
        is_classification = is_classification or y.dtype in ["object", "category"]
        is_classification = is_classification or (
            len(np.unique(y)) < 10 and
            np.all(np.mod(y, 1) == 0)
        )

        if self.model_type == "auto":
            model_type = "classifier" if is_classification else "regressor"
        else:
            model_type = self.model_type

        if model_type == "classifier":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                **self.model_kwargs,
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                **self.model_kwargs,
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fit model to data.

        Args:
            X: Feature dataframe
            y: Target series
            feature_names: Optional feature names

        Returns:
            Training results with metrics
        """
        # Store feature names
        if feature_names is None:
            feature_names = list(X.columns)
        self._feature_names = feature_names

        # Convert to numpy
        X_np = X.values
        y_np = y.values if isinstance(y, pd.Series) else np.array(y)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        # Encode target if categorical
        if y.dtype in ["object", "category"]:
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y_np)
        else:
            y_encoded = y_np

        # Create and train model
        self._model = self._create_model(X_scaled, y_encoded)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded if self.model_type == "classifier" else None,
        )

        # Fit model
        self._model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = self._model.predict(X_val)

        # Compute metrics
        if isinstance(self._model, (RandomForestClassifier, LogisticRegression)):
            y_val_proba = self._model.predict_proba(X_val)[:, 1]
            metrics = self._classification_metrics(y_val, y_val_pred, y_val_proba)
        else:
            metrics = self._regression_metrics(y_val, y_val_pred)

        return {
            "model_type": self.model_type,
            "n_samples": len(X),
            "n_features": len(feature_names),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "metrics": metrics,
        }

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.5

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
        }

    def _regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        return {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "mape": round(mape, 2),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature dataframe

        Returns:
            Predictions array
        """
        if self._model is None:
            raise ValueError("Model not trained yet")

        X_np = X.values
        X_scaled = self._scaler.transform(X_np)

        predictions = self._model.predict(X_scaled)

        # Decode if categorical target
        if self._label_encoder is not None:
            predictions = self._label_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Feature dataframe

        Returns:
            Probability array for positive class
        """
        if self._model is None:
            raise ValueError("Model not trained yet")

        X_np = X.values
        X_scaled = self._scaler.transform(X_np)

        return self._model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(
        self,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self._model is None:
            raise ValueError("Model not trained yet")

        if self._feature_names is None:
            raise ValueError("Feature names not available")

        importances = self._model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances,
        })

        # Sort and return top features
        return importance_df.sort_values("importance", ascending=False).head(top_n)

    def get_model(self) -> BaseEstimator:
        """Get the trained model."""
        if self._model is None:
            raise ValueError("Model not trained yet")
        return self._model
