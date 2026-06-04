"""Model evaluator module for computing performance metrics."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ClassificationMetrics:
    """Metrics for classification models."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "auc_roc": round(self.auc_roc, 4),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


@dataclass
class RegressionMetrics:
    """Metrics for regression/forecasting models."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "mse": round(self.mse, 4),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mape": round(self.mape, 2),
            "r2_score": round(self.r2_score, 4),
        }


@dataclass
class ClusteringMetrics:
    """Metrics for clustering models."""
    silhouette_score: float
    inertial: float
    n_clusters: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "silhouette_score": round(self.silhouette_score, 4),
            "inertial": round(self.inertial, 2),
            "n_clusters": self.n_clusters,
        }


class ModelEvaluator:
    """Evaluate model performance using various metrics."""

    @staticmethod
    def classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> ClassificationMetrics:
        """Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC-ROC)

        Returns:
            ClassificationMetrics object
        """
        # Accuracy
        accuracy = np.mean(y_true == y_pred)

        # Precision, Recall, F1 (binary)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AUC-ROC
        if y_proba is not None:
            fpr, tpr, _ = stats.roc_curve(y_true, y_proba)
            auc_roc = np.trapz(tpr, fpr)
        else:
            auc_roc = 0.5  # Random guess

        # Confusion matrix
        cm = np.array([[np.sum((y_true == 0) & (y_pred == 0)),
                        np.sum((y_true == 0) & (y_pred == 1))],
                       [np.sum((y_true == 1) & (y_pred == 0)),
                        np.sum((y_true == 1) & (y_pred == 1))]])

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_roc=auc_roc,
            confusion_matrix=cm,
        )

    @staticmethod
    def regression(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> RegressionMetrics:
        """Compute regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RegressionMetrics object
        """
        # MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # RMSE
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))

        return RegressionMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2_score=r2_score,
        )

    @staticmethod
    def clustering(
        labels: np.ndarray,
        X: np.ndarray,
    ) -> ClusteringMetrics:
        """Compute clustering quality metrics.

        Uses silhouette score for cluster quality assessment.

        Args:
            labels: Cluster labels
            X: Feature matrix

        Returns:
            ClusteringMetrics object
        """
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        # Number of clusters
        n_clusters = len(np.unique(labels))

        # Inertia
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        inertial = kmeans.inertia_

        # Silhouette score
        if n_clusters > 1:
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = 0.0

        return ClusteringMetrics(
            silhouette_score=sil_score,
            inertial=inertial,
            n_clusters=n_clusters,
        )

    @staticmethod
    def time_series_forecast(
        y_true: pd.Series,
        y_pred: pd.Series,
        y_lower: Optional[pd.Series] = None,
        y_upper: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Compute time series forecasting metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_lower: Lower confidence bound (optional)
            y_upper: Upper confidence bound (optional)

        Returns:
            Dictionary with forecasting metrics
        """
        # Point forecast metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        rmse = np.sqrt(mse)

        # Coverage (if CI bounds provided)
        coverage = None
        if y_lower is not None and y_upper is not None:
            covered = (y_true >= y_lower) & (y_true <= y_upper)
            coverage = covered.mean()

        return {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "coverage": round(coverage, 4) if coverage is not None else None,
        }
