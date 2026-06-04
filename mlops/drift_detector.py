"""Drift detection module for monitoring data and model drift."""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature_name: str
    drift_detected: bool
    p_value: float
    test_statistic: float
    drift_type: Optional[str]
    severity: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "drift_detected": self.drift_detected,
            "p_value": round(self.p_value, 6),
            "test_statistic": round(self.test_statistic, 4),
            "drift_type": self.drift_type,
            "severity": self.severity,
        }


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    overall_drift: bool
    features: List[DriftResult]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_drift": self.overall_drift,
            "features": [f.to_dict() for f in self.features],
            "summary": self.summary,
        }


class DriftDetector:
    """Detect data drift and model performance degradation."""

    # Severity thresholds
    DRIFT_THRESHOLDS = {
        "lenient": {"p_value": 0.05, "psi": 0.1, "ks": 0.15},
        "standard": {"p_value": 0.01, "psi": 0.2, "ks": 0.2},
        "strict": {"p_value": 0.001, "psi": 0.3, "ks": 0.25},
    }

    SEVERITY_LABELS = {
        "none": "No drift",
        "low": "Minor drift",
        "medium": "Moderate drift",
        "high": "Significant drift",
        "critical": "Critical drift detected",
    }

    def __init__(
        self,
        p_value_threshold: float = 0.01,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.2,
    ):
        """Initialize drift detector.

        Args:
            p_value_threshold: P-value threshold for statistical tests
            psi_threshold: PSI threshold for categorical drift
            ks_threshold: KS statistic threshold for distribution drift
        """
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.thresholds = {
            "p_value": p_value_threshold,
            "psi": psi_threshold,
            "ks": ks_threshold,
        }

    def kstest(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for distribution drift.

        Args:
            reference: Baseline/reference distribution
            current: Current/drifted distribution

        Returns:
            Tuple of (KS statistic, p-value)
        """
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return ks_stat, p_value

    def psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index.

        Args:
            reference: Baseline/reference distribution
            current: Current/drifted distribution
            n_bins: Number of bins for histogram

        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        bins = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )

        # Calculate proportions in each bin
        ref_props = np.histogram(reference, bins=bins)[0] / len(reference)
        curr_props = np.histogram(current, bins=bins)[0] / len(current)

        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 1e-10, ref_props)
        curr_props = np.where(curr_props == 0, 1e-10, curr_props)

        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        return abs(psi)

    def t_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """Perform t-test for mean shift detection.

        Args:
            reference: Baseline/reference distribution
            current: Current/drifted distribution

        Returns:
            Tuple of (t-statistic, p-value)
        """
        t_stat, p_value = stats.ttest_ind(reference, current)
        return t_stat, p_value

    def detect_feature_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> DriftResult:
        """Detect drift for a single feature.

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature

        Returns:
            DriftResult for the feature
        """
        # Handle categorical vs numeric
        if reference.dtype in ["object", "category"] or current.dtype in ["object", "category"]:
            return self._detect_categorical_drift(reference, current, feature_name)
        else:
            return self._detect_numeric_drift(reference, current, feature_name)

    def _detect_numeric_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> DriftResult:
        """Detect drift for numeric features."""
        ref_vals = reference.dropna().values
        curr_vals = current.dropna().values

        if len(ref_vals) < 10 or len(curr_vals) < 10:
            return DriftResult(
                feature_name=feature_name,
                drift_detected=False,
                p_value=1.0,
                test_statistic=0.0,
                drift_type=None,
                severity="none",
            )

        # KS test for distribution shift
        ks_stat, ks_pvalue = self.kstest(ref_vals, curr_vals)

        # T-test for mean shift
        t_stat, t_pvalue = self.t_test(ref_vals, curr_vals)

        # PSI for magnitude of change
        psi_value = self.psi(ref_vals, curr_vals)

        # Determine drift type and significance
        drift_p_value = min(ks_pvalue, t_pvalue)
        drift_detected = drift_p_value < self.p_value_threshold

        # Determine severity
        if psi_value < 0.1:
            severity = "none"
            drift_type = None
        elif psi_value < 0.2:
            severity = "low"
            drift_type = "minor_distribution_shift"
        elif psi_value < 0.3:
            severity = "medium"
            drift_type = "moderate_shift"
        else:
            severity = "high"
            drift_type = "significant_drift"

        # Use KS statistic as primary test statistic
        test_stat = ks_stat

        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=drift_p_value,
            test_statistic=test_stat,
            drift_type=drift_type,
            severity=severity,
        )

    def _detect_categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> DriftResult:
        """Detect drift for categorical features."""
        ref_vals = reference.dropna().values
        curr_vals = current.dropna().values

        if len(ref_vals) < 10 or len(curr_vals) < 10:
            return DriftResult(
                feature_name=feature_name,
                drift_detected=False,
                p_value=1.0,
                test_statistic=0.0,
                drift_type=None,
                severity="none",
            )

        # Calculate PSI
        psi_value = self.psi(ref_vals, curr_vals)

        # Chi-square test for distribution change
        ref_unique = np.unique(ref_vals)
        ref_counts = np.array([np.sum(ref_vals == v) for v in ref_unique])
        curr_counts = np.array([np.sum(curr_vals == v) for v in ref_unique])

        # Expected counts based on reference distribution
        total = len(curr_vals)
        expected = ref_counts / len(ref_vals) * total

        if np.sum(expected == 0) > 0:
            chi2_stat = 0
            chi2_pvalue = 1.0
        else:
            chi2_stat, chi2_pvalue = stats.chisquare(curr_counts, expected)

        # Determine severity
        if psi_value < 0.1:
            severity = "none"
            drift_type = None
        elif psi_value < 0.2:
            severity = "low"
            drift_type = "minor_category_shift"
        elif psi_value < 0.3:
            severity = "medium"
            drift_type = "moderate_shift"
        else:
            severity = "high"
            drift_type = "significant_category_drift"

        return DriftResult(
            feature_name=feature_name,
            drift_detected=chi2_pvalue < self.p_value_threshold,
            p_value=chi2_pvalue,
            test_statistic=psi_value,
            drift_type=drift_type,
            severity=severity,
        )

    def detect_dataset_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> DriftReport:
        """Detect drift across an entire dataset.

        Args:
            reference: Reference dataset
            current: Current dataset

        Returns:
            DriftReport with results for all features
        """
        features = []
        drift_detected_count = 0

        for col in reference.columns:
            if col not in current.columns:
                continue

            result = self.detect_feature_drift(
                reference[col],
                current[col],
                col,
            )
            features.append(result)

            if result.drift_detected:
                drift_detected_count += 1

        # Overall drift determination
        overall_drift = drift_detected_count > 0

        # Summary statistics
        summary = {
            "total_features": len(features),
            "features_with_drift": drift_detected_count,
            "drift_rate": drift_detected_count / len(features) if features else 0,
            "severity_counts": {
                "none": sum(1 for f in features if f.severity == "none"),
                "low": sum(1 for f in features if f.severity == "low"),
                "medium": sum(1 for f in features if f.severity == "medium"),
                "high": sum(1 for f in features if f.severity == "high"),
            },
        }

        return DriftReport(
            overall_drift=overall_drift,
            features=features,
            summary=summary,
        )

    def detect_model_drift(
        self,
        predictions_reference: np.ndarray,
        predictions_current: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Detect model output drift.

        Args:
            predictions_reference: Reference predictions
            predictions_current: Current predictions
            thresholds: Optional custom thresholds

        Returns:
            Dictionary with drift analysis results
        """
        if thresholds is None:
            thresholds = self.thresholds

        # Distribution drift
        ks_stat, ks_pvalue = self.kstest(
            predictions_reference,
            predictions_current,
        )

        # Mean shift
        t_stat, t_pvalue = self.t_test(
            predictions_reference,
            predictions_current,
        )

        # PSI
        psi_value = self.psi(
            predictions_reference,
            predictions_current,
        )

        # Drift detection
        drift_detected = (
            ks_pvalue < self.p_value_threshold or
            psi_value > self.psi_threshold
        )

        # Severity
        if psi_value < 0.1:
            severity = "none"
        elif psi_value < 0.2:
            severity = "low"
        elif psi_value < 0.3:
            severity = "medium"
        else:
            severity = "high"

        return {
            "drift_detected": drift_detected,
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 6),
            "psi": round(psi_value, 4),
            "mean_shift_t_stat": round(t_stat, 4),
            "mean_shift_pvalue": round(t_pvalue, 6),
            "severity": severity,
            "severity_label": self.SEVERITY_LABELS[severity],
        }
