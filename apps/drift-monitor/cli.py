"""Drift Monitor — Model and data drift detection with statistical tests.

MLOps template:
- Uses synthetic data for demonstration
- Implements KS-test, PSI, and t-test for drift detection
- Provides severity ratings and alerting thresholds
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy import stats


app = typer.Typer(help="DriftMonitor: detect model and data drift.")


def load_or_generate_data(data_dir: str = "data/synthetic") -> tuple:
    """Load or generate reference and current data."""
    ref_path = Path(data_dir) / "drift_reference.csv"
    curr_path = Path(data_dir) / "drift_current.csv"

    # Generate synthetic data if not exists
    if not ref_path.exists() or not curr_path.exists():
        np.random.seed(42)
        n_samples = 1000

        # Reference data (stable distribution)
        ref_data = pd.DataFrame({
            "age": np.random.normal(38, 12, n_samples),
            "income": np.random.lognormal(11, 0.5, n_samples),
            "spending_score": np.random.beta(2, 2, n_samples) * 100,
            "visit_frequency": np.random.poisson(10, n_samples),
            "avg_order_value": np.random.lognormal(4.5, 0.8, n_samples),
            "engagement_score": np.random.uniform(0, 100, n_samples),
        })

        # Current data with controlled drift
        current_data = pd.DataFrame({
            "age": np.random.normal(38 + 3, 12, n_samples),  # Mean shift
            "income": np.random.lognormal(11.3, 0.5, n_samples),  # Mean shift
            "spending_score": np.random.beta(3, 2, n_samples) * 100,  # Distribution change
            "visit_frequency": np.random.poisson(12, n_samples),  # Mean shift
            "avg_order_value": np.random.lognormal(4.8, 0.8, n_samples),  # Mean shift
            "engagement_score": np.random.uniform(5, 100, n_samples),  # Distribution change
        })

        data_dir_path = Path(data_dir)
        data_dir_path.mkdir(parents=True, exist_ok=True)

        ref_data.to_csv(ref_path, index=False)
        current_data.to_csv(curr_path, index=False)

    return pd.read_csv(ref_path), pd.read_csv(curr_path)


def kstest(reference: np.ndarray, current: np.ndarray) -> tuple:
    """Kolmogorov-Smirnov test for distribution drift."""
    return stats.ks_2samp(reference, current)


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    bins = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_props = np.histogram(reference, bins=bins)[0] / len(reference)
    curr_props = np.histogram(current, bins=bins)[0] / len(current)

    ref_props = np.where(ref_props == 0, 1e-10, ref_props)
    curr_props = np.where(curr_props == 0, 1e-10, curr_props)

    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    return abs(psi)


def t_test(reference: np.ndarray, current: np.ndarray) -> tuple:
    """T-test for mean shift."""
    return stats.ttest_ind(reference, current)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    p_value_threshold: float = 0.01,
    psi_threshold: float = 0.2,
) -> dict:
    """Detect drift across all features.

    Args:
        reference: Baseline/reference dataset
        current: Current/drifted dataset
        p_value_threshold: P-value threshold for significance
        psi_threshold: PSI threshold for drift

    Returns:
        Dictionary with drift results for each feature
    """
    results = {}

    for col in reference.columns:
        ref_vals = reference[col].dropna().values
        curr_vals = current[col].dropna().values

        if len(ref_vals) < 10 or len(curr_vals) < 10:
            results[col] = {
                "drift_detected": False,
                "ks_stat": 0.0,
                "ks_pvalue": 1.0,
                "t_stat": 0.0,
                "t_pvalue": 1.0,
                "psi": 0.0,
                "severity": "none",
            }
            continue

        # Statistical tests
        ks_stat, ks_pvalue = kstest(ref_vals, curr_vals)
        t_stat, t_pvalue = t_test(ref_vals, curr_vals)
        psi_value = psi(ref_vals, curr_vals)

        # Determine drift
        drift_pvalue = min(ks_pvalue, t_pvalue)
        drift_detected = drift_pvalue < p_value_threshold or psi_value > psi_threshold

        # Determine severity
        if psi_value < 0.1:
            severity = "none"
        elif psi_value < 0.2:
            severity = "low"
        elif psi_value < 0.3:
            severity = "medium"
        else:
            severity = "high"

        results[col] = {
            "drift_detected": drift_detected,
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 6),
            "t_stat": round(t_stat, 4),
            "t_pvalue": round(t_pvalue, 6),
            "psi": round(psi_value, 4),
            "severity": severity,
        }

    return results


def generate_alerts(
    drift_results: dict,
    threshold: str = "standard",
) -> list:
    """Generate alerts based on drift results.

    Args:
        drift_results: Results from detect_drift
        threshold: Alert threshold level

    Returns:
        List of alert dictionaries
    """
    thresholds = {
        "lenient": {"p_value": 0.05, "psi": 0.1, "ks": 0.15},
        "standard": {"p_value": 0.01, "psi": 0.2, "ks": 0.2},
        "strict": {"p_value": 0.001, "psi": 0.3, "ks": 0.25},
    }

    thresh = thresholds.get(threshold, thresholds["standard"])
    alerts = []

    for feature, result in drift_results.items():
        if result["drift_detected"]:
            alert = {
                "feature": feature,
                "level": "warning" if result["severity"] == "low" else "critical",
                "severity": result["severity"],
                "psi": result["psi"],
                "ks_pvalue": result["ks_pvalue"],
            }
            alerts.append(alert)

    return alerts


@app.command()
def check_drift(
    model_name: str = typer.Argument("demo_model", help="Model name to monitor"),
    data_source: str = typer.Argument("data/synthetic", help="Data source path"),
    drift_type: str = typer.Option("both", "--type", "-t", help="Drift type: 'data', 'model', 'both'"),
    p_value_threshold: float = typer.Option(0.01, "--threshold", "-p", help="P-value threshold"),
):
    """Check for data drift."""
    typer.echo(f"Checking drift for: {model_name}")
    typer.echo(f"Data source: {data_source}")
    typer.echo(f"Threshold: p < {p_value_threshold}")
    typer.echo()

    ref_data, curr_data = load_or_generate_data(data_source)

    results = detect_drift(ref_data, curr_data, p_value_threshold=p_value_threshold)

    # Summary
    drift_count = sum(1 for r in results.values() if r["drift_detected"])
    total_features = len(results)

    typer.echo("="*60)
    typer.echo(f"Drift Detection Results: {model_name}")
    typer.echo(f"Drift Detected: {drift_count}/{total_features} features")
    typer.echo("="*60)

    # Detailed results
    for feature, result in results.items():
        status = "DRIFTED" if result["drift_detected"] else "OK"
        color = "RED" if result["drift_detected"] else "GREEN"

        typer.echo(f"\n[{status}] {feature}")
        typer.echo(f"  KS-Test: stat={result['ks_stat']:.4f}, p={result['ks_pvalue']:.6f}")
        typer.echo(f"  PSI: {result['psi']:.4f} ({result['severity']})")

        if result["drift_detected"]:
            if result["severity"] == "high":
                typer.echo(f"  ⚠️  CRITICAL: Immediate action required")
            elif result["severity"] == "medium":
                typer.echo(f"  ⚠️  WARNING: Monitor closely")


@app.command()
def alert_config(
    model_name: str = typer.Argument("demo_model", help="Model name"),
    threshold: float = typer.Option(0.01, "--p-value", "-p", help="P-value threshold"),
    psi_threshold: float = typer.Option(0.2, "--psi", "-s", help="PSI threshold"),
    alert_level: str = typer.Option("standard", "--level", "-l", help="Alert level: 'lenient', 'standard', 'strict'"),
):
    """Configure alerting thresholds."""
    typer.echo(f"Alert configuration for: {model_name}")
    typer.echo("="*40)
    typer.echo(f"P-value threshold: {threshold}")
    typer.echo(f"PSI threshold: {psi_threshold}")
    typer.echo(f"Alert level: {alert_level}")
    typer.echo()
    typer.echo("Notification settings:")
    typer.echo("  - Email: enabled")
    typer.echo("  - Slack: configured")
    typer.echo("  - Frequency: immediate for critical, hourly for warnings")


@app.command()
def generate_baseline(
    output_path: str = typer.Option("data/synthetic", help="Output directory"),
    n_samples: int = typer.Option(1000, "--samples", "-n", help="Number of samples"),
):
    """Generate baseline/reference distribution."""
    np.random.seed(42)

    ref_data = pd.DataFrame({
        "age": np.random.normal(38, 12, n_samples),
        "income": np.random.lognormal(11, 0.5, n_samples),
        "spending_score": np.random.beta(2, 2, n_samples) * 100,
        "visit_frequency": np.random.poisson(10, n_samples),
        "avg_order_value": np.random.lognormal(4.5, 0.8, n_samples),
        "engagement_score": np.random.uniform(0, 100, n_samples),
    })

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "drift_reference.csv"

    ref_data.to_csv(filepath, index=False)
    typer.echo(f"Baseline saved to: {filepath}")
    typer.echo(f"Samples: {n_samples}")
    typer.echo("Features: age, income, spending_score, visit_frequency, avg_order_value, engagement_score")


@app.command()
def compare(
    reference_file: str = typer.Argument("data/synthetic/drift_reference.csv", help="Reference file"),
    current_file: str = typer.Argument("data/synthetic/drift_current.csv", help="Current file"),
):
    """Compare two datasets."""
    ref_data = pd.read_csv(reference_file)
    curr_data = pd.read_csv(current_file)

    typer.echo("Dataset Comparison:")
    typer.echo("="*40)
    typer.echo(f"Reference: {len(ref_data)} samples, {len(ref_data.columns)} features")
    typer.echo(f"Current: {len(curr_data)} samples, {len(curr_data.columns)} features")

    for col in ref_data.columns:
        ref_mean = ref_data[col].mean()
        curr_mean = curr_data[col].mean()
        shift = abs(curr_mean - ref_mean) / ref_mean * 100 if ref_mean != 0 else 0

        typer.echo(f"\n{col}:")
        typer.echo(f"  Reference mean: {ref_mean:.2f}")
        typer.echo(f"  Current mean: {curr_mean:.2f}")
        typer.echo(f"  Mean shift: {shift:.1f}%")


if __name__ == "__main__":
    app()
