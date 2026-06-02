# Drift Monitor

Model and data drift detection with statistical tests.

## Overview

Drift Monitor detects distribution shifts between reference and current data using statistical hypothesis testing. It provides:

- **KS-Test**: Kolmogorov-Smirnov test for distribution drift
- **PSI**: Population Stability Index for categorical drift
- **T-Test**: Mean shift detection
- **Severity ratings**: None, Low, Medium, High
- **Alerting**: Configurable thresholds for notifications

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Drift Monitor                           │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py check-drift demo_model data/synthetic   │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import DriftMonitor                 │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/drift_reference.csv,                  │
│        data/synthetic/drift_current.csv                     │
│  Tests: KS-test, PSI, t-test                                │
│  Output: Drift detection report with severity ratings      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit scikit-learn pandas numpy typer scipy
```

## Usage

### CLI

```bash
# Check drift with default thresholds
python cli.py check-drift demo_model data/synthetic

# Check with custom p-value threshold
python cli.py check-drift demo_model data/synthetic -p 0.05

# Configure alerting thresholds
python cli.py alert-config demo_model -p 0.01 -s 0.2 -l standard

# Generate baseline/reference distribution
python cli.py generate-baseline -n 1000

# Compare two datasets directly
python cli.py compare data/ref.csv data/current.csv
```

### Streamlit UI

```bash
# From project root
streamlit run apps/drift-monitor/streamlit_app.py

# With custom port
streamlit run apps/drift-monitor/streamlit_app.py --server.port 8504
```

### Jupyter Notebook

```bash
# From project root
jupyter notebook apps/drift-monitor/notebooks/drift_detection_example.ipynb

# Or start jupyter and navigate to:
# apps/drift-monitor/notebooks/drift_detection_example.ipynb
```

### As a Library

```python
from catalog import DriftMonitor
import pandas as pd

# Load data
reference = pd.read_csv("data/synthetic/drift_reference.csv")
current = pd.read_csv("data/synthetic/drift_current.csv")

# Initialize drift monitor
monitor = DriftMonitor(p_value_threshold=0.01, psi_threshold=0.2)

# Detect drift
results = monitor.detect(reference, current)

# Print results
for feature, result in results.items():
    status = "DRIFTED" if result["drift_detected"] else "OK"
    print(f"[{status}] {feature}: PSI={result['psi']:.4f} ({result['severity']})")

# Get alerts
alerts = monitor.generate_alerts(results, threshold="standard")
for alert in alerts:
    print(f"Alert: {alert['feature']} - {alert['severity']} level")
```

## Model Details

### Statistical Tests

| Test | Purpose | Drift Detection |
|------|---------|-----------------|
| **KS-Test** | Compare distributions | p < 0.01 |
| **PSI** | Population stability | PSI > 0.2 |
| **T-Test** | Mean shift | p < 0.01 |

### PSI Severity Levels

| PSI Value | Severity | Action |
|-----------|----------|--------|
| < 0.1 | None | No action needed |
| 0.1 - 0.2 | Low | Monitor closely |
| 0.2 - 0.3 | Medium | Investigate |
| > 0.3 | High | Immediate action required |

### Alert Thresholds

| Level | P-value | PSI | KS Statistic |
|-------|---------|-----|--------------|
| Lenient | 0.05 | 0.1 | 0.15 |
| Standard | 0.01 | 0.2 | 0.2 |
| Strict | 0.001 | 0.3 | 0.25 |

## Data Format

Input CSV format (both reference and current):

```csv
age,income,spending_score,visit_frequency,avg_order_value,engagement_score
38,95000,72,12,85.50,68
42,120000,45,6,120.00,52
35,85000,58,8,95.00,75
```

**Required columns:**
- Numeric features only (no IDs or categorical columns)
- Reference data should represent stable baseline distribution
- Current data should represent production/monitoring distribution

**Minimum samples:** 10 samples per feature for reliable testing

## Project Structure

```
drift-monitor/
├── cli.py              # Command-line interface
├── streamlit_app.py    # Web UI
├── catalog.py          # Public API library (to be created)
├── README.md           # This file
├── data/
│   └── synthetic/
│       ├── drift_reference.csv
│       └── drift_current.csv
└── notebooks/
    └── drift_detection_example.ipynb
```

## CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `check-drift` | Check for data drift | `python cli.py check-drift model_name data_dir` |
| `alert-config` | Configure alerting | `python cli.py alert-config model -p 0.01 -s 0.2` |
| `generate-baseline` | Generate reference data | `python cli.py generate-baseline -n 1000` |
| `compare` | Compare two datasets | `python cli.py compare ref.csv curr.csv` |

## Drift Monitoring Workflow

```
1. Generate Baseline
   └─> python cli.py generate-baseline -n 1000

2. Regular Monitoring (Daily/Weekly)
   └─> python cli.py check-drift my_model data/synthetic

3. Investigate Alerts
   └─> Review severity levels and feature drift

4. Retrain if Needed
   └─> If drift is high severity, retrain model
```

## License

See parent directory for license information.
