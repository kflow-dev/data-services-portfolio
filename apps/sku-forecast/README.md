# SKU Forecaster

Hierarchical demand forecasting using ensemble ML models.

## Overview

SKU Forecaster predicts product demand at multiple hierarchy levels:
- **Department level** (e.g., Electronics, Apparel)
- **Product group level** (e.g., Laptops, T-Shirts)
- **SKU level** (e.g., LAP-001, TSH-BLK-M)

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SKU Forecaster                           │
├─────────────────────────────────────────────────────────────┤
│  CLI: python cli.py forecast <department> <product_group>   │
│  Streamlit: streamlit run streamlit_app.py                 │
│  Library: from catalog import SKUForecaster                │
├─────────────────────────────────────────────────────────────┤
│  Data: data/synthetic/sku_demand.csv                        │
│  Model: GradientBoostingRegressor (sklearn)                │
│  Features: lag features, holiday indicators, seasonality   │
│  Output: Point forecast + 80%/95% confidence intervals     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install streamlit scikit-learn pandas numpy typer
```

## Usage

### CLI

```bash
# Forecast demand for a product
python cli.py forecast "Electronics" "Laptops" --weeks 12

# Forecast with custom data directory
python cli.py forecast "Apparel" "T-Shirts" --weeks 8 --data-dir data/custom/

# Evaluate model performance
python cli.py evaluate

# Train and save model
python cli.py train --save-path models/

# Generate baseline data
python cli.py train --generate-data
```

### Streamlit UI

```bash
# From project root
streamlit run apps/sku-forecast/streamlit_app.py

# With custom port
streamlit run apps/sku-forecast/streamlit_app.py --server.port 8502
```

### Jupyter Notebook

```bash
# From project root
jupyter notebook apps/sku-forecast/notebooks/demand_forecasting_example.ipynb

# Or start jupyter and navigate to:
# apps/sku-forecast/notebooks/demand_forecasting_example.ipynb
```

### As a Library

```python
from catalog import SKUForecaster
import pandas as pd

# Load data
df = pd.read_csv("data/synthetic/sku_demand.csv")

# Filter and train
df_filtered = df[(df["department"] == "Electronics") &
                 (df["product_group"] == "Laptops")]

forecaster = SKUForecaster(n_estimators=100, max_depth=5)
forecaster.train(df_filtered)

# Generate forecast
forecasts = forecaster.forecast(df_filtered, horizon=12)
print(forecasts)

# Get metrics
metrics = forecaster.get_metrics()
print(f"MAE: {metrics['mae']}, RMSE: {metrics['rmse']}")
```

## Model Details

### Features Used

| Feature | Description |
|---------|-------------|
| `week` | Week number (trend) |
| `is_holiday` | Binary holiday indicator |
| `discount_pct` | Discount percentage |
| `lag_1_dept` | Previous week's department average |
| `lag_4_dept` | 4 weeks ago's department average |

### Algorithm

**Gradient Boosting Regressor**:
- Ensemble of decision trees built sequentially
- Each tree corrects errors from previous trees
- Configurable depth and learning rate
- Robust to outliers and non-linear relationships

### Performance Metrics

Typical performance on test data:
- **MAE**: ~10-15 units
- **RMSE**: ~15-20 units
- **MAPE**: ~15-25%

## Data Format

Input CSV format:

```csv
week,department,product_group,sku,quantity_sold,unit_price,discount_pct,revenue,inventory_level,is_holiday
1,Electronics,Laptops,LAP-001,45,1299.99,5,55249.58,50,false
2,Electronics,Laptops,LAP-001,48,1299.99,0,62399.52,47,false
```

**Required columns:**
- `week`: Week number (integer)
- `department`: Department name (e.g., Electronics, Apparel)
- `product_group`: Product group (e.g., Laptops, T-Shirts)
- `sku`: SKU identifier
- `quantity_sold`: Units sold

**Optional columns:**
- `unit_price`: Unit price
- `discount_pct`: Discount percentage
- `is_holiday`: Boolean holiday indicator

## Project Structure

```
sku-forecast/
├── cli.py              # Command-line interface
├── streamlit_app.py    # Web UI
├── catalog.py          # Public API library
├── README.md           # This file
└── requirements.txt    # Dependencies (optional)
```

## License

See parent directory for license information.
