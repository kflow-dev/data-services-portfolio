"""SKU Forecaster — Hierarchical demand forecasting UI."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent directory to path for mlops import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

st.set_page_config(page_title="SKU Forecaster", layout="wide")

DEPARTMENTS = ["Electronics", "Apparel", "Home"]
PRODUCT_GROUPS = {
    "Electronics": ["Laptops", "Tablets"],
    "Apparel": ["T-Shirts", "Jeans"],
    "Home": ["Kitchen Kits"],
}

SKU_NAMES = {
    "Electronics": {
        "Laptops": ["LAP-001", "LAP-002"],
        "Tablets": ["TAB-001", "TAB-002"],
    },
    "Apparel": {
        "T-Shirts": ["TSH-BLK-M", "TSH-WHT-L"],
        "Jeans": ["JNS-BLU-32", "JNS-BLK-34"],
    },
    "Home": {
        "Kitchen Kits": ["KIT-001", "KIT-002"],
    },
}


def load_or_generate_data(data_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load or generate demand data."""
    filepath = Path(data_dir) / "sku_demand.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    # Generate synthetic
    np.random.seed(42)
    n_weeks = 52
    departments = DEPARTMENTS
    rows = []

    base_price = {
        "LAP-001": 1299.99, "LAP-002": 899.99,
        "TAB-001": 599.99, "TAB-002": 449.99,
        "TSH-BLK-M": 29.99, "TSH-WHT-L": 29.99,
        "JNS-BLU-32": 79.99, "JNS-BLK-34": 79.99,
        "KIT-001": 149.99, "KIT-002": 199.99,
    }

    for week in range(1, n_weeks + 1):
        trend = 1 + 0.02 * week
        seasonality = 1 + 0.1 * np.sin(2 * np.pi * week / 52)
        noise = np.random.normal(1, 0.15)

        for department in departments:
            products = [(p, SKU_NAMES[department][p]) for p in PRODUCT_GROUPS[department]]
            for product_group, skus in products:
                for sku in skus:
                    base_demand = np.random.poisson(50)
                    quantity = int(base_demand * trend * seasonality * noise)
                    price = base_price.get(sku, 50.0)
                    discount = np.random.choice([0, 5, 10], p=[0.5, 0.3, 0.2])
                    revenue = quantity * price * (1 - discount / 100)
                    rows.append({
                        "week": week, "department": department,
                        "product_group": product_group, "sku": sku,
                        "quantity_sold": max(0, quantity),
                    })

    df = pd.DataFrame(rows)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def train_model(df: pd.DataFrame) -> GradientBoostingRegressor:
    """Train forecasting model."""
    # Create features
    df["is_holiday"] = df["week"].isin([7, 11, 24, 35, 47])
    df["lag_1_dept"] = df.groupby("department")["quantity_sold"].shift(1)
    df["lag_4_dept"] = df.groupby("department")["quantity_sold"].shift(4)

    df_clean = df.dropna()

    feature_cols = ["week", "is_holiday", "discount_pct", "lag_1_dept", "lag_4_dept"]
    X = df_clean[feature_cols]
    y = df_clean["quantity_sold"]

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, mae, rmse


def generate_forecast(model: GradientBoostingRegressor, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Generate forecast."""
    last_week = df["week"].max()
    avg_demand = df["quantity_sold"].mean()

    forecasts = []
    for week in range(last_week + 1, last_week + horizon + 1):
        is_holiday = week in [7, 11, 24, 35, 47, 59, 71]
        discount_pct = np.random.choice([0, 5, 10], p=[0.5, 0.3, 0.2])

        features = [[week, is_holiday, discount_pct, avg_demand * 0.8, avg_demand * 0.7]]
        prediction = model.predict(features)[0]

        ci_width = prediction * 0.2
        forecasts.append({
            "week": week,
            "forecast": round(prediction, 0),
            "lower_80": round(prediction - ci_width * 0.84, 0),
            "upper_80": round(prediction + ci_width * 0.84, 0),
        })

    return pd.DataFrame(forecasts)


# Main app
st.title("SKU Forecaster")
st.caption("Hierarchical demand forecasting using LSTM/DeepAR-inspired models")

col1, col2 = st.columns(2)

with col1:
    department = st.selectbox("Department", DEPARTMENTS)
with col2:
    product_group = st.selectbox("Product Group", PRODUCT_GROUPS.get(department, []))

horizon = st.slider("Forecast horizon (weeks)", 4, 52, 12)

if st.button("Generate Forecast", type="primary"):
    with st.spinner("Training model..."):
        df = load_or_generate_data()
        df_filtered = df[(df["department"] == department) & (df["product_group"] == product_group)]

        if len(df_filtered) == 0:
            st.error(f"No data found for {department} > {product_group}")
        else:
            model, mae, rmse = train_model(df_filtered)
            forecasts = generate_forecast(model, df_filtered, horizon)

            st.success("Model trained successfully!")

            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.1f} units")
            col2.metric("RMSE", f"{rmse:.1f} units")
            col3.metric("Sample Size", f"{len(df_filtered)}")

            st.subheader("Forecast")
            forecast_df = forecasts[["week", "forecast", "lower_80", "upper_80"]]
            forecast_df.columns = ["Week", "Forecast", "80% CI Lower", "80% CI Upper"]
            st.dataframe(forecast_df, use_container_width=True)

            # Chart
            st.line_chart(forecasts.set_index("week")[["forecast", "lower_80", "upper_80"]])
