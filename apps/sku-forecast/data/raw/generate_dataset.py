"""Generate realistic hierarchical demand dataset for fashion retail forecasting.

This script creates a comprehensive synthetic dataset with all required columns
for multi-horizon, multi-granularity hierarchical forecasting.

Data includes:
- Hierarchical structure: Country > Shop > Category > Group > SKU > Size
- Temporal features: Week, day of year, day of week
- Promotional features: Discount percentages, holiday indicators
- Demand patterns: Trend, seasonality, noise, promotions
- Generated with realistic parameters for fashion retail
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

HIERARCHY_CONFIG = {
    "countries": ["US", "UK", "DE", "FR", "JP"],
    "shops": ["Downtown", "Mall", "Airport", "Outlet", "Online"],
    "categories": ["Tops", "Bottoms", "Outerwear", "Footwear", "Accessories"],
    "groups": {
        "Tops": ["T-Shirts", "Blouses", "Sweaters", "Hoodies"],
        "Bottoms": ["Jeans", "Leggings", "Shorts", "Skirts"],
        "Outerwear": ["Jackets", "Coats", "Vests"],
        "Footwear": ["Sneakers", "Boots", "Sandals"],
        "Accessories": ["Scarves", "Hats", "Belts"],
    },
    "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
}

PRICE_CONFIG = {
    "Tops": {"T-Shirts": 29.99, "Blouses": 49.99, "Sweaters": 59.99, "Hoodies": 69.99},
    "Bottoms": {"Jeans": 79.99, "Leggings": 49.99, "Shorts": 39.99, "Skirts": 54.99},
    "Outerwear": {"Jackets": 129.99, "Coats": 199.99, "Vests": 89.99},
    "Footwear": {"Sneakers": 119.99, "Boots": 149.99, "Sandals": 79.99},
    "Accessories": {"Scarves": 34.99, "Hats": 24.99, "Belts": 39.99},
}

SIZE_MULTIPLIERS = {"XS": 0.7, "S": 0.9, "M": 1.1, "L": 1.1, "XL": 0.9, "XXL": 0.6}

COUNTRY_PARAMS = {
    "US": {"trend": 1.02, "seasonality_amp": 0.15},
    "UK": {"trend": 1.01, "seasonality_amp": 0.12},
    "DE": {"trend": 1.00, "seasonality_amp": 0.10},
    "FR": {"trend": 1.01, "seasonality_amp": 0.13},
    "JP": {"trend": 1.03, "seasonality_amp": 0.11},
}

HOLIDAYS = [8, 24, 35, 48, 52, 56, 71, 80, 88, 96]
BACK_TO_SCHOOL = [17, 18, 70, 71]
BLACK_FRIDAY = [48, 100]
CHRISTMAS = [50, 51, 52, 4]


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_hierarchical_demand_data(
    n_weeks: int = 104,
    n_skus_per_group: int = 4,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate comprehensive hierarchical demand data for fashion retail.

    Parameters
    ----------
    n_weeks : int
        Number of weeks of historical data (default: 104 = 2 years)
    n_skus_per_group : int
        Number of SKUs per product group (default: 4)
    seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    pd.DataFrame
        DataFrame with all required columns for hierarchical forecasting
    """
    np.random.seed(seed)

    # Unpack configuration
    countries = HIERARCHY_CONFIG["countries"]
    shops = HIERARCHY_CONFIG["shops"]
    categories = HIERARCHY_CONFIG["categories"]
    groups = HIERARCHY_CONFIG["groups"]
    sizes = HIERARCHY_CONFIG["sizes"]

    rows = []

    for week in range(1, n_weeks + 1):
        # Time features
        day_of_year = (week * 7) % 365
        week_of_year = week % 52
        day_of_week = (week * 7) % 7

        # Holiday flags
        is_holiday = 1 if week in HOLIDAYS else 0
        is_back_to_school = 1 if week in BACK_TO_SCHOOL else 0
        is_black_friday = 1 if week in BLACK_FRIDAY else 0
        is_christmas = 1 if week in CHRISTMAS else 0

        # Trend component (compound across countries)
        base_trend = np.prod([COUNTRY_PARAMS[c]["trend"] for c in countries]) ** (week / 52)

        # Seasonality components
        annual_seasonality = np.sin(2 * np.pi * day_of_year / 365)
        semi_annual = np.sin(4 * np.pi * day_of_year / 365)
        overall_seasonality = 1 + 0.15 * annual_seasonality + 0.08 * semi_annual

        # Weekend effect
        weekend_effect = 1.3 if day_of_week in [5, 6] else 1.0

        for country in countries:
            country_trend = COUNTRY_PARAMS[country]["trend"] ** (week / 52)
            country_factor = np.random.uniform(0.8, 1.2)
            country_seasonal_amp = COUNTRY_PARAMS[country]["seasonality_amp"]
            country_seasonality = 1 + country_seasonal_amp * annual_seasonality

            for shop in shops:
                shop_factor = np.random.uniform(0.7, 1.3)

                for category in categories:
                    for product_group in groups[category]:
                        group_factor = np.random.uniform(0.8, 1.2)
                        base_price = PRICE_CONFIG[category][product_group]

                        for sku_idx in range(n_skus_per_group):
                            sku = f"{category[:3].upper()}-{product_group[:3].upper()}-{sku_idx:03d}"

                            for size in sizes:
                                size_multiplier = SIZE_MULTIPLIERS[size]

                                # Base demand (Poisson distributed)
                                base_demand = np.random.poisson(100)

                                # Discount logic
                                if is_black_friday:
                                    discount = np.random.choice([20, 30, 40], p=[0.4, 0.4, 0.2])
                                elif is_holiday:
                                    discount = np.random.choice([10, 15, 20], p=[0.5, 0.3, 0.2])
                                else:
                                    discount = np.random.choice([0, 5, 10], p=[0.6, 0.25, 0.15])

                                # Demand calculation with all factors
                                promo_effect = 1 + (discount / 50)
                                holiday_effect = 1.2 if (is_christmas or is_back_to_school) else 1.0
                                noise = np.random.lognormal(0, 0.2)

                                demand = (
                                    base_demand
                                    * base_trend
                                    * overall_seasonality
                                    * country_seasonality
                                    * country_factor
                                    * shop_factor
                                    * group_factor
                                    * weekend_effect
                                    * size_multiplier
                                    * promo_effect
                                    * holiday_effect
                                    * noise
                                )

                                quantity = max(0, int(demand))
                                revenue = quantity * base_price * (1 - discount / 100)

                                rows.append({
                                    # Hierarchy columns
                                    "country": country,
                                    "shop": shop,
                                    "product_category": category,
                                    "product_group": product_group,
                                    "sku": sku,
                                    "sku_size": size,

                                    # Time columns
                                    "week": week,
                                    "day_of_year": day_of_year,
                                    "week_of_year": week_of_year,
                                    "day_of_week": day_of_week,

                                    # Demand columns
                                    "quantity_sold": quantity,
                                    "revenue": round(revenue, 2),

                                    # Price columns
                                    "base_price": base_price,
                                    "discount_pct": discount,

                                    # Holiday/promo flags
                                    "is_holiday": is_holiday,
                                    "is_back_to_school": is_back_to_school,
                                    "is_black_friday": is_black_friday,
                                    "is_christmas": is_christmas,
                                })

    df = pd.DataFrame(rows)

    # Add derived columns for feature engineering
    df["week_number"] = df["week"]

    return df


def save_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    """Save dataset to CSV."""
    filepath = output_dir / "hierarchical_demand.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    print(f"Total records: {len(df):,}")


def load_dataset(input_dir: Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    filepath = input_dir / "hierarchical_demand.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configuration
    N_WEEKS = 104  # 2 years of data
    N_SKUS_PER_GROUP = 4
    SEED = 42

    # Paths
    DATA_DIR = Path(__file__).parent
    OUTPUT_DIR = DATA_DIR  # Already in data/raw directory

    print("=" * 60)
    print("Generating Hierarchical Demand Dataset")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Weeks: {N_WEEKS}")
    print(f"  - SKUs per group: {N_SKUS_PER_GROUP}")
    print(f"  - Seed: {SEED}")
    print(f"  - Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Generate data
    print("\nGenerating data...")
    df = generate_hierarchical_demand_data(
        n_weeks=N_WEEKS,
        n_skus_per_group=N_SKUS_PER_GROUP,
        seed=SEED
    )

    # Display statistics
    print("\nDataset Statistics:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Unique countries: {df['country'].nunique()}")
    print(f"  - Unique shops: {df['shop'].nunique()}")
    print(f"  - Unique categories: {df['product_category'].nunique()}")
    print(f"  - Unique product groups: {df['product_group'].nunique()}")
    print(f"  - Unique SKUs: {df['sku'].nunique()}")
    print(f"  - Unique sizes: {df['sku_size'].nunique()}")
    print(f"  - Weeks covered: {df['week'].min()} to {df['week'].max()}")
    print(f"  - Total demand: {df['quantity_sold'].sum():,} units")
    print(f"  - Total revenue: ${df['revenue'].sum():,.2f}")

    # Save dataset
    print("\nSaving dataset...")
    save_dataset(df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
