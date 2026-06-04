"""CoolDrinks - Streamlit UI for context-aware beverage recommender.

Features:
- Context selector: weather, time, occasion
- Taste preference sliders: bitterness, sweetness, strength
- Top-K recommendation display with explanations
- Drink details and flavor profile visualization
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.data_layer import load_drink_data, load_interaction_data, generate_drink_catalog, generate_context_scenarios, generate_interaction_logs
from models.recommender import HybridRecommenderEngine


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CoolDrinks",
    page_icon="🥤",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "recommender" not in st.session_state:
        st.session_state.recommender = None
    if "drink_df" not in st.session_state:
        st.session_state.drink_df = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []


def load_or_generate_data():
    """Load or generate data for the recommender."""
    if st.session_state.drink_df is None:
        data_dir = "data/synthetic"

        # Check if data exists, if not generate it
        drinks_path = Path(data_dir) / "drinks_catalog.csv"
        interactions_path = Path(data_dir) / "interaction_logs.csv"

        if drinks_path.exists() and interactions_path.exists():
            st.session_state.drink_df = load_drink_data(data_dir)
            interactions_df = load_interaction_data(data_dir)
        else:
            # Generate synthetic data
            with st.spinner("Generating synthetic data..."):
                st.session_state.drink_df = generate_drink_catalog(n_drinks=120)
                scenarios_df = generate_context_scenarios(n_scenarios=50)
                interactions_df = generate_interaction_logs(
                    n_interactions=10000,
                    n_users=500,
                    drinks_df=st.session_state.drink_df,
                    scenarios_df=scenarios_df
                )
                # Save generated data
                drinks_path.parent.mkdir(parents=True, exist_ok=True)
                st.session_state.drink_df.to_csv(drinks_path, index=False)
                interactions_df.to_csv(interactions_path, index=False)

        # Initialize recommender
        with st.spinner("Training models (SASRec + Fusion + LinUCB)..."):
            st.session_state.recommender = HybridRecommenderEngine(
                st.session_state.drink_df,
                interactions_df
            )


# ============================================================================
# SIDEBAR - CONTEXT & PREFERENCES
# ============================================================================

with st.sidebar:
    st.header("Context & Preferences")

    # User ID
    user_id = st.text_input("User ID", value="U001", help="User identifier for personalized recommendations")

    st.divider()

    # Weather
    st.subheader("Weather")
    weather = st.selectbox(
        "Condition",
        options=["sunny", "rainy", "cloudy", "snowy", "stormy"],
        index=0,
        help="Current weather condition"
    )

    # Temperature slider
    temp_slider = st.slider(
        "Temperature (°C)",
        min_value=-10,
        max_value=40,
        value=22,
        help="Current temperature"
    )

    st.divider()

    # Time period
    st.subheader("Time of Day")
    hour = st.slider(
        "Hour",
        min_value=6,
        max_value=23,
        value=14,
        help="Hour of day (6-23)"
    )

    if hour < 12:
        time_period = "morning"
    elif hour < 18:
        time_period = "afternoon"
    else:
        time_period = "evening"

    st.caption(f"{time_period.capitalize()} ({hour:02d}:00)")

    st.divider()

    # Occasion
    st.subheader("Occasion")
    occasion = st.selectbox(
        "Occasion",
        options=["casual", "celebration", "pairing", "recovery", "social", "business"],
        index=0,
        help="Current occasion or activity"
    )

    st.divider()

    # Taste preferences
    st.subheader("Taste Preferences")

    col1, col2, col3 = st.columns(3)

    with col1:
        bitterness = st.slider(
            "Bitterness",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Preference for bitter flavors (0=none, 1=very bitter)"
        )

    with col2:
        sweetness = st.slider(
            "Sweetness",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Preference for sweet flavors (0=dry, 1=sweet)"
        )

    with col3:
        strength = st.slider(
            "Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Preference for strong drinks (0=mild, 1=strong)"
        )

    st.divider()

    # Top-K
    top_k = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=20,
        value=5,
        help="Number of drinks to recommend"
    )

    # Generate recommendations button
    generate_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🥤 CoolDrinks")
st.caption("Context-aware beverage recommender powered by SOTA ML (SASRec + Fusion + LinUCB)")

# Load data and initialize recommender on first run
if st.session_state.recommender is None:
    load_or_generate_data()

# Generate recommendations on button click or first load
if generate_btn or len(st.session_state.recommendations) == 0:
    with st.spinner("Generating personalized recommendations..."):
        st.session_state.recommendations = st.session_state.recommender.recommend(
            user_id=user_id,
            weather=weather,
            time_period=time_period,
            occasion=occasion,
            bitterness_pref=bitterness,
            sweetness_pref=sweetness,
            strength_pref=strength,
            top_k=top_k
        )

    st.session_state.context = {
        "weather": weather,
        "time_period": time_period,
        "occasion": occasion,
        "bitterness_pref": bitterness,
        "sweetness_pref": sweetness,
        "strength_pref": strength,
    }

# Display summary
if st.session_state.recommendations:
    st.divider()

    st.subheader(f"📊 Recommendations for {user_id}")

    # Context summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Weather", weather.capitalize())
    with col2:
        st.metric("Time", time_period.capitalize())
    with col3:
        st.metric("Occasion", occasion.capitalize())

    st.divider()

    # Display recommendations
    for i, rec in enumerate(st.session_state.recommendations, 1):
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**#{i}. {rec['name']}**")
                st.caption(f"{rec['type'].capitalize()} - {rec['style'].replace('_', ' ').title()}")

                # Flavor profile
                flavor_cols = st.columns(3)
                with flavor_cols[0]:
                    st.markdown(f"**Bitterness**")
                    st.progress(rec['bitterness'] / 100)
                with flavor_cols[1]:
                    st.markdown(f"**Sweetness**")
                    st.progress(rec['sweetness'] / 100)
                with flavor_cols[2]:
                    st.markdown(f"**Carbonation**")
                    st.progress(rec['carbonation'] / 5)

            with col2:
                st.markdown(f"**ABV**: {rec['abv']}%")
                st.markdown(f"**Season**: {rec['seasonality']}")

                # Score breakdown
                st.markdown("**Score Breakdown**")
                st.progress(rec['overall_score'])
                st.caption(f"Total: {rec['overall_score']:.4f}")

            # Explanation
            explanation = st.session_state.recommender.explain_recommendation(rec["drink_id"], st.session_state.context)
            st.caption(f"✨ {explanation}")

            # Component scores
            with st.expander("View component scores"):
                st.write(f"**SASRec (Sequential)**: {rec['sasrec_score']:.4f}")
                st.write(f"**Fusion (Context)**: {rec['fusion_score']:.4f}")
                st.write(f"**LinUCB (Exploration)**: {rec['linucb_score']:.4f}")

    st.divider()

    # Statistics
    with st.expander("View dataset statistics"):
        stats = st.session_state.recommender.get_catalog_stats()
        st.write(f"Total drinks: {stats['total_drinks']}")
        st.write(f"By type: {stats['by_type']}")

        ctx_stats = st.session_state.recommender.get_interaction_stats()
        st.write(f"Interactions: {ctx_stats['total_interactions']}")
        st.write(f"Unique users: {ctx_stats['unique_users']}")

else:
    st.info("Click 'Generate Recommendations' to get personalized drink suggestions!")

# Footer
st.divider()
st.caption(
    "Powered by SASRec (Transformer-based sequential recommendation), "
    "Multi-modal Fusion (context-aware), and LinUCB (exploration-exploitation)."
)
