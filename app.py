"""
E20 Mileage Drop & Fuel Cost Predictor
Public-facing Streamlit app.

Run locally with:  streamlit run app.py
Model files (e20_drop_model.pkl, model_columns.pkl) must be in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Load model + car lookup (cached so it loads once, not on every interaction) ----------
@st.cache_resource
def load_model():
    model = joblib.load("e20_drop_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_car_lookup():
    return pd.read_csv("car_lookup.csv")

model, model_columns = load_model()
car_lookup = load_car_lookup()

VALID_LABELS = ['Yes', 'No', 'N/A (CNG)', 'N/A (Diesel)', 'TBA (Upcoming)', 'Discontinued']


# ---------- Core prediction logic (same as notebook) ----------
def predict_e20_drop(engine_cc, launch_year, e20_compatible, brand=None):
    warnings = []

    if engine_cc < 600 or engine_cc > 3000:
        warnings.append(f"Engine CC ({engine_cc}) is outside the typical training range (600-3000cc). Prediction may be unreliable.")

    if launch_year < 2010 or launch_year > 2026:
        warnings.append(f"Launch Year ({launch_year}) is outside the training range (2010-2026). Prediction may be unreliable.")

    if e20_compatible not in VALID_LABELS:
        return {"error": f"Invalid E20 Compatible value '{e20_compatible}'.", "predicted_drop_pct": None}

    if e20_compatible in ['N/A (CNG)', 'N/A (Diesel)']:
        return {
            "predicted_drop_pct": 0.0,
            "note": "E20 blending does not apply to this fuel type (CNG/Diesel). Drop is 0 by definition.",
            "warnings": warnings
        }

    row = pd.DataFrame([{"Engine (CC)": engine_cc, "Launch Year": launch_year}])
    for col in model_columns:
        if col.startswith("E20 Compatible_"):
            label = col.replace("E20 Compatible_", "")
            row[col] = 1 if e20_compatible == label else 0
    row = row.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(row)[0]
    pred = float(np.clip(pred, 0, 15))

    return {
        "brand": brand,
        "predicted_drop_pct": round(pred, 2),
        "warnings": warnings
    }


def apply_age_wear_adjustment(base_mileage, car_age_years, decay_rate_per_year=0.008, max_decay=0.30):
    """Degrades claimed/known mileage based on car age (general wear, separate from E20 effect)."""
    if car_age_years < 0:
        car_age_years = 0
    decay = min(car_age_years * decay_rate_per_year, max_decay)
    return base_mileage * (1 - decay)


def estimate_monthly_cost(mileage_tested, monthly_km, price_per_litre, engine_cc, launch_year, e20_compatible, brand=None):
    if monthly_km <= 0 or price_per_litre <= 0 or mileage_tested <= 0:
        return {"error": "Monthly km, fuel price, and tested mileage must all be positive numbers."}

    drop_result = predict_e20_drop(engine_cc, launch_year, e20_compatible, brand)
    if drop_result.get("error"):
        return drop_result

    drop_pct = drop_result["predicted_drop_pct"]
    mileage_e20 = mileage_tested * (1 - drop_pct / 100)

    cost_petrol = (monthly_km / mileage_tested) * price_per_litre
    cost_e20 = (monthly_km / mileage_e20) * price_per_litre if mileage_e20 > 0 else None

    return {
        "brand": brand,
        "mileage_tested_petrol": round(mileage_tested, 2),
        "predicted_drop_pct": drop_pct,
        "estimated_mileage_e20": round(mileage_e20, 2) if mileage_e20 else None,
        "monthly_cost_petrol_rs": round(cost_petrol, 2),
        "monthly_cost_e20_rs": round(cost_e20, 2) if cost_e20 else None,
        "extra_monthly_cost_rs": round(cost_e20 - cost_petrol, 2) if cost_e20 else None,
        "warnings": drop_result.get("warnings", []),
        "note": drop_result.get("note")
    }


# ---------- UI ----------
st.set_page_config(page_title="E20 Mileage Drop & Fuel Cost Estimator", page_icon="⛽", layout="centered")

st.title(" E20 Mileage Drop & Fuel Cost Estimator")
st.caption("Estimate how much mileage your petrol car loses on E20 (ethanol-blended) fuel, and what it costs you monthly.")

mode = st.radio("How do you want to enter your car details?",
                ["Select my car from the list", "Enter specs manually"], horizontal=True)

default_cc, default_year, default_compat, default_mileage = 1200, 2022, "Yes", 18.5
picked_brand = None

if mode == "Select my car from the list":
    c1, c2, c3 = st.columns(3)
    with c1:
        picked_brand = st.selectbox("Brand", sorted(car_lookup["Brand"].unique()))
    models_for_brand = sorted(car_lookup.loc[car_lookup["Brand"] == picked_brand, "Model"].unique())
    with c2:
        picked_model = st.selectbox("Model", models_for_brand)
    variants_for_model = sorted(car_lookup.loc[
        (car_lookup["Brand"] == picked_brand) & (car_lookup["Model"] == picked_model), "Variant"
    ].unique())
    with c3:
        picked_variant = st.selectbox("Variant", variants_for_model)

    row = car_lookup[
        (car_lookup["Brand"] == picked_brand) &
        (car_lookup["Model"] == picked_model) &
        (car_lookup["Variant"] == picked_variant)
    ].iloc[0]

    default_cc = int(row["Engine (CC)"]) if pd.notna(row["Engine (CC)"]) else default_cc
    default_year = int(row["Launch Year"]) if pd.notna(row["Launch Year"]) else default_year
    default_compat = row["E20 Compatible"] if pd.notna(row["E20 Compatible"]) else default_compat
    default_mileage = float(row["Mileage Tested (km/L)"]) if pd.notna(row["Mileage Tested (km/L)"]) else default_mileage

    st.caption(f"Auto-filled from dataset: {default_cc}cc, launched {default_year}, E20: {default_compat}")

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        engine_cc = st.number_input("Engine size (CC)", min_value=100, max_value=6000, value=default_cc, step=50)
        launch_year = st.number_input("Launch Year", min_value=1990, max_value=2026, value=default_year, step=1)
        car_age_years = st.number_input("Car Age (years, as of today)", min_value=0, max_value=35,
                                         value=max(2026 - default_year, 0), step=1,
                                         help="How old your car currently is. Used to account for general wear-and-tear on mileage, separate from the E20 effect.")

    with col2:
        e20_compatible = st.selectbox("E20 Compatible?", VALID_LABELS,
                                       index=VALID_LABELS.index(default_compat) if default_compat in VALID_LABELS else 0)
        knows_actual_mileage = st.checkbox("I know my car's actual current mileage", value=False)
        if knows_actual_mileage:
            mileage_tested = st.number_input("Your car's actual current mileage (km/L)", min_value=1.0, max_value=50.0, value=default_mileage, step=0.1)
        else:
            claimed_mileage = st.number_input("Claimed/showroom mileage (km/L)", min_value=1.0, max_value=50.0, value=default_mileage, step=0.1)
            mileage_tested = apply_age_wear_adjustment(claimed_mileage, car_age_years)
            st.caption(f"Estimated current mileage after {car_age_years} yrs of wear: **{mileage_tested:.2f} km/L** (heuristic, not ML-learned)")

    monthly_km = st.number_input("Your monthly driving distance (km)", min_value=1, max_value=10000, value=1200, step=50)
    price_per_litre = st.number_input("Current petrol price (Rs/litre)", min_value=1.0, max_value=300.0, value=96.5, step=0.5)

    submitted = st.form_submit_button("Estimate")
    brand = picked_brand

if submitted:
    result = estimate_monthly_cost(
        mileage_tested=mileage_tested,
        monthly_km=monthly_km,
        price_per_litre=price_per_litre,
        engine_cc=engine_cc,
        launch_year=launch_year,
        e20_compatible=e20_compatible,
        brand=brand if brand else None
    )

    if result.get("error"):
        st.error(result["error"])
    else:
        if result.get("note"):
            st.info(result["note"])

        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted E20 Drop", f"{result['predicted_drop_pct']}%")
        c2.metric("Estimated E20 Mileage", f"{result['estimated_mileage_e20']} km/L" if result['estimated_mileage_e20'] else "—")
        c3.metric("Extra Monthly Cost", f"₹{result['extra_monthly_cost_rs']}" if result['extra_monthly_cost_rs'] else "—")

        st.write(f"**Petrol monthly cost:** ₹{result['monthly_cost_petrol_rs']}")
        if result['monthly_cost_e20_rs']:
            st.write(f"**E20 monthly cost:** ₹{result['monthly_cost_e20_rs']}")

        for w in result.get("warnings", []):
            st.warning(w)

st.divider()
st.caption("Estimates based on ARAI, SIAM, and Autocar India real-world test data. Actual mileage may vary by driving conditions, vehicle maintenance, and traffic.")
