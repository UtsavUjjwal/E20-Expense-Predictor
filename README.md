# E20 Mileage Drop & Fuel Cost Estimator

Predicts how much mileage a petrol car in India loses on **E20 (20% ethanol-blended) fuel**, then estimates the extra monthly fuel cost. The analysis uses Indian passenger-car records compiled from ARAI, SIAM, and Autocar India real-world tests.

**Live app:** https://e20predictor.streamlit.app/

**Capstone report (PDF upload):** [E20_Capstone_Project_Report.pdf](E20_Capstone_Project_Report.pdf)

**Model performance (notebook test split):** R² = 0.9986, MAE = 0.028 (Random Forest Regressor)

---

## Project Overview

India’s petrol supply is moving to E20. Ethanol has lower energy density than petrol, so many cars lose km/L and owners spend more at the pump. Compatibility is uneven: certified engines lose less than older uncertified petrol cars. This project inspects that dataset, explains the patterns, and ships a calculator a driver can actually use.

## Problem Statement

People still budget fuel using brochure km/L and do not know the extra rupee cost of E20. There was no simple public tool that combined vehicle specs, tested mileage, compatibility, and a predicted drop into a monthly petrol vs E20 comparison.

## Dataset Description

| File | Role |
| --- | --- |
| `E20_predictions_updated.xlsx` | Master analysis table (221 rows × 13 columns) |
| `car_lookup.csv` | Brand / model / variant lookup for the Streamlit dropdowns |

**Columns:** Brand, Model, Variant, Engine (CC), Mileage Claimed (km/L), Mileage Tested (km/L), Launch Year, Launch Year Note, E20 Compatible, E20_Drop_%, Mileage_E20_Estimated, E20_Drop_%_updated, Mileage_E20_Estimated_updated.

**Coverage:** 14 brands (Maruti, Hyundai, Tata, Toyota, and others), launch years 2008–2026, compatibility labels Yes / No / N/A (CNG) / N/A (Diesel) / TBA / Discontinued.

**Domain:** Automotive + household fuel-cost analytics (finance).

## Tools Used

- Python, pandas, NumPy — inspection, cleaning, statistics
- Matplotlib, Seaborn — EDA charts
- scikit-learn — Random Forest regressor
- Streamlit, joblib — public estimator
- Git & GitHub — version control
- ReportLab — capstone PDF

Power BI (internship Task 6) is **not** included in this submission.

## Data Cleaning Process

1. Inspect shape, dtypes, missingness, and duplicates (`221` rows, `0` full-row duplicates).
2. Drop `Launch Year Note` (empty on 219/221 rows).
3. Trim categorical text (Brand, Model, Variant, E20 Compatible).
4. Do **not** impute engine CC or the drop target — those rows are excluded from modelling/EDA.
5. Keep CNG/diesel in the compatibility mix, but exclude them from petrol-drop analysis (E20 does not apply).
6. Drop duplicate brand–model–variant–engine–year keys if present.
7. Result: a petrol-relevant analysis set used for EDA, charts, and insights.

Missingness in the raw file: Engine CC (4), claimed/tested mileage (13 each), drop % (17), estimated E20 mileage (19).

## EDA (summary)

- Mean claimed mileage ≈ **18.96 km/L** vs mean tested mileage ≈ **15.52 km/L** (real-world gap before E20).
- Mean **updated** E20 drop ≈ **13.1%** (median 11%).
- Compatible petrol cars drop less on average than non-compatible petrol cars.
- Newer (2022+) records are mostly E20-certified; older cars are mostly not.
- Engine sizes cluster around 1.2 L and 1.5 L. A few large-displacement outliers exist; the app warns outside 600–3000 CC.

Full write-up, tables, outlier notes, and correlation discussion are in the PDF.

## Visualizations

Charts are regenerated with `python generate_capstone_report.py` into `reports/figures/`:

1. Records by brand
2. E20 compatibility mix
3. Engine size distribution
4. Coverage by launch year
5. Claimed vs tested mileage
6. Correlation heatmap
7. Drop % by compatibility
8. Launch year vs drop %
9. Average drop by brand
10. Analytics snapshot (KPI stand-in for Power BI)

## Power BI Dashboard

Skipped. Task 6 is omitted. The PDF includes a static KPI snapshot (average drop, compatible share, extra Rs/month, claimed-vs-tested gap) plus supporting charts.

## Key Insights

1. E20 is a real household cost, not noise — about **Rs 1,193/month extra** on average in a 1,200 km, Rs 96.5/L scenario (90th percentile ≈ Rs 1,809).
2. Certification matters: compatible petrol cars average **10.7%** drop vs **16.4%** for non-compatible petrol cars.
3. Brochure mileage overstates real-world km/L; stacking E20 on claimed figures understates fuel need twice.
4. The high-risk pocket is the **older used-car / fleet** segment, not 2022+ certified models.
5. Brand-level average drop is not uniform — useful for fleet shortlists.
6. CNG/diesel are a different policy problem (treated as 0% E20 drop).

## Business Recommendations

1. **Used-car platforms / consumers:** Show E20 compatibility and estimated extra monthly rupees on every petrol listing; default to tested km/L.
2. **Fleets / ride-hailing:** Prefer E20-certified petrol or move high-duty cycles to CNG/hybrid — the upper tail of extra cost compounds across a fleet.
3. **OEMs:** Publish certified E20 km/L for volume 1.2 L / 1.5 L engines.
4. **Fuel retailers / policy comms:** Pair pump labelling with a simple household cost explainer.

## Conclusion

The 221-row Indian car table is small but decision-useful. After cleaning, EDA shows a consistent real-world mileage gap, a typical double-digit E20 drop, and a clear certified vs uncertified split. The Streamlit app turns that into a cost estimate. Next steps would be a larger independent test sample, city-level fuel prices, and (if required later) an interactive Power BI file.

## How It Works (app)

1. Select a car (Brand → Model → Variant) or enter specs manually.
2. The Random Forest predicts E20 mileage drop %.
3. The app estimates E20 km/L and compares monthly fuel cost (petrol vs E20), with an optional age-based wear adjustment on claimed mileage.

## Project Structure

```
├── app.py                            # Streamlit web app
├── train_model.py                    # Model training script
├── EDA.ipynb                         # Original EDA + model notebook
├── generate_capstone_report.py       # Rebuilds charts + PDF
├── E20_Capstone_Project_Report.pdf   # Submission report (Option 3)
├── reports/figures/                  # Chart PNGs used in the PDF
├── car_lookup.csv                    # Dropdown lookup
├── E20_predictions_updated.xlsx      # Cleaned/analysis dataset
├── e20_drop_model.pkl                # Trained Random Forest
├── model_columns.pkl                 # Feature column order
└── requirements.txt
```

## Run Locally

```bash
git clone https://github.com/UtsavUjjwal/E20-Expense-Predictor.git
cd E20-Expense-Predictor
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`. Rebuild the report with:

```bash
python generate_capstone_report.py
```

## Data Sources

- ARAI (Automotive Research Association of India)
- SIAM (Society of Indian Automobile Manufacturers)
- Autocar India real-world mileage tests

## Disclaimer

Estimates are model-based and can differ from real-world results depending on driving conditions, maintenance, traffic, and fuel quality.
