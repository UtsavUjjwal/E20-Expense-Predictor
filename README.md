# E20 Mileage Drop & Fuel Cost Estimator

Predicts how much mileage a petrol car in India loses when using E20 (20% ethanol-blended) fuel, and estimates the resulting monthly fuel cost — based on ARAI, SIAM, and Autocar India real-world test data.

**Model performance:** R² = 0.9986, MAE = 0.028 (Random Forest Regressor)

## Live App
https://e20predictor.streamlit.app/

## Project Structure
```
├── app.py                     # Streamlit web app (main entry point)
├── car_lookup.csv             # Brand/Model/Variant lookup for dropdown auto-fill
├── e20_drop_model.pkl         # Trained Random Forest model
├── model_columns.pkl          # Feature column order used at training time
├── requirements.txt           # Python dependencies
├── E20_Model_Pipeline.ipynb   # Full pipeline notebook (EDA, training, evaluation)
└── E20_predictions_final.xlsx # Cleaned dataset used for training
```

## How It Works
1. User selects their car (Brand → Model → Variant) or enters specs manually
2. App predicts E20 mileage drop % using the trained Random Forest model
3. App calculates estimated E20 mileage and compares monthly fuel cost (regular petrol vs E20)

## Run Locally
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```
App opens at `http://localhost:8501`

## Data Sources
- ARAI (Automotive Research Association of India)
- SIAM (Society of Indian Automobile Manufacturers)
- Autocar India real-world mileage tests

## Disclaimer
Estimates are model-based and may vary from real-world results depending on driving conditions, vehicle maintenance, traffic, and fuel quality.
