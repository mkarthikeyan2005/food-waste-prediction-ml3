# FoodWise AI — Food Waste Prediction & Demand Forecasting

A polished Streamlit + scikit-learn academic prototype for exploring demand prediction, waste estimation, inventory risk, model evaluation, and explainable ML.

## What is included

- **Executive overview:** demand trends, waste totals, waste severity, reasons, and data-quality indicators.
- **7-day scenario forecast:** select business type, food category, purchase quantity, shelf life, storage temperature, promotion, weather, and safety buffer; export the forecast as CSV.
- **Inventory & expiry-risk review:** flags short-shelf-life records with remaining stock. This is explicitly a heuristic, not a batch-level food-safety alert.
- **Model comparison:** Linear Regression, Decision Tree, Random Forest, and Extra Trees compared using a chronological 80/20 split.
- **Waste prediction:** regression for `Waste_Quantity` and classification for `Waste_Level`.
- **Explainable AI:** permutation feature importance on held-out demand records.
- **Data validation & exports:** checks, a downloadable run report, full data export, and CSV upload support.
- **Offline fallback:** if the public dataset URL is unavailable, the app uses the included `demo_food_waste_dataset.csv` so the dashboard can still open. The app labels which data source is active.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Optional model evaluation from the public source:

```bash
python train_model.py
```

## Deploy on Streamlit Community Cloud

1. Upload **all files in this folder** to the root of your GitHub repository (do not upload only the ZIP file).
2. Commit the changes to your `main` branch.
3. In Streamlit Community Cloud, choose your repository and set **Main file path** to `app.py`.
4. Deploy. If the app was already deployed, committing the updated files should trigger a rebuild; check **Manage app → Logs** if it fails.

## Data

The app attempts to download the CSV from:

`https://raw.githubusercontent.com/kalyanipatange/Waste-Food-Management-Analysis-/main/food_waste_dataset.csv`

Expected target columns: `Date`, `Daily_Demand`, `Waste_Quantity`, and `Waste_Level`. Context fields such as `Business_Type`, `Food_Category`, `Purchase_Quantity`, `Units_Sold`, `Remaining_Stock`, `Shelf_Life_Days`, `Storage_Temperature`, `Promotion`, `Weather`, and `Waste_Reason` are used when available. The app validates required columns and provides a readable error if an uploaded file is incompatible.

## Important research limitations

- The public source describes its dataset as **synthetic**. The included offline fallback is also synthetic. Metrics are prototype demonstrations, not evidence of real-world performance.
- The 7-day page is a **scenario forecast**: user-selected context is held constant while the date-derived features change. It is not a production-grade autoregressive time-series system.
- The planning quantity is predicted demand plus a user-selected buffer, not an optimized inventory order policy.
- The expiry-risk view is a heuristic. True expiry alerts need item/batch IDs, received dates, actual expiry dates, and current on-hand quantity.
- Feature importance describes model behavior; it does not prove causation.
- For a real deployment, validate on permissioned operational data, verify units, check leakage, and evaluate on a later untouched period or a separate business.

## Repository files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `train_model.py` — optional offline model comparison
- `demo_food_waste_dataset.csv` — offline fallback dataset, clearly synthetic
- `.gitignore` — ignores local Python and environment files
