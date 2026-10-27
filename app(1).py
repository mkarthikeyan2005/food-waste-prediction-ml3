import io
import json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance

DATA_URL = "https://raw.githubusercontent.com/kalyanipatange/Waste-Food-Management-Analysis-/main/food_waste_dataset.csv"
APP_TITLE = "FoodWise AI"

st.set_page_config(page_title="FoodWise AI | Food Intelligence", page_icon="🥗", layout="wide")

st.markdown("""
<style>
.stApp {background:#f4f7fb}
[data-testid="stSidebar"] {background:#10233f}
[data-testid="stSidebar"] * {color:#f5f8fc}
.hero {padding:1.65rem 1.8rem;border-radius:18px;color:white;background:linear-gradient(115deg,#10233f,#17506a 62%,#168477);margin-bottom:1rem}
.hero h1 {color:white;margin:0;font-size:2.25rem}
.hero p {color:#dcebf1;margin:.45rem 0 0 0}
.metric {background:white;border:1px solid #e2e9f2;border-radius:14px;padding:1rem 1.1rem;min-height:110px;box-shadow:0 2px 10px #10233f0a}
.metric-label {font-size:.82rem;color:#64748b}
.metric-value {font-size:1.65rem;font-weight:750;color:#10233f;margin:.25rem 0}
.metric-note {font-size:.75rem;color:#7c8ba0}
.panel {background:white;border:1px solid #e2e9f2;border-radius:14px;padding:1rem 1.1rem}
.good {background:#eaf7f1;border:1px solid #cce9dd;padding:.9rem 1rem;border-radius:12px;color:#175b47}
.caution {background:#fff6e8;border:1px solid #f1dfba;padding:.9rem 1rem;border-radius:12px;color:#79520b}
div[data-testid="stTabs"] button {font-weight:650}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_source_data():
    """Try the published source first; fall back to bundled demo data if offline."""
    try:
        r = requests.get(DATA_URL, timeout=12)
        r.raise_for_status()
        data = pd.read_csv(io.BytesIO(r.content))
        if data.empty:
            raise ValueError("The source dataset is empty.")
        return data, "Published source dataset"
    except Exception as source_error:
        demo_path = Path(__file__).with_name("demo_food_waste_dataset.csv")
        if demo_path.exists():
            return pd.read_csv(demo_path), f"Bundled demo dataset (source unavailable: {type(source_error).__name__})"
        raise RuntimeError(f"Could not download source data and no demo fallback was found: {source_error}") from source_error

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def nfmt(v, digits=1):
    try:
        if pd.isna(v): return "—"
        return f"{float(v):,.{digits}f}"
    except Exception:
        return "—"

def clean_data(raw):
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("The uploaded dataset is empty or could not be read.")
    df = raw.copy()
    required = ["Date", "Daily_Demand", "Waste_Quantity", "Waste_Level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid dates were found. Check the Date column format.")
    for c in ["Daily_Demand", "Waste_Quantity", "Purchase_Quantity", "Shelf_Life_Days", "Storage_Temperature", "Remaining_Stock", "Units_Sold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    return df

CAT_CANDIDATES = ["Business_Type", "Food_Category", "Promotion", "Weather"]
NUM_CANDIDATES = ["Purchase_Quantity", "Shelf_Life_Days", "Storage_Temperature", "DayOfWeek", "Month", "DayOfYear", "IsWeekend"]

def feature_columns(df):
    cats = [c for c in CAT_CANDIDATES if c in df.columns]
    nums = [c for c in NUM_CANDIDATES if c in df.columns]
    return cats, nums, cats + nums

def make_preprocessor(cats, nums, scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), nums),
        ("categorical", cat_pipe, cats),
    ], remainder="drop")

def pipeline_for(model, cats, nums, scale=False):
    return Pipeline([("preprocess", make_preprocessor(cats, nums, scale)), ("model", model)])

def train_everything(df):
    if len(df) < 10:
        raise ValueError(f"At least 10 valid dated rows are required; found {len(df)}.")
    for target in ["Daily_Demand", "Waste_Quantity", "Waste_Level"]:
        if target not in df.columns:
            raise ValueError(f"Required target column missing: {target}")
    df = df.copy()
    df["Daily_Demand"] = pd.to_numeric(df["Daily_Demand"], errors="coerce")
    df["Waste_Quantity"] = pd.to_numeric(df["Waste_Quantity"], errors="coerce")
    df = df.dropna(subset=["Daily_Demand", "Waste_Quantity", "Waste_Level"]).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Fewer than 10 rows have valid demand, waste, and waste-level targets.")
    cats, nums, features = feature_columns(df)
    if not features:
        # A date-only baseline is safer than failing when optional context columns are absent.
        nums = [c for c in ["DayOfWeek", "Month", "DayOfYear", "IsWeekend"] if c in df.columns]
        features = nums
    if not features:
        raise ValueError("No usable feature columns found.")
    # Chronological split. Holdout is the latest 20% of rows.
    split = max(1, min(len(df)-1, int(len(df)*0.8)))
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    results, demand_models = [], {}
    demand_train = train.dropna(subset=["Daily_Demand"])
    demand_test = test.dropna(subset=["Daily_Demand"])
    if demand_train.empty:
        raise ValueError("No valid Daily_Demand values are available in the training period.")
    candidates = [
        ("Linear Regression", LinearRegression(), True),
        ("Decision Tree", DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42), False),
        ("Random Forest", RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1), False),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1), False),
    ]
    for name, estimator, scale in candidates:
        model = pipeline_for(estimator, cats, nums, scale)
        model.fit(demand_train[features], demand_train["Daily_Demand"])
        demand_models[name] = model
        if len(demand_test):
            pred = np.maximum(0, model.predict(demand_test[features]))
            results.append({
                "Model": name,
                "MAE": mean_absolute_error(demand_test["Daily_Demand"], pred),
                "RMSE": np.sqrt(mean_squared_error(demand_test["Daily_Demand"], pred)),
                "R²": r2_score(demand_test["Daily_Demand"], pred) if len(demand_test) > 1 else np.nan,
            })
    best_name = min(results, key=lambda x: x["MAE"])["Model"] if results else "Random Forest"
    best_demand = demand_models[best_name]

    waste_train = train.dropna(subset=["Waste_Quantity"])
    waste_model = pipeline_for(RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1), cats, nums)
    waste_model.fit(waste_train[features], waste_train["Waste_Quantity"])

    risk_train = train.dropna(subset=["Waste_Level"])
    if risk_train["Waste_Level"].astype(str).nunique() < 2:
        # Use a stable fallback class when the training slice has only one class.
        from sklearn.dummy import DummyClassifier
        risk_estimator = DummyClassifier(strategy="most_frequent")
    else:
        risk_estimator = RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    risk_model = pipeline_for(risk_estimator, cats, nums)
    risk_model.fit(risk_train[features], risk_train["Waste_Level"].astype(str))
    risk_test = test.dropna(subset=["Waste_Level"])
    risk_metrics = {}
    if len(risk_test):
        risk_pred = risk_model.predict(risk_test[features])
        risk_metrics = {
            "Accuracy": accuracy_score(risk_test["Waste_Level"].astype(str), risk_pred),
            "Macro F1": f1_score(risk_test["Waste_Level"].astype(str), risk_pred, average="macro", zero_division=0),
        }
    waste_metrics = {}
    waste_test = test.dropna(subset=["Waste_Quantity"])
    if len(waste_test):
        wp = np.maximum(0, waste_model.predict(waste_test[features]))
        waste_metrics = {
            "MAE": mean_absolute_error(waste_test["Waste_Quantity"], wp),
            "RMSE": np.sqrt(mean_squared_error(waste_test["Waste_Quantity"], wp)),
            "R²": r2_score(waste_test["Waste_Quantity"], wp) if len(waste_test) > 1 else np.nan,
        }
    return {
        "cats": cats, "nums": nums, "features": features, "train": train, "test": test,
        "demand_models": demand_models, "best_name": best_name, "demand_model": best_demand,
        "waste_model": waste_model, "risk_model": risk_model, "comparison": pd.DataFrame(results),
        "risk_metrics": risk_metrics, "waste_metrics": waste_metrics
    }

def make_prediction_row(features, business, food, purchase, shelf, temp, promotion, weather, date):
    dt = pd.Timestamp(date)
    values = {
        "Business_Type": business, "Food_Category": food, "Purchase_Quantity": purchase,
        "Shelf_Life_Days": shelf, "Storage_Temperature": temp, "Promotion": promotion,
        "Weather": weather, "DayOfWeek": dt.dayofweek, "Month": dt.month,
        "DayOfYear": dt.dayofyear, "IsWeekend": int(dt.dayofweek >= 5),
    }
    return pd.DataFrame([{f: values.get(f, np.nan) for f in features}], columns=features)

def forecast_seven_days(model, features, business, food, purchase, shelf, temp, promotion, weather, start_date, buffer_pct):
    rows = []
    for i in range(7):
        d = pd.Timestamp(start_date) + timedelta(days=i)
        x = make_prediction_row(features, business, food, purchase, shelf, temp, promotion, weather, d)
        demand = max(0, float(model.predict(x)[0]))
        rows.append({"Date": d.date().isoformat(), "Day": d.strftime("%A"), "Predicted demand": demand,
                     "Planning quantity": demand * (1 + buffer_pct / 100)})
    return pd.DataFrame(rows)

# Sidebar / data
data_source_label = ""

with st.sidebar:
    st.markdown("## 🥗 FoodWise AI")
    st.caption("Food demand & waste intelligence")
    page = st.radio("WORKSPACE", [
        "Executive overview", "7-day forecast", "Inventory & expiry risk",
        "Model comparison", "Explainable AI", "Data validation & reports"
    ])
    st.markdown("---")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], help="Use the source dataset or a compatible dataset with the documented columns.")
    st.markdown("---")
    st.caption("Academic prototype · Synthetic-data source")

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        data_source_label = "Uploaded CSV"
    else:
        raw, data_source_label = fetch_source_data()
    df = clean_data(raw)
    bundle = train_everything(df)
except Exception as e:
    st.markdown('<div class="hero"><h1>🥗 FoodWise AI</h1><p>Predict demand. Reduce avoidable waste. Plan inventory with evidence.</p></div>', unsafe_allow_html=True)
    st.error(f"Unable to load/train the dataset: {e}")
    st.info("Upload the repository file `food_waste_dataset.csv`. Required target columns: Date, Daily_Demand, Waste_Quantity, Waste_Level.")
    st.markdown(f"[Open source dataset]({DATA_URL})")
    st.stop()

features, cats, nums = bundle["features"], bundle["cats"], bundle["nums"]
st.markdown('<div class="hero"><h1>🥗 FoodWise AI</h1><p>Predict demand · Monitor waste risk · Plan inventory more intelligently</p></div>', unsafe_allow_html=True)
st.caption(f"Data source: {data_source_label} · Data range: {df.Date.min().date()} to {df.Date.max().date()} · {len(df):,} records · Best demand model by holdout MAE: {bundle['best_name']}")

if page == "Executive overview":
    st.subheader("Executive overview")
    total_waste = df["Waste_Quantity"].sum()
    waste_ratio = 100 * total_waste / max(1, df["Purchase_Quantity"].sum()) if "Purchase_Quantity" in df else np.nan
    avg_demand = df["Daily_Demand"].mean()
    high_risk = int((df["Waste_Level"].astype(str).str.lower() == "high").sum())
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Records analysed", f"{len(df):,}", "Rows in loaded dataset")
    with c2: metric_card("Average daily demand", nfmt(avg_demand), "Dataset units")
    with c3: metric_card("Recorded waste", nfmt(total_waste), "Dataset units; source unit not specified")
    with c4: metric_card("High-waste records", f"{high_risk:,}", f"{100*high_risk/max(1,len(df)):.1f}% of records")
    left,right = st.columns([1.2,1])
    with left:
        st.markdown("#### Daily demand pattern")
        daily = df.groupby(df["Date"].dt.date)["Daily_Demand"].mean()
        st.line_chart(daily, use_container_width=True)
    with right:
        st.markdown("#### Waste by category")
        if "Food_Category" in df:
            st.bar_chart(df.groupby("Food_Category")["Waste_Quantity"].sum().sort_values(ascending=False), use_container_width=True)
    a,b = st.columns(2)
    with a:
        st.markdown("#### Waste severity")
        st.bar_chart(df["Waste_Level"].astype(str).value_counts(), use_container_width=True)
    with b:
        st.markdown("#### Waste reasons")
        if "Waste_Reason" in df:
            st.bar_chart(df["Waste_Reason"].astype(str).value_counts(), use_container_width=True)
        st.dataframe(pd.DataFrame({"Data check":["Missing cells","Duplicate rows","Date range"],
            "Result":[f"{df.isna().sum().sum():,}", f"{df.duplicated().sum():,}", f"{df.Date.min().date()} → {df.Date.max().date()}"]}), hide_index=True, use_container_width=True)
    st.markdown('<div class="caution"><b>Research limitation:</b> The linked dataset is synthetic. This dashboard demonstrates methods and does not establish real-world waste reduction.</div>', unsafe_allow_html=True)

elif page == "7-day forecast":
    st.subheader("7-day demand forecast")
    st.write("Set the operating context below. The selected demand model generates seven daily estimates.")
    businesses = sorted(df["Business_Type"].dropna().astype(str).unique()) if "Business_Type" in df else ["Restaurant"]
    foods = sorted(df["Food_Category"].dropna().astype(str).unique()) if "Food_Category" in df else ["Prepared Food"]
    weathers = sorted(df["Weather"].dropna().astype(str).unique()) if "Weather" in df else ["Sunny"]
    promos = sorted(df["Promotion"].dropna().astype(str).unique()) if "Promotion" in df else ["No","Yes"]
    c1,c2,c3 = st.columns(3)
    with c1:
        business = st.selectbox("Business type", businesses)
        food = st.selectbox("Food category", foods)
        start = st.date_input("Forecast start date", value=pd.Timestamp.today().date())
    with c2:
        purchase = st.number_input("Planned purchase quantity", min_value=0.0, value=100.0, step=5.0)
        shelf = st.number_input("Shelf life (days)", min_value=1.0, value=3.0, step=1.0)
        temp = st.number_input("Storage temperature (°C)", value=5.0, step=1.0)
    with c3:
        promotion = st.selectbox("Promotion", promos)
        weather = st.selectbox("Weather assumption (held constant)", weathers)
        buffer_pct = st.slider("Safety buffer", 0, 30, 10, 5)
    forecast = forecast_seven_days(bundle["demand_model"], features, business, food, purchase, shelf, temp, promotion, weather, start, buffer_pct)
    total = forecast["Predicted demand"].sum()
    planned = forecast["Planning quantity"].sum()
    x1,x2,x3 = st.columns(3)
    with x1: metric_card("7-day predicted demand", nfmt(total), "Sum of daily predictions")
    with x2: metric_card("Planning quantity", nfmt(planned), f"Includes {buffer_pct}% buffer")
    with x3: metric_card("Forecast horizon", "7 days", "Daily estimates")
    st.line_chart(forecast.set_index("Date")[["Predicted demand","Planning quantity"]], use_container_width=True)
    st.dataframe(forecast, hide_index=True, use_container_width=True)
    st.download_button("Download 7-day forecast CSV", forecast.to_csv(index=False).encode(), "foodwise_7_day_forecast.csv", "text/csv")
    st.caption("Forecast conditions (weather, promotion, purchase quantity, storage) are user assumptions. Model estimates are not guarantees; verify against recent actual sales.")

elif page == "Inventory & expiry risk":
    st.subheader("Inventory and expiry-risk monitor")
    st.write("This view flags records with short shelf life and large remaining stock. It is a **risk heuristic**, not an item-level expiry alert.")
    if "Remaining_Stock" in df:
        stock = pd.to_numeric(df["Remaining_Stock"], errors="coerce")
    elif "Purchase_Quantity" in df and "Units_Sold" in df:
        stock = pd.to_numeric(df["Purchase_Quantity"], errors="coerce") - pd.to_numeric(df["Units_Sold"], errors="coerce")
    else:
        stock = pd.Series(np.nan, index=df.index)
    view = df.copy()
    view["Remaining stock (derived)"] = stock
    view["Shelf life (days)"] = pd.to_numeric(view["Shelf_Life_Days"], errors="coerce") if "Shelf_Life_Days" in view.columns else np.nan
    valid = view.dropna(subset=["Remaining stock (derived)","Shelf life (days)"]).copy()
    short_days = st.slider("Short shelf-life threshold (days)", 1, 7, 3)
    min_stock = st.number_input("Minimum remaining stock to flag", min_value=0, value=5, step=1)
    flagged = valid[(valid["Shelf life (days)"] <= short_days) & (valid["Remaining stock (derived)"] >= min_stock)].copy()
    flagged["Risk reason"] = "Short shelf life + remaining stock"
    k1,k2 = st.columns(2)
    with k1: metric_card("Records flagged", f"{len(flagged):,}", "Heuristic review queue")
    with k2: metric_card("Short-life stock units", nfmt(flagged["Remaining stock (derived)"].sum()), "Units; source unit not specified")
    if len(flagged):
        cols = [c for c in ["Date","Business_Type","Food_Category","Shelf_Life_Days","Remaining stock (derived)","Waste_Level","Waste_Reason","Risk reason"] if c in flagged.columns]
        st.dataframe(flagged[cols].sort_values("Shelf_Life_Days"), use_container_width=True, hide_index=True)
        st.download_button("Download risk review CSV", flagged[cols].to_csv(index=False).encode(), "foodwise_inventory_risk.csv", "text/csv")
    else:
        st.info("No records meet the current thresholds, or the uploaded data lacks stock/shelf-life fields.")
    st.markdown('<div class="caution"><b>For true expiry alerts:</b> add batch-level item ID, received date, expiry date, current on-hand quantity, and food-safety status. The current dataset has shelf-life duration but not each batch’s actual expiry date.</div>', unsafe_allow_html=True)

elif page == "Model comparison":
    st.subheader("Demand model comparison")
    st.write("Models are compared on the latest 20% of chronologically sorted records. The earlier 80% is used for training.")
    comp = bundle["comparison"].copy()
    st.dataframe(comp.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.3f}"}), use_container_width=True, hide_index=True)
    if len(comp):
        st.markdown("#### Error comparison (lower is better)")
        st.bar_chart(comp.set_index("Model")[["MAE","RMSE"]], use_container_width=True)
        best = comp.sort_values("MAE").iloc[0]
        st.markdown(f'<div class="good"><b>Selected model:</b> {best["Model"]} based on the lowest holdout MAE ({best["MAE"]:.3f}).</div>', unsafe_allow_html=True)
    st.markdown("#### Additional model checks")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Waste quantity regression**")
        if bundle["waste_metrics"]:
            st.json({k:(None if pd.isna(v) else round(float(v),4)) for k,v in bundle["waste_metrics"].items()})
        else: st.info("Not enough holdout rows for waste metrics.")
    with c2:
        st.markdown("**Waste severity classification**")
        if bundle["risk_metrics"]:
            st.json({k:round(float(v),4) for k,v in bundle["risk_metrics"].items()})
        else: st.info("Not enough holdout rows for classification metrics.")
    st.markdown('<div class="caution"><b>Interpret carefully:</b> these scores describe a synthetic dataset and one chronological holdout. For stronger validation, test on a later period or a separate real-world dataset.</div>', unsafe_allow_html=True)

elif page == "Explainable AI":
    st.subheader("Explainable AI")
    st.write("Feature importance is estimated with permutation importance on the held-out demand records. It shows which inputs matter most to this fitted model, not causation.")
    test = bundle["test"].dropna(subset=["Daily_Demand"]).copy()
    if len(test) >= 10:
        sample = test.sample(min(300, len(test)), random_state=42)
        with st.spinner("Calculating permutation importance..."):
            try:
                perm = permutation_importance(bundle["demand_model"], sample[features], sample["Daily_Demand"],
                    n_repeats=3, random_state=42, scoring="neg_mean_absolute_error", n_jobs=-1)
                imp = pd.DataFrame({"Feature":features, "Importance (MAE increase)":perm.importances_mean})
                imp = imp.sort_values("Importance (MAE increase)", ascending=False)
                st.bar_chart(imp.head(12).set_index("Feature"), use_container_width=True)
                st.dataframe(imp, hide_index=True, use_container_width=True)
                st.download_button("Download feature importance CSV", imp.to_csv(index=False).encode(), "foodwise_feature_importance.csv", "text/csv")
            except Exception as ex:
                st.warning(f"Importance calculation could not complete: {ex}")
    else:
        st.info("At least 10 held-out target rows are needed for feature importance.")
    st.markdown("#### How to interpret this")
    st.write("- Positive importance means shuffling that feature worsened prediction error in this test sample.")
    st.write("- Correlated features can share or distort importance.")
    st.write("- Importance is not proof that changing a feature will change demand.")
    st.write("- Date-derived features are calendar signals; they do not guarantee seasonal behavior will persist.")

elif page == "Data validation & reports":
    st.subheader("Data validation and exportable report")
    checks = pd.DataFrame([
        {"Check":"Rows", "Result":f"{len(df):,}", "Status":"Info"},
        {"Check":"Duplicate rows", "Result":f"{df.duplicated().sum():,}", "Status":"Review" if df.duplicated().any() else "Pass"},
        {"Check":"Missing cells", "Result":f"{df.isna().sum().sum():,}", "Status":"Review" if df.isna().sum().sum() else "Pass"},
        {"Check":"Date coverage", "Result":f"{df.Date.min().date()} to {df.Date.max().date()}", "Status":"Info"},
        {"Check":"Negative demand values", "Result":str(int((df["Daily_Demand"]<0).sum())), "Status":"Review" if (df["Daily_Demand"]<0).any() else "Pass"},
        {"Check":"Negative waste values", "Result":str(int((df["Waste_Quantity"]<0).sum())), "Status":"Review" if (df["Waste_Quantity"]<0).any() else "Pass"},
    ])
    st.dataframe(checks, hide_index=True, use_container_width=True)
    st.markdown("#### Generate project summary")
    comp = bundle["comparison"]
    best_score = comp.sort_values("MAE").iloc[0] if len(comp) else None
    report = [
        "# FoodWise AI — Run Summary", "",
        f"- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Records analysed: {len(df)}",
        f"- Date range: {df.Date.min().date()} to {df.Date.max().date()}",
        f"- Data source: {data_source_label}",
        "- Dataset provenance: public repository dataset documented as synthetic; bundled demo is synthetic as well.",
        "- Evaluation method: chronological 80/20 train/test split.",
        f"- Selected demand model: {bundle['best_name']}",
    ]
    if best_score is not None:
        report += [f"- Demand MAE: {best_score['MAE']:.4f}", f"- Demand RMSE: {best_score['RMSE']:.4f}", f"- Demand R²: {best_score['R²']:.4f}"]
    if bundle["waste_metrics"]:
        report += ["", "## Waste regression metrics"] + [f"- {k}: {v:.4f}" for k,v in bundle["waste_metrics"].items() if pd.notna(v)]
    if bundle["risk_metrics"]:
        report += ["", "## Waste risk classification metrics"] + [f"- {k}: {v:.4f}" for k,v in bundle["risk_metrics"].items()]
    report += ["", "## Limitations", "- Synthetic data; results are not real-world validation.",
               "- Forecast depends on user-entered assumptions and available features.",
               "- Expiry-risk view is heuristic; item-level expiry requires batch expiry dates.",
               "- Predictions support planning and should be reviewed by a human."]
    report_text = "\n".join(report)
    st.text_area("Report preview", report_text, height=300)
    st.download_button("Download project report (Markdown)", report_text, "foodwise_run_report.md", "text/markdown")
    st.download_button("Download full loaded dataset CSV", df.to_csv(index=False).encode(), "foodwise_loaded_dataset.csv", "text/csv")
    st.markdown("#### Upload validation guidance")
    st.write("For external operational data, match the source columns, verify units and date semantics, remove leakage fields from pre-decision prediction tasks, and keep a separate later-period test set.")

st.markdown("---")
st.caption("FoodWise AI · Academic prototype · Source dataset: kalyanipatange/Waste-Food-Management-Analysis- · Synthetic data is not real-world evidence.")
