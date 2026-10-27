"""
Optional offline evaluation script for the FoodWise AI source dataset.
Run: python train_model.py
"""
import io
import requests
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

URL = "https://raw.githubusercontent.com/kalyanipatange/Waste-Food-Management-Analysis-/main/food_waste_dataset.csv"
df = pd.read_csv(io.BytesIO(requests.get(URL, timeout=30).content))
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
df["DayOfWeek"] = df.Date.dt.dayofweek
df["Month"] = df.Date.dt.month
df["DayOfYear"] = df.Date.dt.dayofyear
df["IsWeekend"] = (df.Date.dt.dayofweek >= 5).astype(int)
cats = [c for c in ["Business_Type","Food_Category","Promotion","Weather"] if c in df]
nums = [c for c in ["Purchase_Quantity","Shelf_Life_Days","Storage_Temperature","DayOfWeek","Month","DayOfYear","IsWeekend"] if c in df]
for c in cats: df[c] = df[c].astype("string").fillna("Unknown")
for c in nums: df[c] = pd.to_numeric(df[c], errors="coerce")
features = cats + nums
split = int(len(df)*0.8)
train, test = df.iloc[:split], df.iloc[split:]
models = [
    ("Linear Regression", LinearRegression(), True),
    ("Decision Tree", DecisionTreeRegressor(max_depth=8,min_samples_leaf=5,random_state=42), False),
    ("Random Forest", RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=-1), False),
    ("Extra Trees", ExtraTreesRegressor(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=-1), False),
]
rows = []
for name, estimator, scale in models:
    num_steps = [("imputer",SimpleImputer(strategy="median"))]
    if scale: num_steps.append(("scale",StandardScaler()))
    prep = ColumnTransformer([
        ("num",Pipeline(num_steps),nums),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats)
    ])
    model = Pipeline([("prep",prep),("model",estimator)])
    tr = train.dropna(subset=["Daily_Demand"])
    te = test.dropna(subset=["Daily_Demand"])
    model.fit(tr[features],tr["Daily_Demand"])
    pred = np.maximum(0,model.predict(te[features]))
    rows.append({"Model":name,"MAE":mean_absolute_error(te["Daily_Demand"],pred),
                 "RMSE":np.sqrt(mean_squared_error(te["Daily_Demand"],pred)),
                 "R2":r2_score(te["Daily_Demand"],pred)})
out = pd.DataFrame(rows).sort_values("MAE")
print(out.to_string(index=False))
out.to_csv("model_comparison_results.csv",index=False)
print("\\nSaved model_comparison_results.csv")
