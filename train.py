import json
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = "data/winequality-red.csv"
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
RESULT_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=";")

# ===== FEATURE SELECTION =====
# ---- OPTION 1: ALL FEATURES ----
X = df.drop("quality", axis=1)

# ---- OPTION 2: CORRELATION-BASED FEATURES (ENABLE FOR EXP-02 / EXP-04) ----
corr = df.corr()["quality"].abs()
selected_features = corr[corr > 0.2].index.drop("quality")
X = df[selected_features]

y = df["quality"]

# ===== TRAIN-TEST SPLIT =====
# ---- OPTION A: 80/20 SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- OPTION B: 70/30 SPLIT (ENABLE FOR EXP-04) ----
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# ===== PREPROCESSING =====
# ---- OPTION 1: STANDARDIZATION (FOR LINEAR MODELS) ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- OPTION 2: NO SCALING (FOR TREE MODELS) ----
# (Comment out StandardScaler block above for RandomForest)

# ===== MODEL SELECTION =====
# ---- EXP-01: LINEAR REGRESSION ----
# model = LinearRegression()

# ---- EXP-02: RIDGE REGRESSION ----
model = Ridge(alpha=1.0)

# ---- EXP-03 / EXP-04: RANDOM FOREST ----
# model = RandomForestRegressor(
#     n_estimators=50,
#     max_depth=10,
#     random_state=42
# )

# ---- EXP-04 VARIANT ----
# model = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=15,
#     random_state=42
# )

# ===== TRAIN MODEL =====
model.fit(X_train, y_train)

# ===== PREDICT & EVALUATE =====
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ===== SAVE OUTPUTS =====
joblib.dump(model, MODEL_PATH)

results = {
    "MSE": mse,
    "R2_Score": r2
}

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

# ===== PRINT METRICS (CI NEEDS THIS) =====
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
