import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = "data/winequality-red.csv"
# It will check for output folder

OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
RESULT_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model
joblib.dump(model, MODEL_PATH)

# Save results
results = {
    "MSE": mse,
    "R2_Score": r2
}

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

# Print metrics (important for GitHub Actions)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
