import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------
df = pd.read_csv("region10_water_quality_2021.csv")

# Handle missing values (pH sometimes missing in raw EMB data)
df["pH"].fillna(df["pH"].mean(), inplace=True)
df["FecalColiform"].fillna(df["FecalColiform"].mean(), inplace=True)
df["DO"].fillna(df["DO"].mean(), inplace=True)
df["BOD"].fillna(df["BOD"].mean(), inplace=True)
df["Turbidity"].fillna(df["Turbidity"].mean(), inplace=True)
df["Temp"].fillna(df["Temp"].mean(), inplace=True)

# ------------------------------------------------------
# 2. SELECT FEATURES & LABEL
# ------------------------------------------------------
X = df[["pH", "FecalColiform", "DO", "BOD", "Turbidity", "Temp"]]
y = df["Label"]

# ------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------
# 4. TRAIN RANDOM FOREST MODEL
# ------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------------------
# 5. EVALUATE MODEL PERFORMANCE
# ------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=====================================")
print(" WATER QUALITY MODEL PERFORMANCE")
print("=====================================")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------
# 6. SAVE TRAINED MODEL
# ------------------------------------------------------
joblib.dump(model, "water_quality_model.pkl")
print("\nModel saved successfully as water_quality_model.pkl")
