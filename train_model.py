import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("region10_water_quality_2021.csv")

# Fill missing values
df["pH"].fillna(df["pH"].mean(), inplace=True)
df["FecalColiform"].fillna(df["FecalColiform"].mean(), inplace=True)

X = df[["pH", "FecalColiform"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "water_quality_model.pkl")

print("Model training complete. Saved as water_quality_model.pkl")
