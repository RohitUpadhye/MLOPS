import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
df = pd.read_csv("data/data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

print("✅ Model trained and saved!")