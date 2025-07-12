import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import has_max_64_atoms

if not os.path.exists("results"):
    os.makedirs("results")

parser = ArgumentParser()
parser.add_argument(
    "--embeddings",
    type=str,
    required=True,
    help="Path to the generated embeddings file",
)
parser.add_argument(
    "--data", type=str, required=True, help="Path to the original data file for labels"
)
args = parser.parse_args()
embedding_file = args.embeddings
data_file = args.data

results_df = pd.read_pickle(embedding_file)
print(f"Loaded {len(results_df)} embeddings from pickle file.")
df = pd.read_csv(data_file)
df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
df_filtered = (
    df[df["valid_smiles"]].drop(columns=["valid_smiles"]).reset_index(drop=True)
)
merged_df = pd.merge(df_filtered, results_df, on="smiles", how="inner")
prediction_df = merged_df.dropna(subset=["logp"]).copy()
print(f"Original merged count: {len(merged_df)}")
print(f"Count for LogP prediction after dropping NaNs: {len(prediction_df)}")

X_features = np.stack(prediction_df["embedding"].values)
y_target = prediction_df["logp"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Property Prediction Results ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("-----------------------------------")

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

plt.xlabel("True LogP Values", fontsize=14)
plt.ylabel("Predicted LogP Values", fontsize=14)
plt.title(f"LogP Prediction Performance (R² = {r2:.3f})", fontsize=16)
plt.grid(True)
plt.savefig("results/property_prediction_scatter.png")
