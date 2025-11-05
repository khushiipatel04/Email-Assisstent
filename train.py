import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

print("Script started...")

# 1. Load data
csv_file_path = "emails.csv"
if not os.path.exists(csv_file_path):
    print(f"Error: '{csv_file_path}' not found.")
    print("Please make sure 'emails.csv' is in the same directory.")
else:
    print(f"Loading data from '{csv_file_path}'...")
    df = pd.read_csv(csv_file_path)

    # 2. Clean data
    df.dropna(subset=["body", "intent"], inplace=True)
    df["body"] = df["body"].astype(str)
    df["intent"] = df["intent"].astype(str)
    print(f"Loaded {len(df)} valid training samples.")

    # 3. Define model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(multi_class="ovr", solver="liblinear", max_iter=1000)),
    ])

    # 4. Train model
    print("Training the model... This might take a moment.")
    model.fit(df["body"], df["intent"])

    # 5. Save model
    model_filename = "triage_model.pkl"
    joblib.dump(model, model_filename)

    print(f"\n✅ Success! Model trained and saved as '{model_filename}'.")
    print("You can now run the Streamlit app.")