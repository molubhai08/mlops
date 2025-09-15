import pandas as pd
import pickle
import sys
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(data_path, model_path, params_path):
    params = yaml.safe_load(open(params_path))["train"]
    df = pd.read_csv(data_path)

    # ✅ Drop rows where Message is missing
    df = df.dropna(subset=["Message"])
    df["Message"] = df["Message"].astype(str)  # ensure all are strings

    X = df["Message"]
    y = df["Category"]

    vectorizer = TfidfVectorizer(max_features=params["max_features"])
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=params["test_size"], random_state=42
    )

    model = LogisticRegression(max_iter=params["max_iter"])
    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump((model, vectorizer, X_test, y_test), f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    data_path = r"C:\Users\SARTHAK\Desktop\mlops\mlops\dvc\cleaned.csv"      # e.g., data/clean.csv
    model_path = r"C:\Users\SARTHAK\Desktop\mlops\mlops\dvc\model.pkl"    # e.g., model.pkl
    params_path = r"C:\Users\SARTHAK\Desktop\mlops\mlops\dvc\param.yaml"   # e.g., params.yaml
    train_model(data_path, model_path, params_path)
