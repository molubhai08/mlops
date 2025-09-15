import pandas as pd
import re
import string
import sys
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df["Message"] = df["Message"].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = r"C:\Users\SARTHAK\Desktop\mlops\mlops\dvc\corrected.csv"    # e.g., data/processed.csv
    output_path = r"C:\Users\SARTHAK\Desktop\mlops\mlops\dvc\cleaned.csv"   # e.g., data/clean.csv
    preprocess_data(input_path, output_path)
