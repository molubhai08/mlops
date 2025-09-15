import pandas as pd
import sys

def ingest_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    input_path = r"C:\Users\SARTHAK\Desktop\mlops\mlops\dvc\spam.csv"  # e.g., data/raw/spam.csv
    output_path = r"C:\Users\SARTHAK\Desktop\mlops\mlops\dvc\corrected.csv"  # e.g., data/processed.csv
    ingest_data(input_path, output_path)
