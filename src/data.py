"""Download and prepare the UCI Cleveland heart-disease dataset."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
UCI_CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def download_dataset(destination="data/heart_disease_uci.csv"):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_path = destination.with_suffix(".raw")
    urlretrieve(UCI_CLEVELAND_URL, raw_path)
    pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?").to_csv(destination, index=False)
    raw_path.unlink(missing_ok=True)
    return destination

def load_dataset(path):
    frame = pd.read_csv(path, na_values=["?", "NA", "N/A", ""])
    if "target" not in frame.columns:
        if "num" not in frame.columns:
            raise ValueError("Dataset needs a 'target' or UCI 'num' column.")
        frame = frame.rename(columns={"num": "target"})
    frame["target"] = (pd.to_numeric(frame["target"], errors="coerce") > 0).astype(int)
    return frame
