from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train.csv")
    for col in ["cookie_created_at", "window_start_ts", "window_end_ts"]:
        df[col] = pd.to_datetime(df[col])
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "test.csv")
    for col in ["cookie_created_at", "window_start_ts", "window_end_ts"]:
        df[col] = pd.to_datetime(df[col])
    return df


def load_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "events.csv")
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    return df