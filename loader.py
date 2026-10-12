import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
FILES = {
    "official": DATA_DIR / "official_data_en.csv",
    "volunteer": DATA_DIR / "volunteer_data_en.csv",
}


def load_raw(name: str) -> pd.DataFrame:
    """Load one raw CSV exactly as stored, with no renaming or filtering."""
    path = FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.resolve()}")
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, name: str) -> None:
    """Print what we need to confirm before writing any real logic."""
    print(f"=== {name} === shape={df.shape}")
    print(df.dtypes, "\n")
    print(df.head(3), "\n")
    for c in df.columns:
        if any(k in c.lower() for k in ("start", "end", "finish", "time", "date")):
            t = pd.to_datetime(df[c], errors="coerce", utc=True)
            print(f"{c}: min={t.min()} max={t.max()} unparseable={t.isna().sum()}")
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() < 60:
            print(f"\n{c} values:\n{df[c].value_counts().head(30)}")


if __name__ == "__main__":
    for n in FILES:
        summarize(load_raw(n), n)