"""Carga del dataset crudo de precios al consumidor."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw(filename: str = "precio_consumidor_2026.csv") -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(
        path,
        sep=",",
        encoding="utf-8-sig",
        decimal=",",
        dtype={"ID region": "Int64"},
        parse_dates=["Fecha inicio", "Fecha termino"],
    )
    return df
