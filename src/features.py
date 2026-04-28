"""Funciones de feature engineering y normalización para CanastaCL."""
from __future__ import annotations

import pandas as pd

# Mapa de normalización de columnas (raw -> snake_case)
RENAME_COLUMNS = {
    "Anio": "anio",
    "Mes": "mes",
    "Semana": "semana",
    "Fecha inicio": "fecha_inicio",
    "Fecha termino": "fecha_termino",
    "ID region": "id_region",
    "Region": "region",
    "Sector": "sector",
    "Tipo de punto monitoreo": "tipo_punto",
    "Grupo": "grupo",
    "Producto": "producto",
    "Unidad": "unidad",
    "Precio minimo": "precio_min",
    "Precio maximo": "precio_max",
    "Precio promedio": "precio_prom",
}

# Categoría de unidad y factor para llevar el precio a la base de la categoría:
#   peso    -> CLP por kilo
#   volumen -> CLP por litro
#   unidad  -> CLP por unidad individual
UNIDAD_MAP: dict[str, tuple[str, float]] = {
    "$/kilo": ("peso", 1.0),
    "$/kilo (en envase de 1 kilo)": ("peso", 1.0),
    "$/kilo (en saco de 25 kilos)": ("peso", 1.0),
    "$/kilo (en saco de 5 kilos)": ("peso", 1.0),
    "$/envase 1 kilo": ("peso", 1.0),
    "$/bolsa 1 kilo": ("peso", 1.0),
    "$/bolsa 800 grs": ("peso", 1000 / 800),
    "$/pote 500 gramos": ("peso", 1000 / 500),
    "$/envase 400 gramos": ("peso", 1000 / 400),
    "$/pan de 250 gramos": ("peso", 1000 / 250),
    "$/litro": ("volumen", 1.0),
    "$/Caja de 1 Litro": ("volumen", 1.0),
    "$/botella 900 ml": ("volumen", 1000 / 900),
    "$/unidad": ("unidad", 1.0),
    "$/bandeja 12 unidades": ("unidad", 1 / 12),
    "$/bandeja 20 unidades": ("unidad", 1 / 20),
    "$/bandeja 30 unidades": ("unidad", 1 / 30),
    "$/caja 100 unidades": ("unidad", 1 / 100),
    "$/caja 180 unidades": ("unidad", 1 / 180),
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas a snake_case según RENAME_COLUMNS."""
    return df.rename(columns=RENAME_COLUMNS)


def add_unit_category(df: pd.DataFrame, unit_col: str = "unidad") -> pd.DataFrame:
    """Agrega columnas `categoria_unidad` y `factor_norm` derivadas de `unit_col`."""
    out = df.copy()
    cat = out[unit_col].map(lambda u: UNIDAD_MAP.get(u, (None, None))[0])
    factor = out[unit_col].map(lambda u: UNIDAD_MAP.get(u, (None, None))[1])
    out["categoria_unidad"] = cat.astype("category")
    out["factor_norm"] = pd.to_numeric(factor, errors="coerce")
    return out


def add_normalized_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `precio_kilo`, `precio_litro`, `precio_unitario` según categoría.

    Aplica `factor_norm` solo a la categoría correspondiente; el resto queda NaN.
    Esto permite comparaciones limpias entre productos de la misma categoría.
    """
    out = df.copy()
    if "factor_norm" not in out.columns:
        out = add_unit_category(out)

    base = out["precio_prom"] * out["factor_norm"]
    out["precio_kilo"] = base.where(out["categoria_unidad"] == "peso")
    out["precio_litro"] = base.where(out["categoria_unidad"] == "volumen")
    out["precio_unitario"] = base.where(out["categoria_unidad"] == "unidad")
    return out


def add_time_features(df: pd.DataFrame, date_col: str = "fecha_inicio") -> pd.DataFrame:
    """Agrega features temporales útiles para modelado."""
    out = df.copy()
    fecha = pd.to_datetime(out[date_col])
    out["anio_iso"] = fecha.dt.isocalendar().year.astype("int64")
    out["semana_iso"] = fecha.dt.isocalendar().week.astype("int64")
    out["mes_num"] = fecha.dt.month.astype("int8")
    out["dia_anio"] = fecha.dt.dayofyear.astype("int16")
    return out


def add_product_id(df: pd.DataFrame) -> pd.DataFrame:
    """Crea un identificador estable de producto para comparar series.

    Combina producto + unidad + tipo de punto para evitar mezclar precios
    de unidades distintas o de canales de venta distintos.
    """
    out = df.copy()
    out["producto_id"] = (
        out["producto"].astype(str)
        + " | "
        + out["unidad"].astype(str)
        + " | "
        + out["tipo_punto"].astype(str)
    )
    return out


def build_clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpieza: aplica todas las transformaciones en orden."""
    return (
        df_raw
        .pipe(normalize_columns)
        .pipe(add_unit_category)
        .pipe(add_normalized_prices)
        .pipe(add_time_features)
        .pipe(add_product_id)
    )
