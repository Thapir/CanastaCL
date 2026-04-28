"""Modelos de pronóstico para CanastaCL.

Helpers para forecasting univariado de series semanales de precios.
Diseñado para series cortas (~15-17 puntos): un solo año o menos.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Resultado de un modelo de pronóstico."""

    name: str
    y_pred: pd.Series
    y_lower: pd.Series | None = None
    y_upper: pd.Series | None = None


def select_series(
    df: pd.DataFrame,
    producto_id: str,
    region: str,
    date_col: str = "fecha_inicio",
    value_col: str = "precio_prom",
) -> pd.Series:
    """Construye la serie semanal de precio para un producto en una región.

    Promedia los registros de la misma semana entre sectores/locales.
    """
    sub = df[(df["producto_id"] == producto_id) & (df["region"] == region)]
    if sub.empty:
        raise ValueError(f"Sin datos para producto_id='{producto_id}' en region='{region}'")
    serie = sub.groupby(date_col)[value_col].mean().sort_index()
    serie.index = pd.DatetimeIndex(serie.index)
    serie.name = "precio"
    return serie


def train_test_split_temporal(
    serie: pd.Series, test_weeks: int = 4
) -> tuple[pd.Series, pd.Series]:
    """Reserva las últimas `test_weeks` observaciones para evaluación."""
    if len(serie) <= test_weeks:
        raise ValueError(f"Serie de {len(serie)} puntos es muy corta para test_weeks={test_weeks}")
    return serie.iloc[:-test_weeks], serie.iloc[-test_weeks:]


def forecast_naive(train: pd.Series, index: pd.DatetimeIndex) -> ForecastResult:
    """Predicción ingenua: el futuro = último valor observado."""
    last = float(train.iloc[-1])
    pred = pd.Series([last] * len(index), index=index)
    return ForecastResult("Naive", pred)


def forecast_arima(
    train: pd.Series, index: pd.DatetimeIndex, order: tuple[int, int, int] = (1, 1, 1)
) -> ForecastResult:
    """ARIMA con orden fijo. Para series muy cortas usar órdenes pequeños."""
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train.values, order=order).fit()
        forecast = model.get_forecast(steps=len(index))
        mean = pd.Series(forecast.predicted_mean, index=index)
        ci = forecast.conf_int(alpha=0.05)
        lower = pd.Series(ci[:, 0], index=index)
        upper = pd.Series(ci[:, 1], index=index)
    return ForecastResult(f"ARIMA{order}", mean, lower, upper)


def forecast_prophet(train: pd.Series, index: pd.DatetimeIndex) -> ForecastResult:
    """Prophet con tendencia simple, sin estacionalidades (no hay historia para detectarlas)."""
    from prophet import Prophet

    df_p = pd.DataFrame({"ds": train.index, "y": train.values})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
        )
        model.fit(df_p)
        future = pd.DataFrame({"ds": index})
        fcst = model.predict(future)
    mean = pd.Series(fcst["yhat"].to_numpy(), index=index)
    lower = pd.Series(fcst["yhat_lower"].to_numpy(), index=index)
    upper = pd.Series(fcst["yhat_upper"].to_numpy(), index=index)
    return ForecastResult("Prophet", mean, lower, upper)


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula MAE, RMSE y MAPE comparando predicción vs realidad."""
    err = y_pred.to_numpy() - y_true.to_numpy()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / y_true.to_numpy()))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}
