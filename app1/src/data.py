"""Загрузка и подготовка данных HiggsML (ATLAS, CERN Open Data, запись 328).

Структура файла atlas-higgs-challenge-2014-v2.csv:
    EventId                  — номер события
    DER_* (13 признаков)     — производные величины, посчитанные физиками
                               (массы, pT системы, угловые разности и т.п.)
    PRI_* (17 признаков)     — "сырые" измеренные величины (pT, eta, phi
                               лептона, тау, струй, недостающая энергия)
    Weight                   — вес события (нормировка MC на 2012 год)
    Label                    — 's' (сигнал H->tautau) или 'b' (фон)
    KaggleSet                — t / b / v / u — разбиение конкурса Kaggle
    KaggleWeight             — вес, перенормированный внутри подмножества

Значение -999.0 означает "величина не определена" (например, нет второй
струи — нечего считать DER_mass_jet_jet). Это не ошибка данных, а
физическая структура, и её нужно сохранять как информацию.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MISSING = -999.0

DER_FEATURES = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality", "DER_lep_eta_centrality",
]
PRI_FEATURES = [
    "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
    "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi",
    "PRI_met", "PRI_met_phi", "PRI_met_sumet",
    "PRI_jet_num",
    "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi",
    "PRI_jet_all_pt",
]
FEATURES = DER_FEATURES + PRI_FEATURES

# Человеческие подписи для графиков (единицы: ГэВ для масс, pT и MET)
FEATURE_LABELS = {
    "DER_mass_MMC": "m_MMC(τ,τ) [ГэВ]",
    "DER_mass_transverse_met_lep": "m_T(MET, ℓ) [ГэВ]",
    "DER_mass_vis": "m_vis(τ,ℓ) [ГэВ]",
    "DER_pt_h": "p_T(H-кандидат) [ГэВ]",
    "DER_deltaeta_jet_jet": "Δη(j,j)",
    "DER_mass_jet_jet": "m(j,j) [ГэВ]",
    "DER_prodeta_jet_jet": "η_j1·η_j2",
    "DER_deltar_tau_lep": "ΔR(τ,ℓ)",
    "DER_pt_tot": "p_T^tot [ГэВ]",
    "DER_sum_pt": "Σp_T [ГэВ]",
    "DER_pt_ratio_lep_tau": "p_T(ℓ)/p_T(τ)",
    "DER_met_phi_centrality": "центральность φ(MET)",
    "DER_lep_eta_centrality": "центральность η(ℓ)",
    "PRI_tau_pt": "p_T(τ_had) [ГэВ]",
    "PRI_tau_eta": "η(τ_had)",
    "PRI_tau_phi": "φ(τ_had)",
    "PRI_lep_pt": "p_T(ℓ) [ГэВ]",
    "PRI_lep_eta": "η(ℓ)",
    "PRI_lep_phi": "φ(ℓ)",
    "PRI_met": "E_T^miss [ГэВ]",
    "PRI_met_phi": "φ(E_T^miss)",
    "PRI_met_sumet": "ΣE_T [ГэВ]",
    "PRI_jet_num": "число струй",
    "PRI_jet_leading_pt": "p_T(j1) [ГэВ]",
    "PRI_jet_leading_eta": "η(j1)",
    "PRI_jet_leading_phi": "φ(j1)",
    "PRI_jet_subleading_pt": "p_T(j2) [ГэВ]",
    "PRI_jet_subleading_eta": "η(j2)",
    "PRI_jet_subleading_phi": "φ(j2)",
    "PRI_jet_all_pt": "Σp_T(струи) [ГэВ]",
}


def find_data_file(data_dir: str | Path = "data") -> Path:
    """Ищет файл HiggsML в каталоге data/ (сначала настоящий, потом синтетический)."""
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "atlas-higgs-challenge-2014-v2.csv",
        data_dir / "atlas-higgs-challenge-2014-v2.csv.gz",
        data_dir / "synthetic_higgsml.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Файл данных не найден. Запустите `python scripts/download_data.py` "
        "или `python scripts/make_synthetic.py` (только для проверки кода)."
    )


def load_higgsml(path: str | Path) -> pd.DataFrame:
    """Читает CSV (можно .gz) и добавляет столбец y: 1 = сигнал, 0 = фон."""
    df = pd.read_csv(path)
    missing_cols = set(FEATURES + ["Weight", "Label"]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"В файле нет столбцов: {sorted(missing_cols)}")
    df["y"] = (df["Label"] == "s").astype(int)
    return df


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Булева таблица: где стоят -999 (величина не определена)."""
    return df[FEATURES] == MISSING


def with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Копия, где -999 заменены на NaN (XGBoost/LightGBM умеют работать с NaN)."""
    out = df.copy()
    out[FEATURES] = out[FEATURES].replace(MISSING, np.nan)
    return out


def rescale_weights(part: pd.DataFrame, full: pd.DataFrame, col: str = "Weight") -> pd.Series:
    """Пересчитывает веса подвыборки так, чтобы сумма весов сигнала и фона
    в подвыборке равнялась сумме по полному набору.

    Без этого s и b на test-выборке (20 % событий) были бы в 5 раз меньше
    истинных ожидаемых значений, и значимость была бы занижена.
    """
    w = part[col].astype(float).copy()
    for label in (0, 1):
        m_part = part["y"] == label
        m_full = full["y"] == label
        factor = full.loc[m_full, col].sum() / part.loc[m_part, col].sum()
        w[m_part] = w[m_part] * factor
    return w


def split_train_val_test(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2,
                         seed: int = 42):
    """Стратифицированное разбиение 60/20/20 с пересчётом весов в каждой части.

    Возвращает три DataFrame с дополнительным столбцом `w` — вес,
    приведённый к полной ожидаемой статистике.
    """
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["y"],
                                       random_state=seed)
    val_frac = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_frac,
                                  stratify=train_val["y"], random_state=seed)
    parts = []
    for part in (train, val, test):
        part = part.copy()
        part["w"] = rescale_weights(part, df)
        parts.append(part)
    return tuple(parts)


def expected_yields(df: pd.DataFrame, col: str = "Weight") -> dict:
    """Ожидаемое число сигнальных и фоновых событий (сумма весов)."""
    s = float(df.loc[df["y"] == 1, col].sum())
    b = float(df.loc[df["y"] == 0, col].sum())
    return {"s": s, "b": b, "n_signal_rows": int((df["y"] == 1).sum()),
            "n_background_rows": int((df["y"] == 0).sum())}
