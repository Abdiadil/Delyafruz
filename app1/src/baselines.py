"""Базовые методы и общие обёртки, используемые в ноутбуках 02–04.

Здесь собрано то, что повторяется в каждом эксперименте: пороговый отбор,
логистическая регрессия, конфигурация XGBoost/LightGBM, оценка одного
разбиения по единому протоколу (порог на val → метрики на test).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, MISSING, with_nan
from .metrics import (ams, balanced_weights, best_threshold, evaluate_at_threshold,
                      weighted_counts, weighted_roc)

# --------------------------------------------------------------------------- пороговый отбор
CUT_GRID = [(m_lo, m_hi, mt, dr)
            for m_lo in (95, 100, 105, 110)
            for m_hi in (135, 140, 150, 160)
            for mt in (40, 60, 80, 1e9)
            for dr in (2.5, 3.0, 3.5, 1e9)]


def cut_selection(part: pd.DataFrame, m_lo, m_hi, mt_max, dr_max) -> np.ndarray:
    """Окно по m_MMC + верхние пороги на m_T(MET, l) и dR(tau, l)."""
    m = part["DER_mass_MMC"].to_numpy()
    sel = (m != MISSING) & (m >= m_lo) & (m <= m_hi)
    sel &= part["DER_mass_transverse_met_lep"].to_numpy() <= mt_max
    sel &= part["DER_deltar_tau_lep"].to_numpy() <= dr_max
    return sel


def cut_based(val: pd.DataFrame, test: pd.DataFrame, grid=CUT_GRID) -> dict:
    """Выбирает пороги по максимуму AMS на val, считает метрики на test."""
    best, best_ams = None, -1.0
    for params in grid:
        s, b = weighted_counts(val["y"], cut_selection(val, *params), val["w"])
        a = float(ams(s, b))
        if a > best_ams:
            best, best_ams = params, a
    s, b = weighted_counts(test["y"], cut_selection(test, *best), test["w"])
    ev = {"s": s, "b": b, "ams": float(ams(s, b))}
    from .metrics import z0
    ev["z0"] = float(z0(s, b))
    return {"method": "пороговый отбор", "params": best, "AUC_val": np.nan,
            "AMS_val": best_ams, "AMS_test": ev["ams"], "Z0_test": ev["z0"],
            "s_test": ev["s"], "b_test": ev["b"]}


# --------------------------------------------------------------------------- матрицы признаков
def xy(part: pd.DataFrame, features=FEATURES, weight_col: str = "w"):
    """X (с NaN вместо -999), y, настоящие веса, уравновешенные веса для обучения."""
    X = with_nan(part)[features].to_numpy()
    y = part["y"].to_numpy()
    w = part[weight_col].to_numpy().astype(float)
    return X, y, w, balanced_weights(y, w)


def evaluate_split(y_val, score_val, w_val, y_test=None, score_test=None, w_test=None,
                   method: str = "", min_events: int = 200) -> dict:
    """Единый протокол: порог по максимуму AMS на val, метрики на test.

    Если test не передан (например, во время подбора гиперпараметров),
    считаются только метрики на val — test остаётся нетронутым.
    """
    bt = best_threshold(y_val, score_val, w_val, metric="ams", min_events=min_events)
    _, _, auc_val = weighted_roc(y_val, score_val, w_val)
    res = {"method": method, "AUC_val": auc_val, "AMS_val": bt["significance"],
           "threshold": bt["threshold"], "n_selected_val": bt["n_selected"]}
    if y_test is not None:
        ev = evaluate_at_threshold(y_test, score_test, w_test, bt["threshold"])
        _, _, auc_test = weighted_roc(y_test, score_test, w_test)
        res.update({"AUC_test": auc_test, "AMS_test": ev["ams"], "Z0_test": ev["z0"],
                    "s_test": ev["s"], "b_test": ev["b"]})
    return res


# --------------------------------------------------------------------------- логистическая регрессия
def make_logreg(C: float = 1.0):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000),
    )


def fit_logreg(train, val, test, C: float = 1.0, features=FEATURES) -> dict:
    X_tr, y_tr, _, wb_tr = xy(train, features)
    X_va, y_va, w_va, _ = xy(val, features)
    X_te, y_te, w_te, _ = xy(test, features)
    model = make_logreg(C)
    model.fit(X_tr, y_tr, logisticregression__sample_weight=wb_tr)
    res = evaluate_split(y_va, model.predict_proba(X_va)[:, 1], w_va,
                         y_te, model.predict_proba(X_te)[:, 1], w_te,
                         method="логистическая регрессия")
    res["model"] = model
    return res


# --------------------------------------------------------------------------- градиентный бустинг
def xgb_device() -> str:
    """'cuda', если XGBoost собран с CUDA и видит GPU, иначе 'cpu'.

    XGBoost 2.x/3.x без GPU не бросает исключение, а молча переключается
    на CPU с предупреждением, поэтому проверяем наличие NVIDIA GPU через
    nvidia-smi (есть на Windows и Linux с драйвером NVIDIA; на Mac его нет).
    """
    import shutil
    import subprocess

    import xgboost as xgb
    try:
        if not xgb.build_info().get("USE_CUDA", False):
            return "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        return "cuda" if out.returncode == 0 and "GPU" in out.stdout else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


XGB_DEFAULT = dict(n_estimators=400, learning_rate=0.1, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8)


def fit_xgb(train, val, test=None, params: dict | None = None, device: str = "cpu",
            seed: int = 42, early_stopping_rounds: int = 50, features=FEATURES,
            method: str = "XGBoost", return_model: bool = True) -> dict:
    """Обучает XGBoost с ранней остановкой по val и оценивает по протоколу.
    test=None — только метрики на val (режим подбора гиперпараметров)."""
    import xgboost as xgb
    p = dict(XGB_DEFAULT)
    if params:
        p.update(params)
    X_tr, y_tr, w_tr, wb_tr = xy(train, features)
    X_va, y_va, w_va, wb_va = xy(val, features)
    model = xgb.XGBClassifier(**p, tree_method="hist", device=device,
                              eval_metric="logloss", early_stopping_rounds=early_stopping_rounds,
                              random_state=seed, n_jobs=-1)
    t0 = time.time()
    model.fit(X_tr, y_tr, sample_weight=wb_tr,
              eval_set=[(X_tr, y_tr), (X_va, y_va)],
              sample_weight_eval_set=[wb_tr, wb_va], verbose=False)
    score_va = model.predict_proba(X_va)[:, 1]
    if test is not None:
        X_te, y_te, w_te, _ = xy(test, features)
        score_te = model.predict_proba(X_te)[:, 1]
        res = evaluate_split(y_va, score_va, w_va, y_te, score_te, w_te, method=method)
    else:
        res = evaluate_split(y_va, score_va, w_va, method=method)
    res["best_iteration"] = int(model.best_iteration) + 1
    res["fit_seconds"] = time.time() - t0
    if return_model:
        res["model"] = model
        res["scores"] = {"train": model.predict_proba(X_tr)[:, 1], "val": score_va}
        if test is not None:
            res["scores"]["test"] = score_te
        res["evals"] = model.evals_result()
    return res


LGB_DEFAULT = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


def fit_lgb(train, val, test=None, params: dict | None = None, seed: int = 42,
            early_stopping_rounds: int = 50, features=FEATURES,
            method: str = "LightGBM", return_model: bool = True) -> dict:
    """То же для LightGBM (CPU; NaN поддерживаются напрямую)."""
    import lightgbm as lgb
    p = dict(LGB_DEFAULT)
    if params:
        p.update(params)
    X_tr, y_tr, w_tr, wb_tr = xy(train, features)
    X_va, y_va, w_va, wb_va = xy(val, features)
    model = lgb.LGBMClassifier(**p, random_state=seed, n_jobs=-1, verbose=-1)
    t0 = time.time()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # eval_set переименован в новых версиях LightGBM
        model.fit(X_tr, y_tr, sample_weight=wb_tr,
                  eval_set=[(X_va, y_va)], eval_sample_weight=[wb_va],
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    score_va = model.predict_proba(X_va)[:, 1]
    if test is not None:
        X_te, y_te, w_te, _ = xy(test, features)
        score_te = model.predict_proba(X_te)[:, 1]
        res = evaluate_split(y_va, score_va, w_va, y_te, score_te, w_te, method=method)
    else:
        res = evaluate_split(y_va, score_va, w_va, method=method)
    res["best_iteration"] = int(model.best_iteration_)
    res["fit_seconds"] = time.time() - t0
    if return_model:
        res["model"] = model
        res["scores"] = {"train": model.predict_proba(X_tr)[:, 1], "val": score_va}
        if test is not None:
            res["scores"]["test"] = score_te
    return res


def summarize_repeats(rows: list[dict]) -> pd.DataFrame:
    """Таблица mean ± std по повторам: строки — методы."""
    df = pd.DataFrame(rows)
    cols = [c for c in ("AUC_test", "AMS_val", "AMS_test", "Z0_test", "s_test", "b_test") if c in df]
    agg = df.groupby("method", sort=False)[cols].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" if c.startswith(("AUC", "AMS", "Z0")) else f"{m:.0f} ± {s:.0f}"
                  for m, s in zip(agg[(c, "mean")], agg[(c, "std")])]
    out["n_repeats"] = df.groupby("method", sort=False).size()
    return out
