"""Метрики разделения сигнала и фона.

Все функции работают с весами событий: в данных HiggsML (и в любых
Monte-Carlo выборках ATLAS) каждое событие имеет вес, сумма весов
даёт ожидаемое число событий в реальных данных. Поэтому "accuracy"
бессмысленна, а осмысленны s (ожидаемый сигнал), b (ожидаемый фон)
и статистическая значимость.

Формулы
-------
AMS (HiggsML, регуляризованная значимость):
    AMS = sqrt(2 * ((s + b + b_r) * ln(1 + s / (b + b_r)) - s)),  b_r = 10

Z0 (асимптотическая значимость открытия, Cowan et al., EPJC 71 (2011) 1554):
    Z0 = sqrt(2 * ((s + b) * ln(1 + s / b) - s))  ≈  s / sqrt(b)  при s << b
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def ams(s, b, b_reg: float = 10.0):
    """Approximate Median Significance из HiggsML."""
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(2.0 * ((s + b + b_reg) * np.log1p(s / (b + b_reg)) - s))


def z0(s, b):
    """Асимптотическая значимость открытия (Cowan et al. 2011)."""
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(2.0 * ((s + b) * np.log1p(s / b) - s))
    return np.where(b > 0, out, 0.0)


def weighted_counts(y_true, y_pred, weights):
    """Ожидаемые s и b среди событий, отобранных как сигнал (y_pred == 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(bool)
    weights = np.asarray(weights, dtype=float)
    s = weights[(y_true == 1) & y_pred].sum()
    b = weights[(y_true == 0) & y_pred].sum()
    return float(s), float(b)


def significance_scan(y_true, scores, weights, metric: str = "ams", n_points: int = 300,
                      min_events: int = 200):
    """Скан порога по выходу классификатора.

    Возвращает словарь с массивами: threshold, s, b, значимость.
    Порог берётся по квантилям score, чтобы точки были равномерны по
    числу событий, а не по значению score. В отобранной области должно
    быть не меньше `min_events` строк: на хвосте из нескольких событий
    значимость — статистический шум, и порог, выбранный там, не
    переносится на test.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    func = ams if metric == "ams" else z0

    order = np.argsort(scores)[::-1]          # от самых "сигнальных"
    y_sorted = y_true[order]
    w_sorted = weights[order]
    sc_sorted = scores[order]

    cum_s = np.cumsum(w_sorted * (y_sorted == 1))
    cum_b = np.cumsum(w_sorted * (y_sorted == 0))

    start = min(max(min_events, 1), len(scores)) - 1
    idx = np.unique(np.linspace(start, len(scores) - 1, n_points).astype(int))
    s = cum_s[idx]
    b = cum_b[idx]
    thr = sc_sorted[idx]
    sig = func(s, b)
    return {"threshold": thr, "s": s, "b": b, "significance": sig, "n_selected": idx + 1}


def best_threshold(y_true, scores, weights, metric: str = "ams", n_points: int = 300,
                   min_events: int = 200):
    """Порог, максимизирующий значимость, и сама значимость в максимуме."""
    scan = significance_scan(y_true, scores, weights, metric=metric, n_points=n_points,
                             min_events=min_events)
    i = int(np.nanargmax(scan["significance"]))
    return {
        "threshold": float(scan["threshold"][i]),
        "significance": float(scan["significance"][i]),
        "s": float(scan["s"][i]),
        "b": float(scan["b"][i]),
        "n_selected": int(scan["n_selected"][i]),
    }


def balanced_weights(y, weights, per_class_sum: float | None = None):
    """Веса для обучения: суммарный вклад сигнала и фона уравнивается.

    Без этого классификатор почти не видит сигнал (его суммарный вес в
    сотни раз меньше). Средний вес приводится к 1 (сумма на класс =
    N/2): XGBoost сравнивает сумму весов в листе с min_child_weight, и
    при весах ~1e-5 ни одно разбиение не проходит — модель вырождается
    в константу. Значимость по-прежнему считается с настоящими весами.
    """
    y = np.asarray(y)
    w = np.asarray(weights, dtype=float).copy()
    target = per_class_sum if per_class_sum is not None else len(y) / 2.0
    for label in (0, 1):
        m = y == label
        w[m] *= target / w[m].sum()
    return w


def evaluate_at_threshold(y_true, scores, weights, threshold: float):
    """Значимости и ожидаемые s, b при фиксированном пороге (для test-выборки)."""
    y_pred = np.asarray(scores) >= threshold
    s, b = weighted_counts(y_true, y_pred, weights)
    return {"s": s, "b": b, "ams": float(ams(s, b)), "z0": float(z0(s, b))}


def weighted_roc(y_true, scores, weights):
    """ROC-кривая и AUC с учётом весов."""
    fpr, tpr, _ = roc_curve(y_true, scores, sample_weight=weights)
    auc = roc_auc_score(y_true, scores, sample_weight=weights)
    return fpr, tpr, float(auc)


def ks_overtraining(scores_train, y_train, scores_test, y_test):
    """Тест Колмогорова–Смирнова train vs test отдельно для сигнала и фона.

    Большой p-value (> 0.05) означает, что распределения выхода
    классификатора на train и test статистически не различимы, то есть
    явного переобучения нет. Это стандартная проверка в TMVA/ATLAS.
    """
    scores_train = np.asarray(scores_train)
    scores_test = np.asarray(scores_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    out = {}
    for label, name in ((1, "signal"), (0, "background")):
        ks = stats.ks_2samp(scores_train[y_train == label], scores_test[y_test == label])
        out[name] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
    return out
