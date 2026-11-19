"""Графики для дипломной работы в стиле экспериментов LHC.

Все гистограммы взвешенные и нормированы на единичную площадь, чтобы
сравнивать форму распределений сигнала и фона, а не их абсолютный
масштаб (сигнала в сотни раз меньше).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import FEATURE_LABELS, FEATURES, MISSING

SIG_COLOR = "#d62728"
BKG_COLOR = "#1f77b4"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "legend.frameon": False,
    })


def hist_sig_bkg(df, col, ax=None, bins=50, weight_col="Weight", log=False, xrange=None):
    """Нормированные гистограммы сигнала и фона по одному признаку.
    Значения -999 исключаются и указываются в подписи."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    x = df[col].to_numpy()
    w = df[weight_col].to_numpy()
    y = df["y"].to_numpy()
    ok = x != MISSING
    frac_missing = 1.0 - ok.mean()
    if xrange is None:
        lo, hi = np.percentile(x[ok], [0.5, 99.5]) if ok.any() else (0, 1)
    else:
        lo, hi = xrange
    edges = np.linspace(lo, hi, bins + 1)
    for label, color, name in ((0, BKG_COLOR, "фон"), (1, SIG_COLOR, "сигнал")):
        m = ok & (y == label)
        if m.sum() == 0:
            continue
        ax.hist(x[m], bins=edges, weights=w[m], density=True, histtype="step",
                lw=1.6, color=color, label=name)
    ax.set_xlabel(FEATURE_LABELS.get(col, col))
    ax.set_ylabel("плотность (взвеш.)")
    title = col
    if frac_missing > 0:
        title += f"  (−999: {100 * frac_missing:.0f} %)"
    ax.set_title(title, fontsize=9)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    return ax


def grid_all_features(df, ncols=5, weight_col="Weight", save_to: str | Path | None = None):
    """Сетка гистограмм по всем 30 признакам."""
    n = len(FEATURES)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows))
    for ax, col in zip(axes.ravel(), FEATURES):
        hist_sig_bkg(df, col, ax=ax, bins=40, weight_col=weight_col)
        ax.set_ylabel("")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    return fig


def plot_roc(curves: dict, ax=None):
    """curves = {"название": (fpr, tpr, auc), ...} — в HEP-координатах:
    эффективность сигнала (x) против подавления фона 1 − ε_b (y)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.2))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(tpr, 1.0 - fpr, lw=1.8, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [1, 0], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("эффективность сигнала ε_s")
    ax.set_ylabel("подавление фона 1 − ε_b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
    return ax


def plot_significance_scan(scans: dict, metric_name: str = "AMS", ax=None):
    """scans = {"название": scan_dict из metrics.significance_scan}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for name, scan in scans.items():
        ax.plot(scan["threshold"], scan["significance"], lw=1.8, label=name)
    ax.set_xlabel("порог на выход классификатора")
    ax.set_ylabel(metric_name)
    ax.legend(fontsize=8)
    return ax


def plot_classifier_output(score_train, y_train, w_train, score_test, y_test, w_test,
                           bins=40, ax=None):
    """Классическая картинка TMVA: распределение выхода классификатора
    для сигнала/фона на train (заливка) и test (точки)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))
    edges = np.linspace(0, 1, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    for label, color, name in ((0, BKG_COLOR, "фон"), (1, SIG_COLOR, "сигнал")):
        m = y_train == label
        ax.hist(score_train[m], bins=edges, weights=w_train[m], density=True,
                alpha=0.35, color=color, label=f"{name}, train")
        m = y_test == label
        h, _ = np.histogram(score_test[m], bins=edges, weights=w_test[m], density=True)
        ax.plot(centers, h, "o", ms=3.5, color=color, label=f"{name}, test")
    ax.set_xlabel("выход классификатора")
    ax.set_ylabel("плотность (взвеш.)")
    ax.legend(fontsize=8, ncol=2)
    return ax
