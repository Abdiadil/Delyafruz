"""Генерирует СИНТЕТИЧЕСКИЙ файл со структурой HiggsML для проверки кода.

ВАЖНО: это не физические данные. Распределения подобраны лишь так, чтобы
сигнал и фон различались и код ноутбуков можно было прогнать без
скачивания 186 МиБ. Никакие числа из этого файла не идут в дипломную
работу. Запуск:  python scripts/make_synthetic.py [--n 40000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = -999.0
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def make(n: int = 40000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.34).astype(int)            # доля строк сигнала как в HiggsML
    sig = y == 1
    df = pd.DataFrame({"EventId": np.arange(100000, 100000 + n)})

    # Число струй задаёт структуру -999: без второй струи нет jj-переменных
    jet_num = rng.choice([0, 1, 2, 3], size=n, p=[0.40, 0.31, 0.21, 0.08])
    df["PRI_jet_num"] = jet_num

    def gauss(mu_s, mu_b, sd_s, sd_b, lo=None):
        x = np.where(sig, rng.normal(mu_s, sd_s, n), rng.normal(mu_b, sd_b, n))
        return np.clip(x, lo, None) if lo is not None else x

    def expo(scale_s, scale_b, shift=0.0):
        return np.where(sig, rng.exponential(scale_s, n), rng.exponential(scale_b, n)) + shift

    df["DER_mass_MMC"] = gauss(122, 92, 24, 35, lo=0)
    df["DER_mass_transverse_met_lep"] = expo(30, 50)
    df["DER_mass_vis"] = gauss(80, 65, 25, 30, lo=0)
    df["DER_pt_h"] = expo(45, 35)
    df["DER_deltar_tau_lep"] = gauss(2.5, 2.9, 0.6, 0.8, lo=0.4)
    df["DER_pt_tot"] = expo(20, 25)
    df["DER_sum_pt"] = expo(120, 100) + 40
    df["DER_pt_ratio_lep_tau"] = gauss(1.1, 1.5, 0.5, 0.9, lo=0.05)
    df["DER_met_phi_centrality"] = gauss(0.6, -0.1, 0.9, 1.0)
    df["PRI_tau_pt"] = expo(25, 20) + 20
    df["PRI_tau_eta"] = rng.uniform(-2.5, 2.5, n)
    df["PRI_tau_phi"] = rng.uniform(-np.pi, np.pi, n)
    df["PRI_lep_pt"] = expo(25, 20) + 20
    df["PRI_lep_eta"] = rng.uniform(-2.5, 2.5, n)
    df["PRI_lep_phi"] = rng.uniform(-np.pi, np.pi, n)
    df["PRI_met"] = expo(35, 30)
    df["PRI_met_phi"] = rng.uniform(-np.pi, np.pi, n)
    df["PRI_met_sumet"] = expo(150, 140) + 60
    df["PRI_jet_all_pt"] = np.where(jet_num > 0, expo(90, 80), 0.0)

    # Переменные ведущей струи: определены при jet_num >= 1
    has1 = jet_num >= 1
    df["PRI_jet_leading_pt"] = np.where(has1, expo(50, 45) + 30, MISSING)
    df["PRI_jet_leading_eta"] = np.where(has1, rng.uniform(-4.5, 4.5, n), MISSING)
    df["PRI_jet_leading_phi"] = np.where(has1, rng.uniform(-np.pi, np.pi, n), MISSING)
    # Второй струи и jj-переменных: при jet_num >= 2
    has2 = jet_num >= 2
    df["PRI_jet_subleading_pt"] = np.where(has2, expo(35, 30) + 30, MISSING)
    df["PRI_jet_subleading_eta"] = np.where(has2, rng.uniform(-4.5, 4.5, n), MISSING)
    df["PRI_jet_subleading_phi"] = np.where(has2, rng.uniform(-np.pi, np.pi, n), MISSING)
    df["DER_deltaeta_jet_jet"] = np.where(has2, gauss(3.0, 2.0, 1.2, 1.3, lo=0), MISSING)
    df["DER_mass_jet_jet"] = np.where(has2, expo(400, 250) + 50, MISSING)
    df["DER_prodeta_jet_jet"] = np.where(has2, gauss(-2.0, -0.5, 2.5, 2.5), MISSING)
    df["DER_lep_eta_centrality"] = np.where(has2, np.clip(gauss(0.7, 0.4, 0.3, 0.35), 0, 1), MISSING)
    # MMC иногда не сходится (~15 % событий), как в реальном наборе
    mmc_fail = rng.random(n) < 0.15
    df.loc[mmc_fail, "DER_mass_MMC"] = MISSING

    # Веса: сигнал редкий. Суммы ~ 700 (s) и ~ 400 000 (b), как в train-части Kaggle.
    w = np.where(sig, rng.gamma(2.0, 0.0010, n), rng.gamma(2.0, 1.0, n))
    w[sig] *= 700.0 / w[sig].sum()
    w[~sig] *= 400_000.0 / w[~sig].sum()
    df["Weight"] = w
    df["Label"] = np.where(sig, "s", "b")
    df["KaggleSet"] = rng.choice(["t", "b", "v", "u"], size=n, p=[0.3, 0.1, 0.5, 0.1])
    df["KaggleWeight"] = w

    order = ["EventId",
             "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
             "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
             "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
             "DER_met_phi_centrality", "DER_lep_eta_centrality",
             "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta",
             "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num",
             "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
             "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi",
             "PRI_jet_all_pt", "Weight", "Label", "KaggleSet", "KaggleWeight"]
    return df[order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40000)
    args = p.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "synthetic_higgsml.csv"
    make(args.n).to_csv(out, index=False)
    print(f"Синтетический файл: {out}  (строк: {args.n}). Только для проверки кода!")


if __name__ == "__main__":
    main()
