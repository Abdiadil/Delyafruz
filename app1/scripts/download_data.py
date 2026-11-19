"""Скачивает набор HiggsML с портала CERN Open Data (запись 328) в data/.

Запуск:  python scripts/download_data.py

Если прямая ссылка не сработает (портал иногда меняет имена файлов),
откройте https://opendata.cern.ch/record/328 в браузере, скачайте
единственный файл записи (~186 МиБ в распакованном виде) и положите его
в каталог data/ под именем atlas-higgs-challenge-2014-v2.csv(.gz).
"""
from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

RECORD = "https://opendata.cern.ch/record/328"
CANDIDATES = [
    "https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz",
    "https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv",
]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _progress(count, block, total):
    if total > 0:
        done = min(100, int(100 * count * block / total))
        sys.stdout.write(f"\r  загружено {done:3d} %")
        sys.stdout.flush()


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    for url in CANDIDATES:
        target = DATA_DIR / url.rsplit("/", 1)[-1]
        if target.exists():
            print(f"Уже есть: {target}")
            break
        print(f"Пробую {url}")
        try:
            urllib.request.urlretrieve(url, target, _progress)
            print(f"\nСохранено: {target}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"\n  не удалось: {exc}")
            if target.exists():
                target.unlink()
    else:
        print(f"Скачайте файл вручную со страницы {RECORD} и положите в {DATA_DIR}")
        return 1

    gz = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv.gz"
    csv = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv"
    if gz.exists() and not csv.exists():
        print("Распаковываю .gz → .csv (pandas читает и .gz, это лишь для удобства)")
        with gzip.open(gz, "rb") as src, open(csv, "wb") as dst:
            shutil.copyfileobj(src, dst)
    # Быстрая проверка структуры
    import pandas as pd  # noqa: E402

    path = csv if csv.exists() else gz
    df = pd.read_csv(path, nrows=5)
    print(f"Столбцов: {df.shape[1]}, первые: {list(df.columns[:5])} ...")
    print("Ожидается 35 столбцов (EventId, 30 признаков, Weight, Label, KaggleSet, KaggleWeight).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
