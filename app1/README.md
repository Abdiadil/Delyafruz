# Дипломная работа: ML для разделения сигнала и фона по открытым данным LHC

Казахский национальный университет им. аль-Фараби, специальность «Киберфизика», бакалавриат.
Научный руководитель — Аскарулы Абдиадиль. Защита — лето 2027.

Репозиторий содержит весь код работы. Любой результат в тексте диплома должен
воспроизводиться запуском ноутбуков из этого репозитория по порядку.

## Структура

```
thesis-lhc-ml/
  data/            # данные (не в git): скачиваются скриптом, см. data/README.md
  notebooks/       # 01_eda → 02_baseline → 03_bdt → 04_mlp → 05_shap → 06_atlas_4lep
  src/             # общий код: data.py (загрузка, веса, разбиение), metrics.py (AMS, Z0,
                   # скан порога, KS-тест), plots.py (гистограммы, ROC, выход классификатора),
                   # baselines.py (пороговый отбор, логистическая регрессия, XGBoost/LightGBM,
                   # оценка одного разбиения по протоколу, таблица mean ± σ по повторам)
  scripts/         # download_data.py (HiggsML с CERN Open Data), make_synthetic.py (проверка кода)
  tests/           # проверки формул значимости
  results/         # рисунки и таблицы, которые идут в текст работы
  thesis/          # текст, презентация, библиография
```

## Установка

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# PyTorch для этапа 3 ставится отдельно под вашу CUDA: https://pytorch.org/get-started/locally/
python tests/test_metrics.py          # 4 строки "ok"
python scripts/download_data.py       # ~60 МБ .gz, распакуется в data/
jupyter lab                           # открыть notebooks/01_eda.ipynb
```

Если скачивание скриптом не удалось — файл записи 328 портала CERN Open Data
(https://opendata.cern.ch/record/328) скачивается в браузере и кладётся в `data/`.

Для проверки, что окружение работает, без 186 МБ данных:
`python scripts/make_synthetic.py` создаст `data/synthetic_higgsml.csv` той же структуры.
Ноутбуки сами найдут его, если настоящего файла нет, и напечатают предупреждение.
**Числа с синтетических данных в работу не идут.**

## Ноутбуки

| Ноутбук | Что делает | Время на CPU / GPU |
|---|---|---|
| `01_eda` | Разведочный анализ: веса, структура −999, распределения, одиночный AUC, окно по массе | 2–3 мин |
| `02_baseline` | Пороговый отбор, логистическая регрессия, XGBoost по умолчанию; первая таблица | 3–5 мин |
| `03_bdt` | Optuna для XGBoost и LightGBM, лучшая модель, кривая обучения, 5 повторов с σ, протокол Kaggle, важность признаков | ~1 ч / 10–15 мин |

Быстрая проверка, что ноутбук запускается (3 пробы Optuna, 2 повтора):
`THESIS_FAST=1 jupyter lab` (Windows PowerShell: `$env:THESIS_FAST="1"; jupyter lab`).
Число проб и повторов задаётся в первой ячейке каждого ноутбука (`N_TRIALS`, `N_REPEATS`).

## Протокол сравнения (одинаковый для всех методов)

1. Разбиение 60/20/20 (train/val/test), стратифицированное, seed фиксирован.
   Веса в каждой части пересчитаны так, чтобы суммы s и b совпадали с полным набором
   (`src.data.split_train_val_test`) — иначе значимость на test занижена в 5 раз.
2. Для обучения веса классов уравнены, средний вес = 1 (`src.metrics.balanced_weights`).
   Значимость всегда считается с настоящими весами.
3. Порог классификатора — максимум AMS на val при ≥ 200 отобранных событиях
   (`src.metrics.best_threshold`). На test порог фиксирован, считается один раз.
4. Каждая модель — 5 повторов с разными seed (с ноутбука 03), в таблицу идёт среднее ± σ.
5. Проверка переобучения: KS-тест train/test по выходу классификатора (`ks_overtraining`).

## Метрики

- AMS (HiggsML): `sqrt(2((s+b+10) ln(1 + s/(b+10)) − s))`
- Z0 (Cowan et al. 2011): `sqrt(2((s+b) ln(1 + s/b) − s))`
- Взвешенный ROC AUC; кривая «эффективность сигнала — подавление фона».

## GPU

XGBoost: `XGBClassifier(device="cuda")`. LightGBM: `device="gpu"` (нужна сборка с GPU).
PyTorch: `model.to("cuda")`. Для HiggsML (818 тыс. строк × 30 признаков) всё считается
на RTX 4080/5090 за минуты, на CPU — за минуты–десятки минут.

## Данные и цитирование

- HiggsML: ATLAS Collaboration (2014). Dataset from the ATLAS Higgs Boson Machine Learning
  Challenge 2014. CERN Open Data Portal, запись 328. Статья: Adam-Bourdarios C. et al.,
  PMLR 42:19–55, 2015.
- ATLAS Open Data 13 ТэВ (релиз 2020, 10 фб⁻¹, реальные данные + MC): https://opendata.atlas.cern/
  и https://opendata.cern.ch/; выборка 4 лептона: `https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/`
