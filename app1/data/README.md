# data/

Каталог для данных. Файлы данных не хранятся в git (см. .gitignore).

## HiggsML (основной набор, этапы 1–3)

- Источник: CERN Open Data, запись 328 — https://opendata.cern.ch/record/328
- Скачать: `python scripts/download_data.py` → `atlas-higgs-challenge-2014-v2.csv(.gz)`
- 818 238 событий, 35 столбцов: EventId, 13 DER_ + 17 PRI_ признаков, Weight, Label (s/b),
  KaggleSet (t/b/v/u), KaggleWeight. Пропуски обозначены −999.0.
- Лицензия CC0. Это Monte-Carlo симуляция ATLAS (8 ТэВ, 2012 г.), реальных данных здесь нет.

## ATLAS Open Data 13 ТэВ, релиз 2020 (расширение, этап 4) — реальные данные

- Выборка с ≥ 4 лептонами (канал H → ZZ* → 4ℓ), ROOT-файлы, читаются `uproot`:
  `https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/`
  Официальный ноутбук HZZ_BDT_demo.ipynb берёт данные именно отсюда.
- Реальные столкновения: файлы периодов `data_A`, `data_B`, `data_C`, `data_D` (вместе 10 фб⁻¹, 2016 г.).
- Симуляция: сигнал (ggH → ZZ → 4ℓ, VBF H → ZZ → 4ℓ), фон (ZZ → 4ℓ, Z + струи, t t̄).
  Точные имена файлов (`mc_<DSID>.<процесс>.4lep.root`) и сечения — в ноутбуке HZZAnalysis.ipynb
  из коллекции https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata
- Скрипт загрузки появится в `scripts/download_atlas_4lep.py` на этапе 4.

## synthetic_higgsml.csv

Создаётся `python scripts/make_synthetic.py` только для проверки, что код запускается.
Не физика. Удалите перед запуском на настоящих данных, чтобы не перепутать.
