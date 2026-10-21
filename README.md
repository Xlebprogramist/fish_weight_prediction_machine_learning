# FishGrow, ЛР2: прогностическая модель массы рыбы и MLflow

**Дисциплина:** Введение в искусственный интеллект, ПетрГУ, 2026  
**Студент:** Глеб Осипов  
**Ветка:** `lab/02-regression`  
**Исходный проект:** [esharma3/fish_weight_prediction_machine_learning](https://github.com/esharma3/fish_weight_prediction_machine_learning), коммит `c948d7f` (его описание сохранено в конце файла)

Модель предсказывает массу рыбы (г) по виду и измерениям тела. Итоговая
модель — степенной закон `log(масса) = −2.40 + 1.60·log(Length3) + 0.71·log(Height) + 0.70·log(Width) + сдвиг(вид)`:
CV MSE 3288 ± 2058, тест MSE 1429, MAE 23.8 г, R² 0.986.
Подробности — [отчёт](reports/lab02-report.md), [паспорт модели](models/model-card.md),
[паспорт данных](data/README.md).

## 1. Окружение

Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-course.txt
```

## 2. Основной сценарий

Все команды — из корня репозитория.

```bash
pytest                              # 17 быстрых тестов, ~3 с, без полного обучения
ruff check src tests                # статическая проверка
python -m src.train --all           # E0–E4: CV, подбор, тест, запись в MLflow, ≈ 2 мин
python -m src.evaluate              # пересчёт метрик сохранённой моделью + анализ ошибок
python -m src.export_mlflow         # выгрузка метаданных MLflow в reports/mlflow/
```

Что получится:

| Результат | Где |
|---|---|
| сводная таблица экспериментов, выбранная модель | `reports/lab02-results.csv` |
| артефакты каждого запуска (конфигурация, параметры, метрики, прогнозы, графики, модель) | `runs/<имя>/` |
| итоговая модель | `models/model.joblib` |
| журнал MLflow | `mlflow.db` + `mlruns/` (локально, в git не входит) |
| таблицы и графики анализа ошибок | `reports/tables/`, `reports/figures/` |

Просмотр журнала MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`,
затем http://127.0.0.1:5000, эксперимент `fishgrow-lab02`.

**Быстрый сценарий для защиты** (≈ 1 с, без MLflow):

```bash
python -m src.train --config configs/experiments/e1_esharma3.yaml --no-mlflow
```

Ожидаемый вывод: `CV MSE 3288.2 ± 2058.2 | test MSE 1429.3 MAE 23.8 R2 0.9863`.

## 3. Прогноз для своих данных

```python
import joblib, pandas as pd
model = joblib.load("models/model.joblib")
fish = pd.DataFrame([{"Species": "Perch", "Length1": 25.0, "Length2": 27.0,
                      "Length3": 28.7, "Height": 7.5, "Width": 4.6}])
print(model.predict(fish))          # масса в граммах
```

## 4. Разбиение и воспроизводимость

- Тест заморожен до экспериментов: `python -m src.make_split` → `data/split.json` (88 / 23 строки, стратификация по виду, seed 42, индексы + SHA-256 данных). Повторный запуск не перезаписывает файл.
- Выбор моделей — только `RepeatedKFold` 5×3 внутри train; тест используется один раз в конце каждого запуска.
- Seed 42 для Python, NumPy и `random_state` моделей; версии библиотек — в `requirements-course.txt` и в тегах `env.*` каждого запуска; коммит — в теге `git.commit`.
- `fish_holdout_demo.csv` из исходного проекта не используется: все его строки содержатся в обучающем файле.

## 5. Структура

```
.
├── configs/
│   ├── base.yaml                # данные, разбиение, CV, правило выбора, MLflow
│   └── experiments/*.yaml       # E0–E4, по файлу на эксперимент
├── data/
│   ├── README.md                # происхождение, лицензия, паспорт данных, разбиение
│   ├── fish_participant.csv     # 111 строк, единственный источник
│   ├── fish_holdout_demo.csv    # из исходника, не используется
│   └── split.json               # замороженные индексы train/test
├── src/
│   ├── data.py  config.py  make_split.py
│   ├── pipeline.py              # ColumnTransformer + модель из конфигурации
│   ├── metrics.py               # MSE, MAE, R², VIF, тест Бройша — Пагана
│   ├── train.py                 # обучение, CV, подбор, MLflow
│   ├── evaluate.py              # анализ ошибок, проверка сохранённой модели
│   ├── tracking.py  plots.py  export_mlflow.py
├── tests/                       # test_data.py, test_pipeline.py
├── notebooks/02-analysis.ipynb  # только чтение результатов
├── reports/
│   ├── lab02-report.md          # отчёт
│   ├── lab02-results.csv        # таблица E0–E4
│   ├── lab02-baseline-repro.md  # воспроизведение исходника
│   ├── lab02-issue.md           # постановка задачи, критерии готовности, риски
│   ├── tables/  figures/  logs/  mlflow/
├── models/model-card.md         # паспорт модели
├── runs/                        # временные результаты запусков (не в git)
├── train.py  predict.py  preprocess.py  data_analysis_and_model_selection.ipynb  # исходный проект
├── requirements-course.txt  pyproject.toml
```

## 6. Исходный проект без изменений

```bash
python train.py      # обучает исходную модель на всём fish_participant.csv
python predict.py    # MSE 2330.62, R² 0.9750 — на строках, входящих в обучение
```

Разбор — `reports/lab02-baseline-repro.md`.

---

## Original README (esharma3)

# Fish Weight Prediction Model:

![Fish](images/Fishlength.jpg)

## Requirements

Hello! Welcome to the famous Tsukiji fish market of Tokyo, Japan! We came here to collect data on some of the fish they have here 
but we didn't wake up at 5am for the tuna auction and by the time we showed up they were only left with a few species of fish. 
We got to work and gathered measurements from a few different species of fish and want you to train a regression model to predict
the weight of a fish using some of the features we were able to measure. We have no idea which features will be good predictors. 

We will hold out 30% of the data before we hand it to you and we will use that csv for scoring.

Here's what we need from you:
1. A function that accepts a csv path and returns the predictions of your regression model using our csv. The csv we use will contain all the columns. 
2. Use a pipenv and scikit learn to submit the final model. You may use R for model selection. 

With this function and it's output, we will rank the students by how well their model performed on predicting weight based on naive data. 
Your grade will be determined by ranking according to Mean-squared Error.
If your function does not return a list of predictions or we cannot compute the accuracy of your model that it will be an automatic F. 

## Used

*   Linear Regression
*   Lasso Regression
*   Random Forest
*   Scikit Learn
*   Python
*   Joblib
*   Matplotlib
*   Pipenv
