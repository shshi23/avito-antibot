# Avito Antibot (детекция ботов-парсеров)

`Python 3.12` | `CatBoost` | `scikit-learn` | `pandas` | `numpy` | `matplotlib`

Решение тестового задания команды Антибота: по событиям внутри суточного окна наблюдения предсказать для каждой `cookie_id` вероятность того, что она принадлежит сервису автоматизированного сбора данных.

### **Задача**
- - -

- **Вход:** `train.csv`, `test.csv` (по одной строке на `cookie_id`), `events.csv.gz` $-$ история событий куки.

- **Выход:** `submission.csv` с колонками `cookie_id`, `score`.

- **Метрика:** `P@R≥0.7 = max_{t: Recall(t) ≥ 0.7} Precision(t)` $-$ максимум precision среди порогов, при которых модель находит ≥ 70% ботов.


### **Пайплайн**
- - -

1. **Фильтрация событий по окну наблюдения** — берём только события с `window_start_ts ≤ event_ts ≤ window_end_ts`. Всё, что было до/после, отбрасывается, чтобы не допустить утечки будущего.

2. **Сборка признаков на уровне `cookie_id`** $-$ 61 фича.

3. **Обучение CatBoost** с `auto_class_weights="Balanced"`.

4. **Сохранение модели** и **инференс** на тесте.

### **Блоки признаков**
- - -


| Блок | Примеры | Что ловит |
|---|---|---|
| Активность | `event_count`, `unique_items`, `unique_categories`, `unique_locations` | интенсивность работы куки |
| Интервалы между событиями | `median_interval`, `iqr_interval`, `p10/p90_interval`, `ratio_gt_60s`, `p90_over_p10` | ритмичность и паузы в сессии |
| Pointer | `ptr_mean_step`, `ptr_std_x/y`, `ptr_range_x/y` | характер движения курсора |
| Transition (n-граммы) | `bigrams_per_transition`, `trigrams_per_transition`, `unique_trigrams` | цикличность последовательности событий |
| Нормализованные счётчики | `items_per_event`, `categories_per_event`, `locations_per_event` | разнообразие на единицу активности |
| Платформа / UA | `platform_*_ratio`, `ratio_is_bot_ua`, `ua_length_median` | тип клиента |
| Seller | `seller_private_ratio`, `seller_pro_ratio`, `seller_unknown_ratio` | тип продавца в объявлениях |
| Event types | `ratio_*`, `has_login`, `has_favorite_add` | профиль действий |
| Search | `search_query_ratio`, `unique_search_queries`, `max_search_page` | работа с поисковой выдачей |
| Возраст куки | `cookie_age_log` | как давно кука в системе |

### **Ключевые фичи**
- - -

По важности CatBoost на финальной модели:

1. `median_interval` $-$ медианный интервал между событиями.

2. `ptr_mean_step` $-$ средний шаг курсора между последовательными событиями.

3. `ratio_gt_60s` $-$ доля пауз длиннее 60 секунд.

4. `ptr_std_x` $-$ стандартное отклонение координаты X курсора.

5. `ptr_range_x` $-$ размах координаты X курсора.

6. `unique_categories` $-$ число категорий.

7. `iqr_over_median` $-$ нормированный межквартильный размах интервалов.

8. `items_per_event` $-$ уникальные объявления на событие.

### **Результаты**
- - -


| Модель | P@R≥0.7 (baseline) | P@R≥0.7 (final) |
|---|---|---|
| CatBoost | 0.546 | 0.750 |
| RandomForest | 0.472 | 0.706 |
| LogReg | 0.313 | 0.531 |
| Constant baseline | 0.081 | — |

Прирост **+0.204 абсолютных (+37% относительно baseline)**.

### **Финальная модель (CatBoost)**
- - -


| Метрика | Значение |
|---|---|
| **P@R≥0.7** | **0.750** |
| PR-AUC | 0.774 |
| ROC-AUC | 0.934 |

На скрытой тестовой выборке: **P@R≥0.7 = 0.754**.

### Запуск
- - - 

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.train_model

python -m src.predict
```

### **Гиперпараметры CatBoost**
- - - 

```python
CatBoostClassifier(
    iterations=700,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    random_seed=42,
)