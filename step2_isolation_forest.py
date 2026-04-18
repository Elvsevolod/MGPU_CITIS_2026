"""
Шаг 2: Обнаружение аномалий с помощью Isolation Forest.

Стратегия:
  - Записи, уже помеченные ручными правилами (шаг 1), полностью ИСКЛЮЧАЮТСЯ
    из датасета до начала любой обработки. Модель их вообще не видит.
  - На оставшихся «чистых» записях строятся признаки, обучается модель
    и производится скоринг.
  - Таким образом модель ищет только неочевидные аномалии среди тех записей,
    которые прошли все ручные проверки — без «шума» от уже известных ошибок.
  - contamination задаёт долю аномалий среди чистых записей (не от всего датасета).

Запуск:
    python3 step2_isolation_forest.py

Зависимости:
    pip install scikit-learn pandas numpy
"""

import csv
import os
from collections import defaultdict
from datetime import datetime, date

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────
DATA_FILE = "hakaton.csv"
MANUAL_FLAGS_FILE = "results_manual/manual_anomalies.csv"
OUT_DIR = "results_iforest"
os.makedirs(OUT_DIR, exist_ok=True)

# Гиперпараметры Isolation Forest
N_ESTIMATORS = 200       # количество деревьев — больше → стабильнее
MAX_SAMPLES = "auto"     # min(256, n_samples) — оптимально для этого размера
CONTAMINATION = 0.05     # ожидаемая доля аномалий среди «чистых» записей (5%)
RANDOM_STATE = 42

print("=" * 70)
print("ISOLATION FOREST — ПОИСК АНОМАЛИЙ СРЕДИ «ЧИСТЫХ» ЗАПИСЕЙ")
print("=" * 70)

# ─────────────────────────────────────────────
# Шаг 1: загрузка и фильтрация — убираем всё,
#         что уже нашли ручными правилами
# ─────────────────────────────────────────────
print("\n[1/5] Загрузка данных и исключение ручных аномалий...")

df_full = pd.read_csv(DATA_FILE, sep=";", dtype=str, keep_default_na=False)
print(f"  Всего записей в датасете: {len(df_full)}")

# Читаем номера записей, помеченных ручными правилами
flagged_numbers = set()
if os.path.exists(MANUAL_FLAGS_FILE):
    with open(MANUAL_FLAGS_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flagged_numbers.add(row["our_number"])
    print(f"  Помечено ручными правилами: {len(flagged_numbers)}")

# Полностью исключаем аномальные записи — модель их не увидит
df = df_full[~df_full["our_number"].isin(flagged_numbers)].copy().reset_index(drop=True)
print(f"  Осталось «чистых» записей для модели: {len(df)} "
      f"({len(df) / len(df_full) * 100:.1f}% от датасета)")

# ─────────────────────────────────────────────
# Шаг 2: построение признаков (только на чистых записях)
# ─────────────────────────────────────────────
print("\n[2/5] Построение признаков...")

def parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

df["_bdate"] = df["bdate"].map(parse_date)
df["_test_date"] = df["test_date"].map(parse_date)
df["_guard_bdate"] = df["guard_bdate"].map(parse_date)

features = pd.DataFrame(index=df.index)

# --- Возраст ребёнка на дату тестирования ---
features["age_at_test"] = df.apply(
    lambda r: (r["_test_date"] - r["_bdate"]).days / 365.25
    if r["_bdate"] and r["_test_date"] else np.nan,
    axis=1,
)

# --- Класс (числовой) ---
features["class"] = pd.to_numeric(df["class"], errors="coerce")

# --- Отклонение возраста от ожидаемого для данного класса ---
# Норма: класс N → возраст N+6.5 лет (середина диапазона 6–8 лет от класса)
features["age_class_dev"] = features["age_at_test"] - (features["class"] + 6.5)

# --- Разница возрастов между представителем и ребёнком ---
# Слишком маленькая разница — подозрение на ошибку в данных
features["guard_age_diff"] = df.apply(
    lambda r: (r["_bdate"] - r["_guard_bdate"]).days / 365.25
    if r["_bdate"] and r["_guard_bdate"] else np.nan,
    axis=1,
)

# --- Количество тестов у данного ребёнка в чистом датасете ---
# Пересчитывается заново после исключения аномальных записей —
# чтобы не было смещения из-за удалённых записей того же ребёнка
child_key = df["last_name"].str.upper() + "|" + df["first_name"].str.upper() + "|" + df["bdate"]
features["tests_per_child"] = child_key.map(child_key.value_counts())

# --- Дней с предыдущего теста (0 = первый тест у этого ребёнка) ---
# Пересчитывается по чистому датасету — удалённые аномальные тесты не влияют
child_date_map = defaultdict(list)
for idx, row in df.iterrows():
    if row["_test_date"]:
        key = (row["last_name"].upper(), row["first_name"].upper(), row["bdate"])
        child_date_map[key].append((row["_test_date"], idx))

days_since_prev = pd.Series(0.0, index=df.index)
for key, lst in child_date_map.items():
    lst_sorted = sorted(lst, key=lambda x: x[0])
    for j in range(1, len(lst_sorted)):
        prev_d, _ = lst_sorted[j - 1]
        curr_d, curr_idx = lst_sorted[j]
        days_since_prev[curr_idx] = (curr_d - prev_d).days

features["days_since_prev_test"] = days_since_prev

# --- Месяц тестирования (сезонность) ---
features["test_month"] = df["_test_date"].apply(lambda d: d.month if d else np.nan)

# --- Год тестирования ---
features["test_year"] = df["_test_date"].apply(lambda d: d.year if d else np.nan)

print(f"  Признаков: {features.shape[1]} → {list(features.columns)}")

# ─────────────────────────────────────────────
# Шаг 3: заполнение пропусков и масштабирование
# ─────────────────────────────────────────────
print("\n[3/5] Подготовка матрицы признаков...")

# Заполняем пропуски медианой — на основе самого же чистого датасета
medians = features.median()
X = features.fillna(medians)

# Стандартизация — приводим все признаки к одному масштабу
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Размер матрицы: {X_scaled.shape}")
print(f"  Пропусков до заполнения: {features.isna().sum().sum()}")

# ─────────────────────────────────────────────
# Шаг 4: обучение и скоринг Isolation Forest
# ─────────────────────────────────────────────
print("\n[4/5] Обучение и скоринг Isolation Forest...")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    contamination=CONTAMINATION,
    n_jobs=-1,          # все ядра CPU
    random_state=RANDOM_STATE,
)

# Обучаем и сразу скорим на тех же чистых данных
model.fit(X_scaled)

# anomaly_score: чем ниже — тем аномальнее (отрицательные = аномалия)
df["anomaly_score"] = model.decision_function(X_scaled)
# is_anomaly: 1 = аномалия, 0 = норма
df["is_anomaly_iforest"] = (model.predict(X_scaled) == -1).astype(int)

total_flagged = df["is_anomaly_iforest"].sum()
print(f"  Модель нашла аномалий: {total_flagged} из {len(df)} чистых записей "
      f"({total_flagged / len(df) * 100:.1f}%)")

# ─────────────────────────────────────────────
# Шаг 5: анализ и сохранение
# ─────────────────────────────────────────────
print("\n[5/5] Анализ и сохранение результатов...")

anomalies_df = df[df["is_anomaly_iforest"] == 1].copy()
anomalies_feat = pd.concat([anomalies_df, features.fillna(medians)[df["is_anomaly_iforest"] == 1]], axis=1)

# Топ-20 самых аномальных
print("\n  Топ-20 аномалий (самые низкие scores):")
display_cols = ["our_number", "last_name", "first_name", "bdate", "class",
                "test_date", "variant", "result", "anomaly_score"]
print(anomalies_df.sort_values("anomaly_score")[display_cols].head(20).to_string(index=False))

# Гистограмма распределения scores
print("\n  Распределение anomaly_score (чистые записи):")
bins = np.linspace(df["anomaly_score"].min(), df["anomaly_score"].max(), 11)
hist, edges = np.histogram(df["anomaly_score"], bins=bins)
for i in range(len(hist)):
    bar = "█" * (hist[i] // 100)
    print(f"  [{edges[i]:+.3f}, {edges[i+1]:+.3f}): {hist[i]:5d}  {bar}")

print(f"\n  Порог (score < 0 → аномалия):")
print(f"    score < 0:  {(df['anomaly_score'] < 0).sum()}")
print(f"    score >= 0: {(df['anomaly_score'] >= 0).sum()}")

# ─── Сохранение файлов ───
orig_cols = [c for c in df.columns if not c.startswith("_")]
score_cols = orig_cols + ["anomaly_score", "is_anomaly_iforest"]

# Все чистые записи со score
df[score_cols].to_csv(
    os.path.join(OUT_DIR, "iforest_clean_scored.csv"),
    index=False, sep=";", encoding="utf-8",
)

# Только аномалии, найденные моделью (отсортированы по аномальности)
feat_cols_out = list(features.columns)
anomalies_feat_out = anomalies_feat[[c for c in score_cols if c in anomalies_feat.columns] + feat_cols_out]
anomalies_feat_out = anomalies_feat_out.loc[:, ~anomalies_feat_out.columns.duplicated()]
anomalies_feat_out.sort_values("anomaly_score").to_csv(
    os.path.join(OUT_DIR, "iforest_anomalies.csv"),
    index=False, sep=";", encoding="utf-8",
)

print(f"\nФайлы сохранены в {OUT_DIR}/:")
print(f"  iforest_clean_scored.csv — все чистые записи с оценками модели")
print(f"  iforest_anomalies.csv    — аномалии, найденные Isolation Forest")

print("\n" + "=" * 70)
print("ИТОГ")
print("=" * 70)
print(f"  Всего записей в датасете:              {len(df_full)}")
print(f"  Исключено ручными правилами:           {len(flagged_numbers)}")
print(f"  Подано в модель («чистые»):            {len(df)}")
print(f"  Найдено аномалий Isolation Forest:     {total_flagged} "
      f"({total_flagged / len(df) * 100:.1f}% от чистых)")
