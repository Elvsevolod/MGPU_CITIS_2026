"""
Интерактивная визуализация эмбеддингов через Plotly.

Создаёт HTML-файлы которые открываются в браузере:
- При наведении на точку — имя ребёнка, дата, класс, школа, результат, тип аномалии
- Зум, панорамирование, фильтрация по легенде
- 2D и 3D версии
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

OUTPUT_DIR = "results_embeddings_full"

# ── Загрузка данных ──
print("Загрузка данных...")
coords_2d = np.load(f"{OUTPUT_DIR}/tsne_2d_full.npy")
coords_3d = np.load(f"{OUTPUT_DIR}/tsne_3d_full.npy")

df = pd.read_csv(f"{OUTPUT_DIR}/rows_full.csv", sep=";", dtype=str, keep_default_na=False)
manual_df = pd.read_csv("results_manual/manual_anomalies.csv", dtype=str)
iforest_df = pd.read_csv("results_iforest/iforest_anomalies.csv", sep=";", dtype=str)

manual_set  = set(manual_df["our_number"])
iforest_set = set(iforest_df["our_number"])
manual_cat  = manual_df.groupby("our_number")["category"].apply(
    lambda x: x.value_counts().index[0]
).to_dict()

CAT_RU = {
    "Норма":                  "Норма",
    "ML аномалия":            "ML аномалия (Isolation Forest)",
    "ЧАСТОТА":                "Нарушение частоты тестирования",
    "ДОКУМЕНТ_РЕБЁНОК":       "Отрицательный номер документа",
    "ВАРИАНТ_ФОРМАТ":         "Некорректный формат варианта",
    "ВОЗРАСТ_КЛАСС":          "Несоответствие возраста и класса",
    "РЕЗУЛЬТАТ_РЕГИСТР":      "Некорректный регистр результата",
    "ВАРИАНТ_КЛАСС":          "Вариант не соответствует классу",
    "Прочие ручные":          "Прочие нарушения",
}

COLOR_MAP = {
    "Норма":                           "#AAAAAA",
    "ML аномалия (Isolation Forest)":  "#9467BD",
    "Нарушение частоты тестирования":  "#D62728",
    "Отрицательный номер документа":   "#FF7F0E",
    "Некорректный формат варианта":    "#BCBD22",
    "Несоответствие возраста и класса":"#1F77B4",
    "Некорректный регистр результата": "#17BECF",
    "Вариант не соответствует классу": "#8C564B",
    "Прочие нарушения":                "#E377C2",
}

# Строим метки
labels, hover_texts = [], []
for _, row in df.iterrows():
    our = row["our_number"]
    child = f"{row['last_name']} {row['first_name']} {row['middle_name']}".strip()
    guard = f"{row['guard_last_name']} {row['guard_first_name']}".strip()
    info = (
        f"<b>{child}</b><br>"
        f"Дата рождения: {row['bdate']}<br>"
        f"Класс: {row['class']}  |  Вариант: {row['variant']}<br>"
        f"Дата теста: {row['test_date']}  |  Результат: {row['result']}<br>"
        f"Представитель: {guard}<br>"
        f"Школа: {row['name_naprav'][:60]}...<br>"
        f"Номер: {our}"
    )
    if our in manual_set:
        cat = manual_cat.get(our, "Прочие ручные")
        label = CAT_RU.get(cat, "Прочие нарушения")
    elif our in iforest_set:
        label = "ML аномалия (Isolation Forest)"
    else:
        label = "Норма"
    labels.append(label)
    hover_texts.append(info)

labels = np.array(labels)
print(f"Категорий: {len(set(labels))}")
for cat in sorted(set(labels)):
    print(f"  {cat}: {(labels==cat).sum()}")

# ════════════════════════════════════════════════════
# 2D — всё по категориям
# ════════════════════════════════════════════════════
print("\nСтроим 2D по категориям...")

# Порядок: сначала норма (под остальными), потом аномалии
ORDER = ["Норма"] + [c for c in COLOR_MAP if c != "Норма"]

fig_cat_2d = go.Figure()
for cat in ORDER:
    mask = labels == cat
    if not mask.any():
        continue
    is_norm = cat == "Норма"
    fig_cat_2d.add_trace(go.Scattergl(
        x=coords_2d[mask, 0],
        y=coords_2d[mask, 1],
        mode="markers",
        name=f"{cat} ({mask.sum():,})",
        marker=dict(
            color=COLOR_MAP.get(cat, "#888888"),
            size=4 if is_norm else 7,
            opacity=0.25 if is_norm else 0.80,
            line=dict(width=0),
        ),
        text=[hover_texts[i] for i in np.where(mask)[0]],
        hovertemplate="%{text}<extra></extra>",
    ))

fig_cat_2d.update_layout(
    title=dict(text="GigaChat Embeddings — t-SNE 2D<br><sub>Наведите на точку чтобы увидеть данные | n=25 628</sub>",
               font=dict(size=18)),
    xaxis_title="Компонента 1",
    yaxis_title="Компонента 2",
    plot_bgcolor="#F8F9FA",
    paper_bgcolor="#FFFFFF",
    legend=dict(itemsizing="constant", font=dict(size=12),
                bordercolor="#DDDDDD", borderwidth=1),
    width=1300, height=850,
    hovermode="closest",
)
path = f"{OUTPUT_DIR}/viz_plotly_tsne_2d_categories.html"
fig_cat_2d.write_html(path, include_plotlyjs="cdn")
print(f"  Сохранено: {path}")

# ════════════════════════════════════════════════════
# 2D — два цвета (норма/аномалия)
# ════════════════════════════════════════════════════
print("Строим 2D два цвета...")

is_anomaly = labels != "Норма"
fig_2col_2d = go.Figure()
fig_2col_2d.add_trace(go.Scattergl(
    x=coords_2d[~is_anomaly, 0], y=coords_2d[~is_anomaly, 1],
    mode="markers",
    name=f"Норма ({(~is_anomaly).sum():,})",
    marker=dict(color="#333333", size=3, opacity=0.30, line=dict(width=0)),
    text=[hover_texts[i] for i in np.where(~is_anomaly)[0]],
    hovertemplate="%{text}<extra></extra>",
))
fig_2col_2d.add_trace(go.Scattergl(
    x=coords_2d[is_anomaly, 0], y=coords_2d[is_anomaly, 1],
    mode="markers",
    name=f"Аномалия ({is_anomaly.sum():,})",
    marker=dict(color="#E63946", size=6, opacity=0.80, line=dict(width=0)),
    text=[hover_texts[i] for i in np.where(is_anomaly)[0]],
    hovertemplate="%{text}<extra></extra>",
))
fig_2col_2d.update_layout(
    title=dict(text="GigaChat Embeddings — t-SNE 2D (аномалии vs норма)<br><sub>n=25 628 | наведите на точку</sub>",
               font=dict(size=18)),
    xaxis_title="Компонента 1", yaxis_title="Компонента 2",
    plot_bgcolor="#F8F9FA", paper_bgcolor="#FFFFFF",
    legend=dict(font=dict(size=13), bordercolor="#DDDDDD", borderwidth=1),
    width=1300, height=850, hovermode="closest",
)
path = f"{OUTPUT_DIR}/viz_plotly_tsne_2d_2color.html"
fig_2col_2d.write_html(path, include_plotlyjs="cdn")
print(f"  Сохранено: {path}")

# ════════════════════════════════════════════════════
# 3D — по категориям
# ════════════════════════════════════════════════════
print("Строим 3D по категориям...")

fig_cat_3d = go.Figure()
for cat in ORDER:
    mask = labels == cat
    if not mask.any():
        continue
    is_norm = cat == "Норма"
    fig_cat_3d.add_trace(go.Scatter3d(
        x=coords_3d[mask, 0],
        y=coords_3d[mask, 1],
        z=coords_3d[mask, 2],
        mode="markers",
        name=f"{cat} ({mask.sum():,})",
        marker=dict(
            color=COLOR_MAP.get(cat, "#888888"),
            size=2 if is_norm else 4,
            opacity=0.15 if is_norm else 0.75,
            line=dict(width=0),
        ),
        text=[hover_texts[i] for i in np.where(mask)[0]],
        hovertemplate="%{text}<extra></extra>",
    ))

fig_cat_3d.update_layout(
    title=dict(text="GigaChat Embeddings — t-SNE 3D<br><sub>Вращайте мышью | n=25 628</sub>",
               font=dict(size=18)),
    scene=dict(
        xaxis_title="Ось 1", yaxis_title="Ось 2", zaxis_title="Ось 3",
        bgcolor="#F0F0F0",
        xaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#DDDDDD"),
        yaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#DDDDDD"),
        zaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#DDDDDD"),
    ),
    paper_bgcolor="#FFFFFF",
    legend=dict(itemsizing="constant", font=dict(size=11),
                bordercolor="#DDDDDD", borderwidth=1),
    width=1300, height=900,
)
path = f"{OUTPUT_DIR}/viz_plotly_tsne_3d_categories.html"
fig_cat_3d.write_html(path, include_plotlyjs="cdn")
print(f"  Сохранено: {path}")

# ════════════════════════════════════════════════════
# 3D — два цвета
# ════════════════════════════════════════════════════
print("Строим 3D два цвета...")

fig_2col_3d = go.Figure()
fig_2col_3d.add_trace(go.Scatter3d(
    x=coords_3d[~is_anomaly, 0], y=coords_3d[~is_anomaly, 1], z=coords_3d[~is_anomaly, 2],
    mode="markers", name=f"Норма ({(~is_anomaly).sum():,})",
    marker=dict(color="#333333", size=2, opacity=0.20, line=dict(width=0)),
    text=[hover_texts[i] for i in np.where(~is_anomaly)[0]],
    hovertemplate="%{text}<extra></extra>",
))
fig_2col_3d.add_trace(go.Scatter3d(
    x=coords_3d[is_anomaly, 0], y=coords_3d[is_anomaly, 1], z=coords_3d[is_anomaly, 2],
    mode="markers", name=f"Аномалия ({is_anomaly.sum():,})",
    marker=dict(color="#E63946", size=4, opacity=0.75, line=dict(width=0)),
    text=[hover_texts[i] for i in np.where(is_anomaly)[0]],
    hovertemplate="%{text}<extra></extra>",
))
fig_2col_3d.update_layout(
    title=dict(text="GigaChat Embeddings — t-SNE 3D (аномалии vs норма)<br><sub>Вращайте мышью | n=25 628</sub>",
               font=dict(size=18)),
    scene=dict(
        xaxis_title="Ось 1", yaxis_title="Ось 2", zaxis_title="Ось 3",
        bgcolor="#F0F0F0",
    ),
    paper_bgcolor="#FFFFFF",
    legend=dict(font=dict(size=13), bordercolor="#DDDDDD", borderwidth=1),
    width=1300, height=900,
)
path = f"{OUTPUT_DIR}/viz_plotly_tsne_3d_2color.html"
fig_2col_3d.write_html(path, include_plotlyjs="cdn")
print(f"  Сохранено: {path}")

# ════════════════════════════════════════════════════
print("\n✓ Готово! Все файлы в", OUTPUT_DIR)
print("  Открывай .html файлы в браузере:")
for fn in [
    "viz_plotly_tsne_2d_categories.html",
    "viz_plotly_tsne_2d_2color.html",
    "viz_plotly_tsne_3d_categories.html",
    "viz_plotly_tsne_3d_2color.html",
]:
    p = f"{OUTPUT_DIR}/{fn}"
    if os.path.exists(p):
        print(f"  open {p}")
