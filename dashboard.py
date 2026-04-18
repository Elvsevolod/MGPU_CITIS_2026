"""
Дашборд анализа аномалий тестирования детей по истории.

Запуск:
    streamlit run dashboard.py
"""

import os
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Аномалии тестирования · ЦИТиС 2026",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Загрузка данных (кэшируем чтобы не перечитывать при каждом взаимодействии)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    full = pd.read_csv("hakaton.csv", sep=";", dtype=str, keep_default_na=False)
    manual = pd.read_csv("results_manual/manual_anomalies.csv", dtype=str)
    summary = pd.read_csv("results_manual/manual_anomalies_summary.csv")
    iforest = pd.read_csv("results_iforest/iforest_anomalies.csv", sep=";", dtype=str)
    clean_scored = pd.read_csv("results_iforest/iforest_clean_scored.csv", sep=";", dtype=str)

    # Нормализация типов
    full["test_date"] = pd.to_datetime(full["test_date"], errors="coerce")
    full["bdate"] = pd.to_datetime(full["bdate"], errors="coerce")
    full["class_int"] = pd.to_numeric(full["class"], errors="coerce")
    full["test_month"] = full["test_date"].dt.month
    full["test_year"] = full["test_date"].dt.year

    iforest["anomaly_score"] = pd.to_numeric(iforest["anomaly_score"], errors="coerce")
    iforest["class_int"] = pd.to_numeric(iforest["class"], errors="coerce")
    iforest["test_date"] = pd.to_datetime(iforest["test_date"], errors="coerce")
    iforest["test_month"] = iforest["test_date"].dt.month

    clean_scored["anomaly_score"] = pd.to_numeric(clean_scored["anomaly_score"], errors="coerce")
    clean_scored["class_int"] = pd.to_numeric(clean_scored["class"], errors="coerce")
    clean_scored["test_date"] = pd.to_datetime(clean_scored["test_date"], errors="coerce")

    summary["count"] = summary["count"].astype(int)
    return full, manual, summary, iforest, clean_scored

full, manual, summary, iforest, clean_scored = load_data()

TOTAL = len(full)
MANUAL_UNIQUE = manual["our_number"].nunique()
IFOREST_COUNT = len(iforest)
CLEAN_COUNT = len(clean_scored)

CATEGORY_RU = {
    "РЕЗУЛЬТАТ_РЕГИСТР":       "Некорректный регистр результата",
    "ДОКУМЕНТ_РЕБЁНОК":        "Отрицательный номер документа (ребёнок)",
    "ВАРИАНТ_ФОРМАТ":          "Некорректный формат варианта",
    "ЧАСТОТА":                 "Нарушение частоты тестирования",
    "ВОЗРАСТ_КЛАСС":           "Несоответствие возраста и класса",
    "ВАРИАНТ_КЛАСС":           "Вариант не соответствует классу",
    "ПУСТОЕ_ПОЛЕ":             "Пустые обязательные поля",
    "ПРЕДСТАВИТЕЛЬ_ВОЗРАСТ":   "Слишком молодой представитель",
    "ДОКУМЕНТ_ПРЕДСТАВИТЕЛЬ":  "Отрицательный номер документа (представитель)",
    "ДАТА_РОЖДЕНИЯ":           "Аномальная дата рождения",
    "КЛАСС_ФОРМАТ":            "Некорректный формат класса",
    "ОГРН_НАПРАВИВШАЯ":        "Некорректный ОГРН школы",
}
MONTH_RU = {1:"Янв",2:"Фев",3:"Мар",4:"Апр",5:"Май",6:"Июн",
            7:"Июл",8:"Авг",9:"Сен",10:"Окт",11:"Ноя",12:"Дек"}

# ─────────────────────────────────────────────────────────────
# Боковая панель навигации
# ─────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Навигация")
page = st.sidebar.radio(
    "Раздел",
    ["📊 Обзор", "✋ Ручные аномалии", "📅 Частота тестирований", "🤖 ML (Isolation Forest)", "📋 Таблицы"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Хакатон ЦИТиС 2026 · Анализ соблюдения рекомендаций тестирования")

# ═════════════════════════════════════════════════════════════
# СТРАНИЦА 1: ОБЗОР
# ═════════════════════════════════════════════════════════════
if page == "📊 Обзор":
    st.title("📊 Обзор результатов анализа")
    st.markdown("Анализ датасета результатов тестирования детей по истории. "
                "Цель — выявить нарушения рекомендаций и аномалии в данных.")

    # ── Ключевые метрики ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всего записей", f"{TOTAL:,}".replace(",", " "))
    c2.metric("Ручных аномалий", f"{MANUAL_UNIQUE:,}".replace(",", " "),
              delta=f"{MANUAL_UNIQUE/TOTAL*100:.1f}%", delta_color="inverse")
    c3.metric("Чистых записей", f"{CLEAN_COUNT:,}".replace(",", " "),
              delta=f"{CLEAN_COUNT/TOTAL*100:.1f}%", delta_color="normal")
    c4.metric("ML аномалий", f"{IFOREST_COUNT:,}".replace(",", " "),
              delta=f"{IFOREST_COUNT/CLEAN_COUNT*100:.1f}% от чистых", delta_color="inverse")
    c5.metric("Всего нарушений", f"{MANUAL_UNIQUE + IFOREST_COUNT:,}".replace(",", " "),
              delta=f"{(MANUAL_UNIQUE + IFOREST_COUNT)/TOTAL*100:.1f}% датасета", delta_color="inverse")

    st.markdown("---")

    # ── Воронка обработки ──
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Воронка обработки данных")
        funnel_data = pd.DataFrame({
            "Этап": ["Весь датасет", "Ручные аномалии", "Чистые записи", "ML аномалии"],
            "Количество": [TOTAL, MANUAL_UNIQUE, CLEAN_COUNT, IFOREST_COUNT],
            "Цвет": ["#4C78A8", "#E45756", "#54A24B", "#F58518"],
        })
        bar = alt.Chart(funnel_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Этап:N", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Количество:Q"),
            color=alt.Color("Цвет:N", scale=None, legend=None),
            tooltip=["Этап", "Количество"],
        ).properties(height=320)
        text = bar.mark_text(dy=-8, fontSize=13, fontWeight="bold").encode(
            text=alt.Text("Количество:Q", format=",")
        )
        st.altair_chart(bar + text, use_container_width=True)

    with col_r:
        st.subheader("Доля аномалий в датасете")
        pie_data = pd.DataFrame({
            "Категория": ["Ручные аномалии", "ML аномалии", "Норма"],
            "Количество": [MANUAL_UNIQUE, IFOREST_COUNT, TOTAL - MANUAL_UNIQUE - IFOREST_COUNT],
        })
        pie = alt.Chart(pie_data).mark_arc(innerRadius=60).encode(
            theta=alt.Theta("Количество:Q"),
            color=alt.Color("Категория:N", scale=alt.Scale(
                domain=["Ручные аномалии", "ML аномалии", "Норма"],
                range=["#E45756", "#F58518", "#54A24B"]
            )),
            tooltip=["Категория", "Количество"],
        ).properties(height=320)
        st.altair_chart(pie, use_container_width=True)

    st.markdown("---")

    # ── Распределение по классам ──
    st.subheader("Распределение записей по классам")
    class_counts = full["class_int"].dropna().astype(int).value_counts().sort_index().reset_index()
    class_counts.columns = ["Класс", "Всего"]
    iforest_class = iforest["class_int"].dropna().astype(int).value_counts().sort_index().reset_index()
    iforest_class.columns = ["Класс", "ML аномалий"]
    class_merged = class_counts.merge(iforest_class, on="Класс", how="left").fillna(0)
    class_merged["ML аномалий"] = class_merged["ML аномалий"].astype(int)
    class_merged["Доля ML (%)"] = (class_merged["ML аномалий"] / class_merged["Всего"] * 100).round(1)

    base = alt.Chart(class_merged)
    bars = base.mark_bar(color="#4C78A8", opacity=0.8).encode(
        x=alt.X("Класс:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Всего:Q", title="Кол-во записей"),
        tooltip=["Класс", "Всего", "ML аномалий", "Доля ML (%)"],
    )
    line = base.mark_line(color="#E45756", strokeWidth=2, point=True).encode(
        x="Класс:O",
        y=alt.Y("ML аномалий:Q", title="ML аномалий", axis=alt.Axis(titleColor="#E45756")),
    )
    chart = alt.layer(bars, line).resolve_scale(y="independent").properties(height=280)
    st.altair_chart(chart, use_container_width=True)

    # ── Сезонность ──
    st.subheader("Сезонность тестирований")
    month_counts = full["test_month"].dropna().astype(int).value_counts().sort_index().reset_index()
    month_counts.columns = ["Месяц", "Всего"]
    iforest_month = iforest["test_month"].dropna().astype(int).value_counts().sort_index().reset_index()
    iforest_month.columns = ["Месяц", "ML аномалий"]
    month_merged = month_counts.merge(iforest_month, on="Месяц", how="left").fillna(0)
    month_merged["Месяц_str"] = month_merged["Месяц"].map(MONTH_RU)
    month_merged["ML аномалий"] = month_merged["ML аномалий"].astype(int)

    m_base = alt.Chart(month_merged)
    m_bars = m_base.mark_bar(color="#4C78A8", opacity=0.7).encode(
        x=alt.X("Месяц_str:N", sort=list(MONTH_RU.values()), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Всего:Q"),
        tooltip=["Месяц_str", "Всего", "ML аномалий"],
    )
    m_line = m_base.mark_line(color="#F58518", strokeWidth=2.5, point=True).encode(
        x=alt.X("Месяц_str:N", sort=list(MONTH_RU.values())),
        y=alt.Y("ML аномалий:Q", axis=alt.Axis(titleColor="#F58518")),
    )
    m_chart = alt.layer(m_bars, m_line).resolve_scale(y="independent").properties(height=260)
    st.altair_chart(m_chart, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# СТРАНИЦА 2: РУЧНЫЕ АНОМАЛИИ
# ═════════════════════════════════════════════════════════════
elif page == "✋ Ручные аномалии":
    st.title("✋ Ручные аномалии")
    st.markdown(f"Обнаружено **{MANUAL_UNIQUE:,}** записей с нарушениями по 10 категориям правил "
                f"({MANUAL_UNIQUE/TOTAL*100:.1f}% датасета).")

    # ── Сводка по категориям ──
    st.subheader("Количество нарушений по категориям")
    sum_display = summary.copy()
    sum_display["Описание"] = sum_display["category"].map(CATEGORY_RU).fillna(sum_display["category"])
    sum_display = sum_display.sort_values("count", ascending=False)

    bar_chart = alt.Chart(sum_display).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("Описание:N", sort="-x", axis=alt.Axis(labelLimit=400)),
        x=alt.X("count:Q", title="Количество записей"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["Описание", alt.Tooltip("count:Q", title="Кол-во")],
    ).properties(height=380)
    text_bar = bar_chart.mark_text(align="left", dx=4, fontSize=12).encode(
        text=alt.Text("count:Q", format=",")
    )
    st.altair_chart(bar_chart + text_bar, use_container_width=True)

    st.markdown("---")

    # ── Детали по выбранной категории ──
    st.subheader("Детализация по категории")
    selected_cat = st.selectbox(
        "Выберите категорию",
        options=list(CATEGORY_RU.keys()),
        format_func=lambda k: f"{k} — {CATEGORY_RU[k]}",
    )
    cat_records = manual[manual["category"] == selected_cat].copy()
    st.markdown(f"**Записей в категории:** {len(cat_records)}")

    show_cols = ["our_number", "child", "description"]
    show_n = st.slider("Показать записей", 10, min(500, len(cat_records)), min(50, len(cat_records)), 10)
    st.dataframe(cat_records[show_cols].head(show_n), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Топ записей с наибольшим числом нарушений ──
    st.subheader("Записи с наибольшим числом одновременных нарушений")
    multi_flags = manual.groupby("our_number").agg(
        нарушений=("category", "count"),
        категории=("category", lambda x: ", ".join(sorted(set(x)))),
        ребёнок=("child", "first"),
    ).reset_index().sort_values("нарушений", ascending=False).head(20)
    st.dataframe(multi_flags, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# СТРАНИЦА 3: ЧАСТОТА ТЕСТИРОВАНИЙ
# ═════════════════════════════════════════════════════════════
elif page == "📅 Частота тестирований":
    st.title("📅 Нарушения частоты тестирования")
    st.markdown(
        "Рекомендация: проходить тестирование **не чаще 1 раза в 3 месяца (90 дней)**. "
        "Ниже — анализ детей, у которых интервал между тестами был меньше нормы."
    )

    freq_df = manual[manual["category"] == "ЧАСТОТА"].copy()
    st.metric("Зафиксировано нарушений частоты", len(freq_df))

    # Извлекаем интервал из описания
    import re
    def extract_days(desc):
        m = re.search(r"через (\d+) дн\.", desc)
        return int(m.group(1)) if m else None

    freq_df["дней_между"] = freq_df["description"].apply(extract_days)
    freq_df_clean = freq_df.dropna(subset=["дней_между"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение интервалов (дни)")
        hist_data = pd.cut(
            freq_df_clean["дней_между"],
            bins=[0, 14, 30, 60, 89],
            labels=["0–14 дн.", "15–30 дн.", "31–60 дн.", "61–89 дн."],
        ).value_counts().reset_index()
        hist_data.columns = ["Диапазон", "Количество"]
        hist_chart = alt.Chart(hist_data).mark_bar(color="#E45756", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Диапазон:N", sort=["0–14 дн.", "15–30 дн.", "31–60 дн.", "61–89 дн."],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Количество:Q"),
            tooltip=["Диапазон", "Количество"],
        ).properties(height=280)
        t = hist_chart.mark_text(dy=-8, fontSize=12, fontWeight="bold").encode(text="Количество:Q")
        st.altair_chart(hist_chart + t, use_container_width=True)

    with col2:
        st.subheader("Статистика интервалов")
        stats = freq_df_clean["дней_между"].describe().rename({
            "count": "Кол-во нарушений", "mean": "Среднее (дн.)",
            "std": "Стд. откл.", "min": "Минимум", "25%": "Q1",
            "50%": "Медиана", "75%": "Q3", "max": "Максимум",
        })
        st.dataframe(stats.round(1).reset_index().rename(columns={"index": "Метрика", 0: "Значение"}),
                     use_container_width=True, hide_index=True)
        st.info(f"🔴 Самый короткий интервал: **{int(freq_df_clean['дней_между'].min())} дней**")
        st.info(f"🟡 Медианный интервал: **{int(freq_df_clean['дней_между'].median())} дней**")

    st.markdown("---")
    st.subheader("Примеры нарушений")
    st.dataframe(
        freq_df[["our_number", "child", "description"]].head(50),
        use_container_width=True, hide_index=True,
    )


# ═════════════════════════════════════════════════════════════
# СТРАНИЦА 4: ML ISOLATION FOREST
# ═════════════════════════════════════════════════════════════
elif page == "🤖 ML (Isolation Forest)":
    st.title("🤖 Isolation Forest — неочевидные аномалии")
    st.markdown(
        "Модель обучена на **чистых записях** (после исключения всех ручных аномалий). "
        "Она ищет записи, которые статистически выбиваются из общего профиля, "
        "даже если формально не нарушают ни одно правило."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Подано в модель", f"{CLEAN_COUNT:,}".replace(",", " "), "чистых записей")
    c2.metric("Найдено аномалий", f"{IFOREST_COUNT:,}".replace(",", " "),
              f"{IFOREST_COUNT/CLEAN_COUNT*100:.1f}%", delta_color="inverse")
    c3.metric("Порог contamination", "5%", "параметр модели")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение anomaly score")
        score_hist = clean_scored["anomaly_score"].dropna()
        bins_arr = np.linspace(score_hist.min(), score_hist.max(), 30)
        hist_vals, edges = np.histogram(score_hist, bins=bins_arr)
        score_df = pd.DataFrame({
            "score": (edges[:-1] + edges[1:]) / 2,
            "count": hist_vals,
            "аномалия": (edges[:-1] + edges[1:]) / 2 < 0,
        })
        score_chart = alt.Chart(score_df).mark_bar(binSpacing=1).encode(
            x=alt.X("score:Q", title="Anomaly Score"),
            y=alt.Y("count:Q", title="Количество"),
            color=alt.Color("аномалия:N", scale=alt.Scale(
                domain=[True, False], range=["#E45756", "#4C78A8"]
            ), legend=alt.Legend(title="Аномалия")),
            tooltip=["score", "count"],
        ).properties(height=300)
        st.altair_chart(score_chart, use_container_width=True)
        st.caption("🔴 Score < 0 — аномальные записи | 🔵 Score ≥ 0 — нормальные")

    with col2:
        st.subheader("Доля аномалий по классам")
        all_class = clean_scored["class_int"].dropna().astype(int).value_counts().sort_index().reset_index()
        all_class.columns = ["Класс", "Всего"]
        anom_class = iforest["class_int"].dropna().astype(int).value_counts().sort_index().reset_index()
        anom_class.columns = ["Класс", "Аномалий"]
        class_m = all_class.merge(anom_class, on="Класс", how="left").fillna(0)
        class_m["Аномалий"] = class_m["Аномалий"].astype(int)
        class_m["Доля (%)"] = (class_m["Аномалий"] / class_m["Всего"] * 100).round(1)

        class_chart = alt.Chart(class_m).mark_bar(
            color="#F58518", cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("Класс:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Доля (%):Q", title="% аномалий в классе"),
            tooltip=["Класс", "Всего", "Аномалий", "Доля (%)"],
        ).properties(height=300)
        t = class_chart.mark_text(dy=-8, fontSize=11).encode(
            text=alt.Text("Доля (%):Q", format=".1f")
        )
        st.altair_chart(class_chart + t, use_container_width=True)
        st.caption("Старшие классы (8–11) аномализируются чаще из-за малочисленности выборки")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Количество тестов у аномальных детей")
        iforest_feat = pd.read_csv("results_iforest/iforest_anomalies.csv", sep=";", dtype=str)
        iforest_feat["tests_per_child"] = pd.to_numeric(iforest_feat.get("tests_per_child", pd.Series(dtype=str)), errors="coerce")
        if "tests_per_child" in iforest_feat.columns and iforest_feat["tests_per_child"].notna().any():
            tpc = iforest_feat["tests_per_child"].value_counts().sort_index().reset_index()
            tpc.columns = ["Тестов", "Записей"]
            tpc_chart = alt.Chart(tpc).mark_bar(color="#9C755F", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Тестов:O", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Записей:Q"),
                tooltip=["Тестов", "Записей"],
            ).properties(height=280)
            t2 = tpc_chart.mark_text(dy=-8, fontSize=12, fontWeight="bold").encode(text="Записей:Q")
            st.altair_chart(tpc_chart + t2, use_container_width=True)
        else:
            st.info("Признак tests_per_child недоступен в файле результатов")

    with col4:
        st.subheader("Сезонность ML аномалий")
        m_anom = iforest["test_month"].dropna().astype(int).value_counts().sort_index().reset_index()
        m_anom.columns = ["Месяц", "Аномалий"]
        m_norm = clean_scored[clean_scored["is_anomaly_iforest"] == "0"]["test_date"].dt.month.dropna().astype(int)\
            .value_counts().sort_index().reset_index()
        m_norm.columns = ["Месяц", "Норма"]
        m_merged = m_norm.merge(m_anom, on="Месяц", how="outer").fillna(0)
        m_merged["Месяц_str"] = m_merged["Месяц"].astype(int).map(MONTH_RU)
        m_merged = m_merged.sort_values("Месяц")
        m_melted = m_merged.melt(id_vars=["Месяц_str"], value_vars=["Норма", "Аномалий"],
                                  var_name="Тип", value_name="Количество")
        season_chart = alt.Chart(m_melted).mark_bar(opacity=0.85).encode(
            x=alt.X("Месяц_str:N", sort=list(MONTH_RU.values()), axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Количество:Q"),
            color=alt.Color("Тип:N", scale=alt.Scale(
                domain=["Норма", "Аномалий"], range=["#4C78A8", "#E45756"]
            )),
            tooltip=["Месяц_str", "Тип", "Количество"],
        ).properties(height=280)
        st.altair_chart(season_chart, use_container_width=True)
        st.caption("Январь–март: аномалий непропорционально много относительно общего потока")

    st.markdown("---")

    st.subheader("Топ-50 аномалий по версии модели")
    display_cols = ["our_number", "last_name", "first_name", "bdate",
                    "class", "test_date", "variant", "result", "anomaly_score"]
    available = [c for c in display_cols if c in iforest.columns]
    top50 = iforest.sort_values("anomaly_score")[available].head(50)
    st.dataframe(top50, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# СТРАНИЦА 5: ТАБЛИЦЫ
# ═════════════════════════════════════════════════════════════
elif page == "📋 Таблицы":
    st.title("📋 Таблицы данных")

    tab1, tab2, tab3 = st.tabs(["Все ручные аномалии", "ML аномалии", "Чистые записи (со score)"])

    with tab1:
        st.subheader("Все ручные аномалии")
        cats = ["Все"] + list(CATEGORY_RU.keys())
        filt = st.selectbox("Фильтр по категории", cats,
                            format_func=lambda k: "Все категории" if k == "Все" else f"{k} — {CATEGORY_RU.get(k,k)}")
        data_show = manual if filt == "Все" else manual[manual["category"] == filt]
        st.caption(f"Показано: {len(data_show)} записей")
        st.dataframe(data_show[["index", "our_number", "child", "category", "description"]],
                     use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Аномалии Isolation Forest (800 записей)")
        show_cols_ml = ["our_number", "last_name", "first_name", "bdate",
                        "class", "test_date", "name_naprav", "result", "anomaly_score"]
        avail = [c for c in show_cols_ml if c in iforest.columns]
        score_min = float(iforest["anomaly_score"].min())
        score_max = float(iforest["anomaly_score"].max())
        score_filter = st.slider("Фильтр по anomaly_score", score_min, score_max,
                                 (score_min, -0.02), step=0.001, format="%.3f")
        filtered_ml = iforest[
            (iforest["anomaly_score"] >= score_filter[0]) &
            (iforest["anomaly_score"] <= score_filter[1])
        ]
        st.caption(f"Показано: {len(filtered_ml)} записей")
        st.dataframe(filtered_ml[avail].sort_values("anomaly_score"), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Чистые записи с оценкой модели")
        n_show = st.slider("Показать записей", 50, 500, 100, 50)
        sort_by = st.radio("Сортировка", ["По аномальности (сначала подозрительные)",
                                          "По нормальности (сначала обычные)"], horizontal=True)
        asc = sort_by.startswith("По нормальности")
        sc_cols = ["our_number", "last_name", "first_name", "bdate",
                   "class", "test_date", "result", "anomaly_score", "is_anomaly_iforest"]
        avail_sc = [c for c in sc_cols if c in clean_scored.columns]
        st.dataframe(clean_scored[avail_sc].sort_values("anomaly_score", ascending=not asc).head(n_show),
                     use_container_width=True, hide_index=True)
