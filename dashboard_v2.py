"""
Streamlit-дашборд v2: результаты full_analysis.ipynb + карта РФ.
Запуск: STREAMLIT_BROWSER_GATHER_USAGE_STATS=false python3 -m streamlit run dashboard_v2.py --server.headless true
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ─── Страница ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ЦИТиС — Анализ тестирований",
    page_icon="medium_a975a7ab34479e1d158d6239e5a41963.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Глобальные стили + анимации ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Шапка */
.dash-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    color: white;
    animation: slideDown 0.6s ease;
}
.dash-header h1 { margin:0; font-size:28px; font-weight:700; letter-spacing:-0.5px; }
.dash-header p  { margin:6px 0 0; opacity:.7; font-size:14px; }

/* Метрика-карточка */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    border-left: 5px solid;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    animation: fadeInUp 0.5s ease backwards;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.12);
}
.metric-card .label { font-size: 12px; font-weight:600; text-transform:uppercase; letter-spacing:.6px; opacity:.6; }
.metric-card .value { font-size: 32px; font-weight: 800; margin: 4px 0 2px; line-height:1; }
.metric-card .delta { font-size: 12px; opacity:.55; }

/* Разделитель-заголовок секции */
.section-title {
    font-size: 18px; font-weight: 700;
    border-bottom: 2px solid #f0f2f6;
    padding-bottom: 8px;
    margin: 24px 0 16px;
    color: #1a1a2e;
}

/* Анимации */
@keyframes fadeInUp   { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
@keyframes slideDown  { from { opacity:0; transform:translateY(-15px); } to { opacity:1; transform:translateY(0); } }
@keyframes pulseDot   { 0%,100% { transform:scale(1); } 50% { transform:scale(1.3); } }

/* Статус-пилюли */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
}
.pill-red    { background:#fde8e8; color:#c0392b; }
.pill-yellow { background:#fef9e7; color:#d68910; }
.pill-green  { background:#e8f8f5; color:#1e8449; }
.pill-blue   { background:#eaf4fd; color:#1a5276; }

/* Карточка инсайта */
.insight-card {
    background: linear-gradient(135deg, #f8f9fa, #fff);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #e9ecef;
    margin-bottom: 12px;
    animation: fadeInUp 0.4s ease;
}
.insight-card .icon { font-size: 24px; margin-bottom: 8px; }
.insight-card .title { font-size: 14px; font-weight: 700; color: #1a1a2e; }
.insight-card .text  { font-size: 13px; color: #555; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ─── Справочник регионов РФ (код ОГРН [3:5] → название + lat/lon) ────────────
REGIONS = {
    "01": ("Республика Адыгея",              44.68,  40.10),
    "02": ("Республика Башкортостан",        54.74,  55.97),
    "03": ("Республика Бурятия",             52.30, 107.60),
    "04": ("Республика Алтай",               51.96,  85.96),
    "05": ("Республика Дагестан",            42.98,  47.50),
    "06": ("Республика Ингушетия",           43.17,  44.82),
    "07": ("Кабардино-Балкарская Респ.",     43.48,  43.60),
    "08": ("Республика Калмыкия",            46.31,  44.27),
    "09": ("Карачаево-Черкесская Респ.",     43.88,  41.73),
    "10": ("Республика Карелия",             62.00,  32.90),
    "11": ("Республика Коми",                63.84,  53.85),
    "12": ("Республика Марий Эл",            56.64,  47.90),
    "13": ("Республика Мордовия",            54.19,  45.18),
    "14": ("Республика Саха (Якутия)",       62.03, 129.73),
    "15": ("Республика Северная Осетия",     43.02,  44.68),
    "16": ("Республика Татарстан",           56.00,  50.20),
    "17": ("Республика Тыва",                51.72,  94.44),
    "18": ("Удмуртская Республика",          57.07,  53.65),
    "19": ("Республика Хакасия",             53.72,  91.44),
    "20": ("Чеченская Республика",           43.40,  45.72),
    "21": ("Чувашская Республика",           56.14,  47.25),
    "22": ("Алтайский край",                 52.70,  82.95),
    "23": ("Краснодарский край",             45.04,  38.98),
    "24": ("Красноярский край",              61.04,  89.04),
    "25": ("Приморский край",                44.60, 135.10),
    "26": ("Ставропольский край",            45.04,  42.98),
    "27": ("Хабаровский край",               53.50, 134.90),
    "28": ("Амурская область",               53.74, 127.61),
    "29": ("Архангельская область",          64.54,  40.54),
    "30": ("Астраханская область",           46.35,  48.04),
    "31": ("Белгородская область",           50.60,  36.59),
    "32": ("Брянская область",               53.26,  34.36),
    "33": ("Владимирская область",           56.13,  40.39),
    "34": ("Волгоградская область",          48.71,  44.51),
    "35": ("Вологодская область",            59.22,  39.89),
    "36": ("Воронежская область",            51.66,  39.19),
    "37": ("Ивановская область",             57.00,  41.98),
    "38": ("Иркутская область",              57.75, 103.20),
    "39": ("Калининградская область",        54.71,  20.51),
    "40": ("Калужская область",              54.51,  36.26),
    "41": ("Камчатский край",                53.01, 158.65),
    "42": ("Кемеровская область",            54.00,  86.09),
    "43": ("Кировская область",              58.60,  49.67),
    "44": ("Костромская область",            58.55,  43.99),
    "45": ("Курганская область",             55.44,  65.34),
    "46": ("Курская область",                51.73,  36.19),
    "47": ("Ленинградская область",          59.95,  30.20),
    "48": ("Липецкая область",               52.61,  39.60),
    "49": ("Магаданская область",            59.57, 150.79),
    "50": ("Московская область",             55.81,  37.51),
    "51": ("Мурманская область",             68.97,  33.07),
    "52": ("Нижегородская область",          56.32,  44.00),
    "53": ("Новгородская область",           58.52,  31.27),
    "54": ("Новосибирская область",          54.97,  82.90),
    "55": ("Омская область",                 55.00,  73.38),
    "56": ("Оренбургская область",           51.77,  55.10),
    "57": ("Орловская область",              52.97,  36.07),
    "58": ("Пензенская область",             53.20,  44.99),
    "59": ("Пермский край",                  58.00,  56.25),
    "60": ("Псковская область",              57.82,  28.34),
    "61": ("Ростовская область",             47.22,  39.71),
    "62": ("Рязанская область",              54.63,  39.73),
    "63": ("Самарская область",              53.20,  50.15),
    "64": ("Саратовская область",            51.54,  46.03),
    "65": ("Сахалинская область",            50.69, 142.76),
    "66": ("Свердловская область",           57.00,  60.60),
    "67": ("Смоленская область",             54.78,  32.04),
    "68": ("Тамбовская область",             52.72,  41.44),
    "69": ("Тверская область",               57.00,  35.90),
    "70": ("Томская область",                58.60,  82.68),
    "71": ("Тульская область",               54.19,  37.62),
    "72": ("Тюменская область",              57.15,  68.99),
    "73": ("Ульяновская область",            54.31,  48.39),
    "74": ("Челябинская область",            55.15,  61.40),
    "75": ("Забайкальский край",             52.04, 113.50),
    "76": ("Ярославская область",            57.63,  39.87),
    "77": ("Москва",                         55.75,  37.62),
    "78": ("Санкт-Петербург",               59.94,  30.32),
    "79": ("Еврейская АО",                   48.48, 132.90),
    "80": ("Забайкальский край (общ.)",      52.04, 113.50),
    "83": ("Ненецкий АО",                    67.64,  53.00),
    "85": ("Иркутская обл. (УОБАО)",         52.70, 101.50),
    "86": ("Ханты-Мансийский АО",            61.00,  69.00),
    "87": ("Чукотский АО",                   65.75, 172.70),
    "89": ("Ямало-Ненецкий АО",              66.60,  74.40),
    "91": ("Республика Крым",                45.05,  34.10),
    "92": ("Севастополь",                    44.59,  33.52),
}

# ─── Цветовая схема ──────────────────────────────────────────────────────────
PAL = {
    "primary": "#3B82F6", "danger": "#EF4444", "warning": "#F59E0B",
    "success": "#10B981", "purple": "#8B5CF6", "dark": "#1E293B",
    "light": "#64748B",
}

CAT_COLORS = {
    "ВАРИАНТ_ФОРМАТ": "#EF4444", "ЧАСТОТА": "#F97316",
    "ВОЗРАСТ_КЛАСС": "#3B82F6", "ДОКУМЕНТ_СОВПАДЕНИЕ": "#F59E0B",
    "ВАРИАНТ_КЛАСС": "#84CC16", "ДАТА_РОЖДЕНИЯ": "#8B5CF6",
    "ДОКУМЕНТ_ПРЕДСТАВИТЕЛЬ": "#EC4899", "ДОКУМЕНТ_РЕБЁНОК": "#14B8A6",
    "ПУСТОЕ_ПОЛЕ": "#6B7280", "ПРЕДСТАВИТЕЛЬ_ВОЗРАСТ": "#06B6D4",
    "ОГРН_НАПРАВИВШАЯ": "#A78BFA", "КЛАСС_ФОРМАТ": "#FBD5E8",
}
CAT_RU = {
    "ВАРИАНТ_ФОРМАТ": "Некорректный формат варианта",
    "ЧАСТОТА": "Нарушение частоты (< 90 дн.)",
    "ВОЗРАСТ_КЛАСС": "Несоответствие возраста и класса",
    "ДОКУМЕНТ_СОВПАДЕНИЕ": "Совпадение doc ребёнка и представителя",
    "ВАРИАНТ_КЛАСС": "Вариант не соответствует классу",
    "ДАТА_РОЖДЕНИЯ": "Аномальная дата рождения",
    "ДОКУМЕНТ_ПРЕДСТАВИТЕЛЬ": "Некорректный doc представителя",
    "ДОКУМЕНТ_РЕБЁНОК": "Некорректный doc ребёнка",
    "ПУСТОЕ_ПОЛЕ": "Пустое обязательное поле",
    "ПРЕДСТАВИТЕЛЬ_ВОЗРАСТ": "Представитель слишком молод",
    "ОГРН_НАПРАВИВШАЯ": "Некорректный ОГРН школы",
    "КЛАСС_ФОРМАТ": "Нечисловой класс",
}
FEAT_RU = {
    "age_at_test": "Возраст на тест (лет)", "class_num": "Класс",
    "age_class_dev": "Отклонение возраст/класс",
    "guard_age_diff": "Разница возраста (представитель/ребёнок)",
    "result_code": "Результат (0/1)", "gender_code": "Пол (0/1)",
    "test_month": "Месяц тестирования", "test_year": "Год тестирования",
    "tests_per_child": "Тестов на ребёнка",
    "days_since_prev": "Дней с предыдущего теста",
}
MONTH_LABELS = ["Янв","Фев","Мар","Апр","Май","Июн","Июл","Авг","Сен","Окт","Ноя","Дек"]
FEATURES = ["age_at_test","class_num","age_class_dev","guard_age_diff",
            "result_code","gender_code","test_month","test_year",
            "tests_per_child","days_since_prev"]


# ─── Загрузка данных ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Загрузка данных…")
def load_data():
    summary  = pd.read_csv("results_notebook/summary.csv")
    manual   = pd.read_csv("results_notebook/manual_anomalies.csv", dtype=str)
    fixes    = pd.read_csv("results_notebook/fixes_log.csv", dtype=str)
    ml_anom  = pd.read_csv("results_notebook/ml_anomalies.csv", sep=";", dtype=str)
    full     = pd.read_csv("results_notebook/full_scored.csv",  sep=";", dtype=str)
    hakaton  = pd.read_csv("hakaton.csv", sep=";", dtype=str, keep_default_na=False)

    num_cols = ["age_at_test","class_num","age_class_dev","guard_age_diff",
                "result_code","gender_code","test_month","test_year",
                "tests_per_child","days_since_prev","anomaly_score","is_ml_anomaly"]
    for df in [full, ml_anom]:
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Код региона из ОГРН
    def region_code(ogrn):
        ogrn = str(ogrn).strip()
        return ogrn[3:5] if len(ogrn) == 13 and ogrn.isdigit() else "??"
    hakaton["region_code"] = hakaton["ogrn_naprav"].apply(region_code)
    hakaton["region_name"] = hakaton["region_code"].map(
        lambda c: REGIONS.get(c, (f"Регион {c}", 0, 0))[0]
    )

    return summary.iloc[0], manual, fixes, ml_anom, full, hakaton

summary, manual, fixes, ml_anom, full_df, hakaton = load_data()


# ─── Компоненты ──────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, color=PAL["primary"]):
    """Анимированная метрика-карточка — st.html() обходит markdown-парсер."""
    delta_part = (
        f'<p style="font-size:12px;margin:4px 0 0;opacity:0.6;color:#555">{delta}</p>'
        if delta else ""
    )
    card_style = (
        f"background:white;border-radius:14px;padding:16px 18px 14px;"
        f"border-left:5px solid {color};box-shadow:0 2px 12px rgba(0,0,0,0.07);"
        f"min-height:90px;"
    )
    label_style = "font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;opacity:.55;margin:0 0 4px;color:#333"
    value_style = f"font-size:28px;font-weight:800;color:{color};margin:0 0 2px;line-height:1"
    st.html(
        f'<div style="{card_style}">'
        f'<p style="{label_style}">{label}</p>'
        f'<p style="{value_style}">{value}</p>'
        f'{delta_part}'
        f'</div>'
    )

def section_title(text):
    st.html(
        f'<p style="font-size:18px;font-weight:700;border-bottom:2px solid #f0f2f6;padding-bottom:8px;margin:20px 0 14px;color:#1a1a2e">{text}</p>'
    )

def insight(icon, title, text):
    st.html(
        f'<div style="background:linear-gradient(135deg,#f8f9fa,#fff);border-radius:12px;padding:16px 18px;border:1px solid #e9ecef;margin-bottom:10px">'
        f'<p style="font-size:14px;font-weight:700;color:#1a1a2e;margin:0 0 4px">{title}</p>'
        f'<p style="font-size:13px;color:#555;margin:0">{text}</p>'
        f'</div>'
    )

def page_header(title, subtitle=""):
    sub_part = f'<p style="margin:6px 0 0;opacity:.7;font-size:14px;color:white">{subtitle}</p>' if subtitle else ""
    st.html(
        f'<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);'
        f'border-radius:16px;padding:24px 32px;margin-bottom:24px;color:white">'
        f'<h1 style="margin:0;font-size:28px;font-weight:700;letter-spacing:-.5px;color:white">{title}</h1>'
        f'{sub_part}'
        f'</div>'
    )

def plotly_defaults(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=30, b=20),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"),
    )
    return fig


# ─── Боковая панель ──────────────────────────────────────────────────────────
with st.sidebar:
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image("medium_a975a7ab34479e1d158d6239e5a41963.png", width=150)
    st.html(
        '<p style="font-size:18px;font-weight:800;color:#1E293B;margin:4px 0 2px;text-align:center">ЦИТиС 2026</p>'
        '<p style="font-size:12px;color:#64748B;margin:0 0 8px;text-align:center">Анализ тестирований</p>'
    )
    st.divider()
    page = st.radio(
        "Раздел",
        ["Исправления", "Ручные аномалии", "Частота тестирований",
         "Анализ аномалий (ML)", "Карта РФ", "Обзор"],
        label_visibility="collapsed",
    )
    st.divider()
    total = int(summary["total_records"])
    manual_n = int(summary["manual_anomalies"])
    ml_n = int(summary["ml_anomalies"])
    st.html(
        f'<div style="font-size:12px;color:#64748B;line-height:2">'
        f'Записей: <b style="color:#1E293B">{total:,}</b><br>'
        f'Ручных: <b style="color:#F59E0B">{manual_n:,}</b><br>'
        f'ML: <b style="color:#EF4444">{ml_n:,}</b><br>'
        f'Чистых: <b style="color:#10B981">{int(summary["truly_clean"]):,}</b>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 1: ОБЗОР
# ══════════════════════════════════════════════════════════════════════════════
if page == "Обзор":
    page_header("Анализ тестирований по истории", "Двухэтапный анализ качества данных: ручные правила + Isolation Forest · 25 628 записей")

    # Метрики
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: metric_card("Всего записей",   f"{total:,}",                          color=PAL["dark"])
    with c2: metric_card("Исправлено FIX",  f"{int(summary['fixed_records']):,}",  f"{int(summary['fixed_records'])/total*100:.0f}% датасета", PAL["success"])
    with c3: metric_card("Ручных аномалий", f"{manual_n:,}",                       f"{manual_n/total*100:.1f}%", PAL["warning"])
    with c4: metric_card("Подано в IF",     f"{int(summary['clean_for_ml']):,}",   f"{int(summary['clean_for_ml'])/total*100:.1f}%", PAL["primary"])
    with c5: metric_card("ML-аномалий",     f"{ml_n:,}",                           f"5.0% чистых", PAL["danger"])
    with c6: metric_card("Полностью чистых",f"{int(summary['truly_clean']):,}",    f"{int(summary['truly_clean'])/total*100:.1f}%", PAL["success"])

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        section_title("Воронка обработки данных")
        stages = [
            ("Исходный датасет",         total,                        PAL["dark"]),
            ("Исправлено",          int(summary["fixed_records"]),PAL["success"]),
            ("Ручных аномалий",           manual_n,                     PAL["warning"]),
            ("Подано в Isolation Forest", int(summary["clean_for_ml"]), PAL["primary"]),
            ("ML-аномалий (новых)",       ml_n,                         PAL["danger"]),
            ("Полностью чистых",          int(summary["truly_clean"]),  "#10B981"),
        ]
        fig = go.Figure(go.Bar(
            x=[s[1] for s in stages], y=[s[0] for s in stages],
            orientation="h",
            marker=dict(color=[s[2] for s in stages], line=dict(width=0)),
            text=[f"  {s[1]:,}" for s in stages],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,} записей<extra></extra>",
        ))
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Количество записей",
        )
        plotly_defaults(fig, 320)
        st.plotly_chart(fig, key="funnel", width="stretch")

    with col_r:
        section_title("Итоговое распределение")
        clean = int(summary["truly_clean"])
        fig2 = go.Figure(go.Pie(
            labels=["Чистые", "Ручные аномалии", "ML-аномалии"],
            values=[clean, manual_n, ml_n],
            marker=dict(colors=[PAL["success"], PAL["warning"], PAL["danger"]],
                        line=dict(color="white", width=2)),
            hole=0.55,
            hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
            textinfo="percent",
        ))
        plotly_defaults(fig2, 320)
        st.plotly_chart(fig2, key="pie_total", width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Ключевые инсайты")
    i1, i2, i3 = st.columns(3)
    with i1:
        insight("", "Массовая техническая ошибка",
                "53% записей содержали отрицательные/нечисловые номера документов — "
                "системный баг выгрузки, исправлен автоматически.")
    with i2:
        insight("", "Нарушения частоты тестирования",
                f"1 432 случая тестирования чаще 90 дней. "
                "Минимальный зафиксированный интервал — 1 день.")
    with i3:
        insight("", "ML нашла 1 044 новых аномалии",
                "Старшие классы в нетипичные месяцы (фев–апр), "
                "повторные тестирования вне сезона.")


# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 2: КАРТА РФ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Карта РФ":
    page_header("География нарушений по регионам РФ", "Код региона извлекается из ОГРН направившей школы (позиции 3–4) · наведите на пузырёк")

    # ── Подготовка данных ──
    manual_with_region = manual.merge(
        hakaton[["our_number", "region_code", "region_name", "ogrn_naprav"]],
        on="our_number", how="left"
    )
    all_with_region = hakaton.copy()

    # Агрегация по регионам
    total_by_region = all_with_region.groupby(["region_code","region_name"]).size().reset_index(name="total")
    anom_by_region  = manual_with_region.groupby(["region_code","region_name"]).size().reset_index(name="anomalies")
    freq_by_region  = manual_with_region[manual_with_region["category"]=="ЧАСТОТА"]\
                        .groupby(["region_code","region_name"]).size().reset_index(name="freq_viol")

    reg_stats = total_by_region.merge(anom_by_region,  on=["region_code","region_name"], how="left")\
                               .merge(freq_by_region,  on=["region_code","region_name"], how="left")
    reg_stats = reg_stats.fillna(0)
    reg_stats["anomaly_pct"] = (reg_stats["anomalies"] / reg_stats["total"] * 100).round(1)
    reg_stats["lat"] = reg_stats["region_code"].map(lambda c: REGIONS.get(c, (None,0,0))[1])
    reg_stats["lon"] = reg_stats["region_code"].map(lambda c: REGIONS.get(c, (None,0,0,))[2])
    reg_stats = reg_stats[reg_stats["lat"] != 0].copy()

    # ── Контролы ──
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1,1,2])
    with col_ctrl1:
        metric_field = st.selectbox("Показатель на карте:", [
            "Все аномалии", "Нарушения частоты", "% аномалий от региона", "Всего записей"
        ])
    with col_ctrl2:
        top_n = st.slider("Топ-N регионов на карте:", 5, 85, 40)

    field_map = {
        "Все аномалии": ("anomalies", PAL["danger"], "Аномалий"),
        "Нарушения частоты": ("freq_viol", PAL["warning"], "Нарушений частоты"),
        "% аномалий от региона": ("anomaly_pct", PAL["purple"], "% аномалий"),
        "Всего записей": ("total", PAL["primary"], "Всего записей"),
    }
    field_col, bubble_color, field_label = field_map[metric_field]

    plot_data = reg_stats.nlargest(top_n, field_col)

    # ── Неоновая карта ──
    st.html(
        '<div style="background:linear-gradient(90deg,#060D18,#0B1929);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:14px 20px 8px;margin-bottom:8px">'
        '<span style="color:#00D4FF;font-weight:700;font-size:16px">Карта нарушений · пузырёк = показатель · наведите для деталей</span>'
        '</div>'
    )

    NEON_LAND    = "#0B1929"   # тёмно-синий фон суши
    NEON_OCEAN   = "#060D18"   # почти чёрный океан
    NEON_BORDER  = "#00D4FF"   # циановые границы
    NEON_COAST   = "#0EA5E9"   # береговая линия
    NEON_BG      = "#060D18"   # фон всей карты

    neon_scales = {
        "Все аномалии":          [[0,"rgba(255,0,102,0.13)"],[0.4,"#FF4444"],[1,"#FF0066"]],
        "Нарушения частоты":     [[0,"rgba(255,179,0,0.13)"],[0.4,"#FFB300"],[1,"#FF6B00"]],
        "% аномалий от региона": [[0,"rgba(168,85,247,0.13)"],[0.4,"#A855F7"],[1,"#7C3AED"]],
        "Всего записей":         [[0,"rgba(0,212,255,0.13)"],[0.4,"#00D4FF"],[1,"#0EA5E9"]],
    }
    neon_colorscale = neon_scales.get(metric_field, neon_scales["Все аномалии"])

    fig_map = go.Figure()

    # ── Все регионы — тусклые точки-подложка
    fig_map.add_trace(go.Scattergeo(
        lat=reg_stats["lat"], lon=reg_stats["lon"],
        mode="markers",
        marker=dict(size=5, color="#1E3A5F", opacity=0.6,
                    line=dict(color="rgba(0,212,255,0.27)", width=0.5)),
        hoverinfo="skip", showlegend=False,
    ))

    # Топ регионов — неоновые пузыри
    max_val = plot_data[field_col].max() or 1
    sizes = (plot_data[field_col] / max_val * 55 + 10).clip(lower=10)

    fig_map.add_trace(go.Scattergeo(
        lat=plot_data["lat"],
        lon=plot_data["lon"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=plot_data[field_col],
            colorscale=neon_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=field_label, font=dict(color="#00D4FF", size=12)),
                thickness=12, len=0.55,
                tickfont=dict(color="#7FB3D3"),
                bgcolor="#0B1929",
                bordercolor="rgba(0,212,255,0.27)",
            ),
            opacity=0.9,
            line=dict(color="#00D4FF", width=1.5),
        ),
        hovertemplate=(
            "<b style='color:#00D4FF'>%{customdata[0]}</b><br>"
            f"<b>{field_label}:</b> %{{customdata[1]:,.0f}}<br>"
            "<b>Всего записей:</b> %{customdata[2]:,}<br>"
            "<b>Нарушений частоты:</b> %{customdata[3]:,}<br>"
            "<b>% аномалий:</b> %{customdata[4]:.1f}%<extra></extra>"
        ),
        customdata=plot_data[["region_name", field_col, "total", "freq_viol", "anomaly_pct"]].values,
        name="Регионы",
        showlegend=False,
    ))

    # Подписи крупных пузырей
    big = plot_data[plot_data[field_col] >= plot_data[field_col].quantile(0.75)]
    fig_map.add_trace(go.Scattergeo(
        lat=big["lat"], lon=big["lon"],
        mode="text",
        text=big["region_code"],
        textfont=dict(size=8, color="#E0F7FF", family="Inter"),
        hoverinfo="skip", showlegend=False,
    ))

    fig_map.update_geos(
        projection_type="equirectangular",
        lataxis_range=[40, 82],
        lonaxis_range=[19, 180],
        showland=True,      landcolor=NEON_LAND,
        showocean=True,     oceancolor=NEON_OCEAN,
        showlakes=True,     lakecolor="#0D2137",
        showrivers=True,    rivercolor="#0D2137",
        showcountries=True, countrycolor=NEON_BORDER,
        showcoastlines=True,coastlinecolor=NEON_COAST,
        countrywidth=0.7,   coastlinewidth=1.2,
        bgcolor=NEON_BG,
    )
    fig_map.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=NEON_BG,
        plot_bgcolor=NEON_BG,
        font=dict(family="Inter", color="#7FB3D3"),
        hoverlabel=dict(
            bgcolor="#0B1929",
            bordercolor="#00D4FF",
            font=dict(size=13, color="#E0F7FF"),
        ),
    )
    st.plotly_chart(fig_map, key="russia_map", width="stretch")

    st.divider()

    # ── Таблица + барчарт ──
    col_tbl, col_bar = st.columns([3, 2])
    with col_tbl:
        section_title("Статистика по регионам")
        disp = reg_stats.nlargest(30, field_col)[
            ["region_code","region_name","total","anomalies","freq_viol","anomaly_pct"]
        ].copy()
        disp.columns = ["Код","Регион","Всего","Аномалий","Наруш. частоты","% аномалий"]
        disp["Аномалий"] = disp["Аномалий"].astype(int)
        disp["Наруш. частоты"] = disp["Наруш. частоты"].astype(int)
        st.dataframe(disp.reset_index(drop=True), width="stretch", height=450)

    with col_bar:
        section_title(f"Топ-15 по: {metric_field}")
        top15 = reg_stats.nlargest(15, field_col)
        fig_bar = go.Figure(go.Bar(
            x=top15[field_col],
            y=top15["region_name"].str[:30],
            orientation="h",
            marker=dict(
                color=top15[field_col],
                colorscale=[[0,"#FEE2E2"],[1, bubble_color]],
                line=dict(width=0),
            ),
            text=top15[field_col].apply(lambda v: f"{v:,.0f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>",
        ))
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        plotly_defaults(fig_bar, 450)
        st.plotly_chart(fig_bar, key="top15_bar", width="stretch")

    # ── Мини-инсайты ──
    top3 = reg_stats.nlargest(3, "anomalies")
    st.divider()
    section_title("Топ-3 региона по аномалиям")
    c1, c2, c3 = st.columns(3)
    icons = ["1", "2", "3"]
    for col, (_, row), icon in zip([c1,c2,c3], top3.iterrows(), icons):
        with col:
            insight(icon, row["region_name"],
                    f"Аномалий: {int(row['anomalies']):,} | "
                    f"Всего записей: {int(row['total']):,} | "
                    f"Нарушений частоты: {int(row['freq_viol']):,}")


# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 3: РУЧНЫЕ АНОМАЛИИ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ручные аномалии":
    page_header("Ручные аномалии", "13 правил · технические ошибки, логические противоречия, неполные записи")

    cat_counts = manual["category"].value_counts().reset_index()
    cat_counts.columns = ["category","count"]
    cat_counts["label"] = cat_counts["category"].map(lambda c: CAT_RU.get(c, c))
    cat_counts["color"] = cat_counts["category"].map(lambda c: CAT_COLORS.get(c, "#999"))

    section_title("Аномалии по категориям")
    fig = go.Figure(go.Bar(
        x=cat_counts["count"], y=cat_counts["label"],
        orientation="h",
        marker=dict(color=cat_counts["color"], line=dict(width=0)),
        text=cat_counts["count"].apply(lambda v: f"  {v:,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} записей<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Количество")
    plotly_defaults(fig, 380)
    st.plotly_chart(fig, key="cat_bar", width="stretch")

    st.divider()
    col_l, col_r = st.columns([1, 2])

    with col_l:
        section_title("Фильтр")
        all_labels = ["Все"] + list(cat_counts["label"])
        selected = st.selectbox("Категория:", all_labels)

        multi = manual.groupby("our_number")["category"].apply(list).reset_index()
        multi["n"] = multi["category"].apply(len)
        multi = multi[multi["n"] > 1]
        st.html(
            f'<div style="background:white;border-radius:14px;padding:16px 18px;border-left:5px solid {PAL["danger"]};box-shadow:0 2px 12px rgba(0,0,0,0.07);margin-top:16px">'
            f'<p style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;opacity:.55;margin:0 0 4px;color:#333">Записей с ≥ 2 нарушениями</p>'
            f'<p style="font-size:28px;font-weight:800;color:{PAL["danger"]};margin:0;line-height:1">{len(multi):,}</p>'
            f'</div>'
        )

    with col_r:
        section_title("Записи" + (f": {selected}" if selected != "Все" else ""))
        if selected == "Все":
            filtered = manual
        else:
            code = {v:k for k,v in CAT_RU.items()}.get(selected, selected)
            filtered = manual[manual["category"] == code]

        st.dataframe(
            filtered[["our_number","child","category","description"]]
            .rename(columns={"our_number":"Номер","child":"ФИО","category":"Категория","description":"Описание"})
            .reset_index(drop=True),
            width="stretch", height=400,
        )


# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 4: ЧАСТОТА
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Частота тестирований":
    page_header("Нарушения частоты тестирований", "Рекомендация: не чаще 1 раза в 3 месяца (90 дней) · 1 432 нарушения")

    freq = manual[manual["category"] == "ЧАСТОТА"].copy()
    freq["days"] = pd.to_numeric(
        freq["description"].str.extract(r"через (\d+) дн")[0], errors="coerce"
    )

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Нарушений", f"{len(freq):,}", color=PAL["danger"])
    with c2: metric_card("Мин. интервал", f"{int(freq['days'].min())} дн.", color=PAL["warning"])
    with c3: metric_card("Ср. интервал", f"{freq['days'].mean():.0f} дн.", color=PAL["primary"])
    with c4: metric_card("Интервал ≤ 7 дн.", f"{(freq['days']<=7).sum():,}", color="#8B5CF6")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        section_title("Гистограмма интервалов")
        fig = go.Figure(go.Histogram(
            x=freq["days"], nbinsx=44,
            marker=dict(color=PAL["danger"], line=dict(color="white", width=0.5)),
            opacity=0.85,
            hovertemplate="Интервал %{x} дн. — %{y} случаев<extra></extra>",
        ))
        fig.add_vline(x=90, line_dash="dash", line_color="#1E293B", line_width=2,
                      annotation_text="90 дн. — норма", annotation_position="top left",
                      annotation_font_size=12)
        fig.update_layout(xaxis_title="Дней между тестами", yaxis_title="Случаев")
        plotly_defaults(fig, 330)
        st.plotly_chart(fig, key="freq_hist", width="stretch")

    with col_r:
        section_title("Тяжесть нарушений")
        bins   = [0, 7, 14, 30, 60, 89]
        labels = ["≤ 7 дн.", "8–14 дн.", "15–30 дн.", "31–60 дн.", "61–89 дн."]
        colors = ["#7F1D1D","#DC2626","#F97316","#FCD34D","#FDE68A"]
        freq["sev"] = pd.cut(freq["days"], bins=bins, labels=labels, right=True)
        sev = freq["sev"].value_counts().reindex(labels, fill_value=0)
        fig2 = go.Figure(go.Bar(
            x=sev.index.astype(str), y=sev.values,
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=sev.values, textposition="outside",
        ))
        fig2.update_layout(xaxis_title="Интервал", yaxis_title="Случаев")
        plotly_defaults(fig2, 330)
        st.plotly_chart(fig2, key="freq_sev", width="stretch")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        section_title("Топ-20 детей с нарушениями")
        top_kids = freq.groupby(["our_number","child"]).agg(
            нарушений=("days","count"), мин_интервал=("days","min")
        ).sort_values("нарушений", ascending=False).head(20).reset_index()
        top_kids.columns = ["Номер","ФИО","Нарушений","Мин. интервал (дн.)"]
        st.dataframe(top_kids, width="stretch", hide_index=True)

    with col_b:
        section_title("Топ-15 школ с нарушениями")
        freq_sch = freq.merge(
            hakaton[["our_number","name_naprav","region_name"]],
            on="our_number", how="left"
        )
        top_sch = (freq_sch.groupby("name_naprav").size()
                   .sort_values(ascending=False).head(15).reset_index())
        top_sch.columns = ["Школа","Нарушений"]
        top_sch["Школа"] = top_sch["Школа"].str[:55]
        st.dataframe(top_sch, width="stretch", hide_index=True, height=400)

    section_title("Сезонность нарушений частоты")
    freq_d = freq.merge(hakaton[["our_number","test_date"]], on="our_number", how="left")
    freq_d["month"] = pd.to_datetime(freq_d["test_date"], errors="coerce").dt.month
    mc = freq_d["month"].value_counts().reindex(range(1,13), fill_value=0)
    fig3 = go.Figure(go.Bar(
        x=MONTH_LABELS, y=mc.values,
        marker=dict(color=[PAL["danger"] if v == mc.max() else "#FECACA" for v in mc.values],
                    line=dict(width=0)),
        text=mc.values, textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y} нарушений<extra></extra>",
    ))
    fig3.update_layout(xaxis_title="Месяц", yaxis_title="Нарушений")
    plotly_defaults(fig3, 280)
    st.plotly_chart(fig3, key="freq_month", width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 5: ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Анализ аномалий (ML)":
    page_header(
        "Анализ аномалий",
        "Использован метод Isolation Forest · обучен на 20 881 чистых записях · найдено 1 044 новых аномалии (5%)",
    )

    df_norm = full_df[full_df["is_ml_anomaly"] == 0]
    df_anom = ml_anom.copy()

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Обучающая выборка", f"{len(full_df):,}", color=PAL["primary"])
    with c2: metric_card("ML-аномалий", f"{len(df_anom):,}", "5% от чистых", PAL["danger"])
    with c3: metric_card("Score нормы (ср.)",   f"{df_norm['anomaly_score'].mean():.3f}", color=PAL["success"])
    with c4: metric_card("Score аномалий (ср.)",f"{df_anom['anomaly_score'].mean():.3f}", color=PAL["warning"])

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        section_title("Распределение anomaly_score")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_norm["anomaly_score"], name="Норма", nbinsx=50,
            marker=dict(color=PAL["primary"]), opacity=0.65,
        ))
        fig.add_trace(go.Histogram(
            x=df_anom["anomaly_score"], name="ML аномалия", nbinsx=30,
            marker=dict(color=PAL["danger"]), opacity=0.9,
        ))
        fig.update_layout(
            barmode="overlay",
            xaxis_title="anomaly_score (ниже = аномальнее)",
            legend=dict(orientation="h", y=1.05),
        )
        plotly_defaults(fig, 340)
        st.plotly_chart(fig, key="score_dist", width="stretch")

    with col_r:
        section_title("Класс: норма vs ML-аномалия")
        nc = df_norm["class_num"].dropna().value_counts().sort_index()
        ac = df_anom["class_num"].dropna().value_counts().sort_index()
        all_c = sorted(set(nc.index) | set(ac.index))
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=all_c, y=[nc.get(c,0) for c in all_c],
                              name="Норма", marker_color=PAL["primary"], opacity=0.8))
        fig2.add_trace(go.Bar(x=all_c, y=[ac.get(c,0) for c in all_c],
                              name="ML аномалия", marker_color=PAL["danger"], opacity=0.85))
        fig2.update_layout(barmode="overlay", xaxis_title="Класс",
                           legend=dict(orientation="h", y=1.05))
        plotly_defaults(fig2, 340)
        st.plotly_chart(fig2, key="class_dist", width="stretch")

    section_title("Сравнение признаков: норма vs аномалия")
    feat_data = []
    for f in FEATURES:
        nm = df_norm[f].mean() if f in df_norm.columns else 0
        am = df_anom[f].mean() if f in df_anom.columns else 0
        feat_data.append({"Признак": FEAT_RU.get(f,f), "Норма": round(nm,2),
                           "Аномалия": round(am,2), "Δ": round(am-nm,2)})
    feat_df = pd.DataFrame(feat_data)

    col_t, col_c = st.columns([1,2])
    with col_t:
        def color_delta(v):
            c = "#fde8e8" if v > 0.3 else ("#e8f8f5" if v < -0.3 else "")
            return f"background-color: {c}"
        st.dataframe(
            feat_df.style.map(color_delta, subset=["Δ"]),
            width="stretch", hide_index=True, height=360,
        )
    with col_c:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=feat_df["Признак"], y=feat_df["Норма"],
                              name="Норма", marker_color=PAL["primary"], opacity=0.8))
        fig3.add_trace(go.Bar(x=feat_df["Признак"], y=feat_df["Аномалия"],
                              name="Аномалия", marker_color=PAL["danger"], opacity=0.85))
        fig3.update_layout(barmode="group", xaxis_tickangle=-30,
                           legend=dict(orientation="h", y=1.05))
        plotly_defaults(fig3, 360)
        st.plotly_chart(fig3, key="feat_cmp", width="stretch")

    section_title("Сезонность: норма vs ML-аномалия")
    nm_mon = df_norm["test_month"].value_counts().sort_index()
    am_mon = df_anom["test_month"].value_counts().sort_index()
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=MONTH_LABELS, y=[nm_mon.get(i,0) for i in range(1,13)],
                          name="Норма", marker_color=PAL["primary"], opacity=0.7))
    fig4.add_trace(go.Bar(x=MONTH_LABELS, y=[am_mon.get(i,0) for i in range(1,13)],
                          name="ML аномалия", marker_color=PAL["danger"], opacity=0.9))
    fig4.update_layout(barmode="group", xaxis_title="Месяц",
                       legend=dict(orientation="h", y=1.05))
    plotly_defaults(fig4, 280)
    st.plotly_chart(fig4, key="ml_season", width="stretch")

    section_title("Топ-50 ML-аномалий (наиболее аномальные)")
    cols_show = [c for c in ["our_number","last_name","first_name","bdate","class",
                              "test_date","result","anomaly_score","age_at_test","tests_per_child"]
                 if c in df_anom.columns]
    rename = {"our_number":"Номер","last_name":"Фамилия","first_name":"Имя",
              "bdate":"Дата рожд.","class":"Класс","test_date":"Дата теста",
              "result":"Результат","anomaly_score":"Score",
              "age_at_test":"Возраст","tests_per_child":"Тестов"}
    st.dataframe(
        df_anom.sort_values("anomaly_score").head(50)[cols_show].rename(columns=rename).reset_index(drop=True),
        width="stretch", height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 6: ИСПРАВЛЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Исправления":
    page_header("Автоматические исправления", "Исправимые ошибки не удаляются, а корректируются — данные сохраняются для обучения модели")

    fc = fixes["field"].value_counts().reset_index()
    fc.columns = ["Поле","Исправлено"]

    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Всего исправлений",  f"{len(fixes):,}",            color=PAL["success"])
    with c2: metric_card("Уникальных записей", f"{fixes['our_number'].nunique():,}", color=PAL["primary"])
    with c3: metric_card("Полей исправлялось", f"{fixes['field'].nunique()}", color=PAL["purple"])

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 2])

    with col_l:
        section_title("Исправлений по полям")
        fig = go.Figure(go.Bar(
            x=fc["Исправлено"], y=fc["Поле"],
            orientation="h",
            marker=dict(color=PAL["success"], line=dict(width=0)),
            text=fc["Исправлено"].apply(lambda v: f"  {v:,}"),
            textposition="outside",
        ))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Количество")
        plotly_defaults(fig, 280)
        st.plotly_chart(fig, key="fix_bar", width="stretch")

        insight("", "Почему важно исправлять, а не удалять?",
                "53% датасета содержали технические ошибки в id_doc. "
                "Удаление сократило бы обучающую выборку вдвое — "
                "модель обучалась бы на неполных данных.")

    with col_r:
        section_title("Примеры исправлений по полю")
        sel_field = st.selectbox("Поле:", fc["Поле"].tolist())
        sub = fixes[fixes["field"] == sel_field][
            ["our_number","old_value","new_value","reason"]
        ].rename(columns={"our_number":"Номер","old_value":"Было",
                           "new_value":"Стало","reason":"Причина"})
        st.dataframe(sub.head(200).reset_index(drop=True), width="stretch", height=400)
        st.caption(f"Показано 200 из {len(sub):,} исправлений поля `{sel_field}`")
