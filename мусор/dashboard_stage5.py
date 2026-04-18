import pandas as pd
import streamlit as st


st.set_page_config(page_title="ЦИТиС 2026 Dashboard", layout="wide")
st.title("ЦИТиС 2026: мониторинг качества и нарушений")


@st.cache_data
def load_data():
    violations = pd.read_csv("violations_registry_stage3.csv")
    by_child = pd.read_csv("violations_by_child_stage3.csv")
    school_naprav = pd.read_csv("school_naprav_stats_stage3.csv")
    school_area = pd.read_csv("school_area_stats_stage3.csv")
    monthly = pd.read_csv("monthly_violation_stats_stage3.csv")
    scoring = pd.read_csv("ml_scoring_stage4_iforest.csv")
    importance = pd.read_csv("stage4_iforest_metrics.csv")
    return violations, by_child, school_naprav, school_area, monthly, scoring, importance


violations, by_child, school_naprav, school_area, monthly, scoring, importance = load_data()

violations["current_test_date"] = pd.to_datetime(violations["current_test_date"], errors="coerce")
severity_order = ["critical", "high", "medium"]
violations["violation_severity"] = pd.Categorical(
    violations["violation_severity"], categories=severity_order, ordered=True
)

st.sidebar.header("Фильтры")
selected_severity = st.sidebar.multiselect(
    "Тяжесть нарушения",
    options=severity_order,
    default=severity_order,
)

min_date = violations["current_test_date"].min().date()
max_date = violations["current_test_date"].max().date()
selected_range = st.sidebar.date_input(
    "Период",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

filtered = violations[
    (violations["violation_severity"].isin(selected_severity))
    & (violations["current_test_date"].dt.date >= start_date)
    & (violations["current_test_date"].dt.date <= end_date)
].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Событий-нарушений", f"{len(filtered):,}".replace(",", " "))
col2.metric("Уникальных детей", f"{filtered['child_key_stage2'].nunique():,}".replace(",", " "))
col3.metric("Критических событий", int((filtered["violation_severity"] == "critical").sum()))
col4.metric("Средний интервал, дней", round(filtered["days_since_prev_test_stage3"].mean(), 1) if len(filtered) else 0)

st.subheader("Динамика нарушений по месяцам")
monthly_chart = monthly.copy()
monthly_chart["test_month"] = pd.to_datetime(monthly_chart["test_month"], errors="coerce")
monthly_chart = monthly_chart.set_index("test_month")[["violations_total", "violation_share_pct"]]
st.line_chart(monthly_chart)

left, right = st.columns(2)

with left:
    st.subheader("ТОП-15 направивших школ по доле нарушений")
    st.dataframe(
        school_naprav.sort_values("violation_share_pct", ascending=False).head(15),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("ТОП-15 площадок тестирования по доле нарушений")
    st.dataframe(
        school_area.sort_values("violation_share_pct", ascending=False).head(15),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Распределение тяжести нарушений")
severity_dist = (
    filtered["violation_severity"]
    .value_counts(dropna=False)
    .rename_axis("severity")
    .reset_index(name="count")
)
st.bar_chart(severity_dist.set_index("severity"))

st.subheader("Реестр нарушений (filtered)")
show_cols = [
    "our_number",
    "child_key_stage2",
    "current_test_date",
    "days_since_prev_test_stage3",
    "violation_severity",
    "quality_score",
    "risk_level",
    "result_norm",
    "name_naprav",
    "name_area",
]
st.dataframe(
    filtered.sort_values(["violation_severity", "days_since_prev_test_stage3"]).head(500)[show_cols],
    use_container_width=True,
    hide_index=True,
)

st.subheader("ML риск-скоринг (Isolation Forest)")
scoring["iforest_risk_score"] = pd.to_numeric(scoring["iforest_risk_score"], errors="coerce")
risk_top = scoring.sort_values("iforest_risk_score", ascending=False).head(300)
st.dataframe(
    risk_top[
        [
            "our_number",
            "child_key_stage2",
            "test_date",
            "iforest_risk_score",
            "iforest_risk_bucket",
            "iforest_is_anomaly",
            "quality_score",
            "name_naprav",
            "name_area",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Метрики модели Isolation Forest")
st.dataframe(importance, use_container_width=True, hide_index=True)
