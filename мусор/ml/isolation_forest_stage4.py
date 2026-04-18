import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_stage3_base(records_stage2_path: str) -> pd.DataFrame:
    df = pd.read_csv(records_stage2_path)
    df["test_date_dt"] = pd.to_datetime(df["test_date_dt"], errors="coerce")
    df = df.sort_values(["child_key_stage2", "test_date_dt", "our_number"]).copy()
    df["prev_test_date_dt_stage3"] = df.groupby("child_key_stage2")["test_date_dt"].shift(1)
    df["days_since_prev_test_stage3"] = (df["test_date_dt"] - df["prev_test_date_dt_stage3"]).dt.days
    df["flag_freq_violation_stage3"] = (
        df["days_since_prev_test_stage3"].notna() & (df["days_since_prev_test_stage3"] < 90)
    ).astype(int)
    return df


def run_iforest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["test_month_num"] = work["test_date_dt"].dt.month
    work["test_weekday"] = work["test_date_dt"].dt.weekday
    work["child_attempt_no"] = work.groupby("child_key_stage2").cumcount() + 1
    work["child_total_attempts"] = work.groupby("child_key_stage2")["our_number"].transform("count")
    work["prev_quality_score"] = work.groupby("child_key_stage2")["quality_score"].shift(1)
    work["prev_result_norm"] = work.groupby("child_key_stage2")["result_norm"].shift(1)
    work["class_num_feature"] = pd.to_numeric(work["class"], errors="coerce")

    numeric_features = [
        "quality_score",
        "class_num_feature",
        "child_attempt_no",
        "child_total_attempts",
        "test_month_num",
        "test_weekday",
        "prev_quality_score",
    ]
    categorical_features = [
        "result_norm",
        "prev_result_norm",
        "variant",
        "ogrn_naprav",
        "ogrn_area",
        "risk_level",
    ]
    X = work[numeric_features + categorical_features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    Xp = preprocessor.fit_transform(X)
    iforest = IsolationForest(
        n_estimators=300,
        max_samples=4096,
        contamination=0.03,
        n_jobs=-1,
        random_state=42,
    )
    iforest.fit(Xp)

    decision = iforest.decision_function(Xp)
    pred = iforest.predict(Xp)  # -1 anomaly, 1 normal

    work["iforest_decision"] = decision
    work["iforest_is_anomaly"] = np.where(pred == -1, 1, 0)
    # Чем выше score, тем более подозрительная запись
    work["iforest_risk_score"] = -work["iforest_decision"]

    # Пороговые buckets по квантилям
    q1 = work["iforest_risk_score"].quantile(0.70)
    q2 = work["iforest_risk_score"].quantile(0.90)
    work["iforest_risk_bucket"] = np.where(
        work["iforest_risk_score"] >= q2,
        "high",
        np.where(work["iforest_risk_score"] >= q1, "medium", "low"),
    )

    manual_cnt = int(work["flag_freq_violation_stage3"].sum())
    model_cnt = int(work["iforest_is_anomaly"].sum())
    overlap = int(
        (
            (work["flag_freq_violation_stage3"] == 1)
            & (work["iforest_is_anomaly"] == 1)
        ).sum()
    )

    precision_vs_manual = overlap / model_cnt if model_cnt else 0.0
    recall_vs_manual = overlap / manual_cnt if manual_cnt else 0.0

    metrics = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(work)},
            {"metric": "manual_anomalies_total", "value": manual_cnt},
            {"metric": "iforest_anomalies_total", "value": model_cnt},
            {"metric": "overlap_manual_iforest", "value": overlap},
            {"metric": "precision_vs_manual", "value": round(precision_vs_manual, 4)},
            {"metric": "recall_vs_manual", "value": round(recall_vs_manual, 4)},
            {"metric": "iforest_contamination", "value": 0.03},
        ]
    )

    top_cases = work.sort_values("iforest_risk_score", ascending=False).head(1000)[
        [
            "our_number",
            "child_key_stage2",
            "test_date",
            "result_norm",
            "quality_score",
            "risk_level",
            "iforest_risk_score",
            "iforest_is_anomaly",
            "iforest_risk_bucket",
            "flag_freq_violation_stage3",
            "days_since_prev_test_stage3",
            "name_naprav",
            "name_area",
        ]
    ]

    return work, top_cases, metrics


def main() -> None:
    base = build_stage3_base("records_stage2.csv")
    scored, top_cases, metrics = run_iforest(base)

    scored.to_csv("ml_scoring_stage4_iforest.csv", index=False)
    top_cases.to_csv("ml_high_risk_cases_stage4_iforest.csv", index=False)
    metrics.to_csv("stage4_iforest_metrics.csv", index=False)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
