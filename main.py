"""
Yuva Intern Task 5: Analytics Report & Insights Documentation
Project: Student Performance Analytics
Dataset: Synthetic educational practice data (not real student records).

Run:
    python -m pip install -r requirements.txt
    python main.py

The program creates summary tables, charts, a findings file, and a model
evaluation file inside the outputs/ folder.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Keep the data folder with main.py."
        )

    data = pd.read_csv(DATA_FILE)
    required = [
        "student_id", "study_hours_per_day", "attendance_percent",
        "assignment_score", "previous_score", "final_score"
    ]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data.drop_duplicates().dropna(subset=required).copy()
    numeric_cols = [
        "study_hours_per_day", "attendance_percent", "assignment_score",
        "previous_score", "final_score"
    ]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=numeric_cols)

    # Descriptive analytics
    summary = data[numeric_cols].describe().round(2)
    summary.to_csv(OUT / "descriptive_statistics.csv")

    # Correlation analysis
    corr = data[numeric_cols].corr().round(3)
    corr.to_csv(OUT / "correlation_matrix.csv")

    # Group performance
    data["attendance_group"] = pd.cut(
        data["attendance_percent"],
        bins=[0, 60, 75, 100],
        labels=["Below 60%", "60-75%", "Above 75%"],
        include_lowest=True
    )
    attendance_summary = data.groupby(
        "attendance_group", observed=False
    )["final_score"].agg(["count", "mean"]).round(2)
    attendance_summary.to_csv(OUT / "attendance_group_analysis.csv")

    # Key metrics and findings
    avg_score = float(data["final_score"].mean())
    median_score = float(data["final_score"].median())
    avg_attendance = float(data["attendance_percent"].mean())
    avg_study = float(data["study_hours_per_day"].mean())
    pass_rate = float((data["final_score"] >= 40).mean() * 100)
    correlations = data[numeric_cols].corr()["final_score"].drop("final_score")
    strongest_factor = correlations.abs().idxmax()
    strongest_corr = float(correlations[strongest_factor])

    # Charts
    plt.figure(figsize=(8, 5))
    plt.hist(data["final_score"], bins=20, edgecolor="black")
    plt.title("Distribution of Final Scores")
    plt.xlabel("Final score")
    plt.ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(OUT / "final_score_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    means = data.groupby("attendance_group", observed=False)["final_score"].mean()
    means.plot(kind="bar", edgecolor="black")
    plt.title("Average Final Score by Attendance Group")
    plt.xlabel("Attendance group")
    plt.ylabel("Average final score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "attendance_vs_score.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(data["study_hours_per_day"], data["final_score"], alpha=0.65)
    plt.title("Study Hours and Final Score")
    plt.xlabel("Study hours per day")
    plt.ylabel("Final score")
    plt.tight_layout()
    plt.savefig(OUT / "study_hours_vs_score.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    corr["final_score"].drop("final_score").sort_values().plot(
        kind="barh", edgecolor="black"
    )
    plt.title("Correlation of Factors with Final Score")
    plt.xlabel("Correlation coefficient")
    plt.ylabel("Factor")
    plt.tight_layout()
    plt.savefig(OUT / "factor_correlation.png", dpi=150)
    plt.close()

    # Predictive model (supporting analysis, not a guarantee of real outcomes)
    features = [
        "study_hours_per_day", "attendance_percent",
        "assignment_score", "previous_score"
    ]
    X = data[features]
    y = data["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=150, random_state=42, max_depth=6
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
    r2 = float(r2_score(y_test, predictions))
    metrics = pd.DataFrame([{
        "model": "RandomForestRegressor",
        "test_rows": len(y_test),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2_score": round(r2, 3)
    }])
    metrics.to_csv(OUT / "model_evaluation.csv", index=False)

    findings = f"""YUVA INTERN TASK 5 - KEY FINDINGS
Project: Student Performance Analytics
Dataset: Synthetic educational practice data; not real student records.

DATA QUALITY
- Rows analyzed: {len(data)}
- Duplicate rows removed: records were checked before analysis.
- Missing required values: rows with missing required fields were excluded.

KEY PERFORMANCE INDICATORS
- Average final score: {avg_score:.2f}/100
- Median final score: {median_score:.2f}/100
- Average attendance: {avg_attendance:.2f}%
- Average daily study time: {avg_study:.2f} hours
- Score-based pass rate (score >= 40): {pass_rate:.2f}%

RELATIONSHIPS
- Factor most strongly correlated in absolute value with final score:
  {strongest_factor} (correlation: {strongest_corr:.3f})
- Correlation indicates association, not proof of causation.

MODEL CHECK
- Model: RandomForestRegressor
- Mean Absolute Error: {mae:.2f}
- Root Mean Squared Error: {rmse:.2f}
- R-squared: {r2:.3f}
- The model is an educational demonstration using synthetic data only.

RECOMMENDATIONS
1. Review students with low attendance and provide timely support.
2. Encourage a consistent study routine and monitor progress monthly.
3. Use assignment feedback to identify topic-level learning gaps.
4. Review the charts and group averages before planning interventions.
5. Collect real, consented, privacy-protected data before making operational decisions.
"""
    (OUT / "key_findings.txt").write_text(findings, encoding="utf-8")

    print("Analysis complete.")
    print(f"Rows analyzed: {len(data)}")
    print(f"Average final score: {avg_score:.2f}/100")
    print(f"Pass rate (score >= 40): {pass_rate:.2f}%")
    print(f"Strongest correlated factor: {strongest_factor} ({strongest_corr:.3f})")
    print(f"Model MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
    print(f"Output folder: {OUT}")

if __name__ == "__main__":
    main()
