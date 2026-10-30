import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "smoke_detection_iot.csv"
EDA_REPORT_DIR = PROJECT_ROOT / "reports" / "eda"


def main() -> None:
    EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    numeric = df.select_dtypes(include=np.number)

    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    summary["unique"] = numeric.nunique()
    summary.to_csv(EDA_REPORT_DIR / "eda_numeric_summary.csv")

    alarm_counts = df["Fire Alarm"].value_counts(dropna=False).sort_index()
    alarm_summary = pd.DataFrame(
        {
            "count": alarm_counts,
            "percentage": alarm_counts / len(df) * 100,
        }
    )
    alarm_summary.to_csv(EDA_REPORT_DIR / "eda_alarm_distribution.csv")

    utc_gaps = df["UTC"].diff().dropna()
    gap_summary = pd.DataFrame(
        {
            "gap_seconds": utc_gaps.value_counts().index,
            "count": utc_gaps.value_counts().values,
        }
    ).sort_values("gap_seconds")
    gap_summary.to_csv(EDA_REPORT_DIR / "eda_utc_gaps.csv", index=False)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum()):,}")
    print(f"Duplicate rows: {int(df.duplicated().sum()):,}")
    print("Fire Alarm distribution:")
    print(alarm_summary.to_string())
    print(f"UTC gaps other than 1 second: {int((utc_gaps != 1).sum())}")
    print("\nNumeric summary:")
    print(summary.to_string())
    print(
        f"\nSaved reports to {EDA_REPORT_DIR}"
    )


if __name__ == "__main__":
    main()
