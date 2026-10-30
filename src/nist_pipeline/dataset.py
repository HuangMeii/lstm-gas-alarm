from pathlib import Path

import numpy as np
import pandas as pd

from .config import NIST_RAW_DIR
from .ingest import build_manifest


FEATURE_COLUMNS = ["obscuration", "mic"]
SEQ_LEN = 30


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index)), errors="coerce")


def load_experiments() -> pd.DataFrame:
    manifest = build_manifest()
    records: list[pd.DataFrame] = []
    for path in (NIST_RAW_DIR / "smokebox").rglob("*.tsv"):
        frame = pd.read_csv(path, sep="\t", skiprows=1)
        frame.columns = [str(column).strip() for column in frame.columns]
        alarm_rows = pd.to_numeric(frame.get("Alarm?"), errors="coerce").eq(1)
        if not alarm_rows.any():
            continue
        alarm_time = float(frame.loc[alarm_rows, "Counter"].iloc[0])
        current = pd.DataFrame(
            {
                "experiment_id": path.stem,
                "time_sec": pd.to_numeric(frame["Counter"], errors="coerce"),
                "obscuration": _numeric(frame, "Obscuration (%/ft)"),
                "mic": _numeric(frame, "MIC(pA)"),
                "alarm_time_sec": alarm_time,
            }
        )
        records.append(current)

    room_manifest = manifest[manifest["experiment_type"].eq("room")].drop_duplicates(
        "experiment_id"
    )
    for path in (NIST_RAW_DIR / "room").rglob("*.tsv"):
        match = room_manifest[room_manifest["experiment_id"].eq(path.stem)]
        if match.empty:
            continue
        frame = pd.read_csv(path, sep="\t")
        frame.columns = [str(column).strip() for column in frame.columns]
        current = pd.DataFrame(
            {
                "experiment_id": path.stem,
                "time_sec": _numeric(frame, "Counter (s)"),
                "obscuration": _numeric(frame, "Obsc (%/ft) smoothed"),
                "mic": _numeric(frame, "MIC (pA)"),
                "alarm_time_sec": float(match.iloc[0]["alarm_time_sec"]),
            }
        )
        records.append(current)

    if not records:
        raise ValueError("No labeled NIST experiments were found.")
    data = pd.concat(records, ignore_index=True).dropna(subset=FEATURE_COLUMNS + ["time_sec"])
    data["time_to_alarm_sec"] = data["alarm_time_sec"] - data["time_sec"]
    return data[data["time_to_alarm_sec"] >= 0].reset_index(drop=True)


def create_windows(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, targets, groups = [], [], []
    for experiment_id, frame in data.groupby("experiment_id", sort=False):
        frame = frame.sort_values("time_sec")
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        target = frame["time_to_alarm_sec"].to_numpy(dtype=np.float32)
        if len(frame) <= SEQ_LEN:
            continue
        for i in range(len(frame) - SEQ_LEN):
            windows.append(values[i : i + SEQ_LEN])
            targets.append(target[i + SEQ_LEN])
            groups.append(experiment_id)
    if not windows:
        raise ValueError("No valid NIST windows were created.")
    return np.asarray(windows), np.asarray(targets), np.asarray(groups)
