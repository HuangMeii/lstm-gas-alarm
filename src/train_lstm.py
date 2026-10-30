import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "smoke_detection_iot.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
SEQ_LEN = 30
EARLY_WARNING_SECONDS = 300
EPOCHS = 10
PATIENCE = 3
RANDOM_SEED = 42
FEATURE_COLS = [
    "Temperature[C]",
    "Humidity[%]",
    "TVOC[ppb]",
    "eCO2[ppm]",
    "Pressure[hPa]",
]


def create_time_to_alarm_labels(df: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(df), np.nan, dtype=np.float32)
    timestamps = df["UTC"].to_numpy()
    alarms = df["Fire Alarm"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(timestamps) != 1) + 1]
    ends = np.r_[starts[1:], len(df)]
    for start, end in zip(starts, ends):
        next_alarm = None
        for i in range(end - 1, start - 1, -1):
            if alarms[i] == 1:
                next_alarm = timestamps[i]
            if next_alarm is not None:
                labels[i] = max(0.0, float(next_alarm - timestamps[i]))
    return labels


def create_event_ids(df: pd.DataFrame) -> np.ndarray:
    timestamps = df["UTC"].to_numpy()
    alarm = df["Fire Alarm"].to_numpy()
    segment = np.cumsum(np.r_[True, np.diff(timestamps) != 1]) - 1
    event_id = np.full(len(df), -1, dtype=int)
    next_event = 0
    for segment_id in np.unique(segment):
        indices = np.flatnonzero(segment == segment_id)
        onsets = indices[(alarm[indices] == 1) & (np.r_[0, alarm[indices[:-1]]] == 0)]
        for event_number, onset in enumerate(onsets):
            event_start = indices[0] if event_number == 0 else onsets[event_number - 1]
            event_end = (
                onsets[event_number + 1]
                if event_number + 1 < len(onsets)
                else indices[-1] + 1
            )
            event_id[event_start:event_end] = next_event
            next_event += 1
    return event_id


def create_windows(
    X: np.ndarray,
    times: np.ndarray,
    time_to_alarm: np.ndarray,
    event_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, targets, groups = [], [], []
    for i in range(len(X) - SEQ_LEN):
        end = i + SEQ_LEN
        if not np.all(np.diff(times[i : end + 1]) == 1):
            continue
        target = time_to_alarm[end]
        if np.isnan(target):
            continue
        windows.append(X[i:end])
        targets.append(target)
        groups.append(event_ids[end])
    return np.asarray(windows), np.asarray(targets, dtype=np.float32), np.asarray(groups)


class AlarmLSTM(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, 2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class WeightedHuberLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + 2.0 * torch.exp(-target / 60.0)
        return (weights * nn.functional.huber_loss(prediction, target, reduction="none")).mean()


def split_by_event(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_ids = np.sort(np.unique(groups[groups >= 0]))
    if len(event_ids) < 3:
        raise ValueError("Cần ít nhất 3 sự kiện alarm để chia train/validation/test.")
    train_end = max(1, int(len(event_ids) * 0.6))
    validation_end = max(train_end + 1, int(len(event_ids) * 0.8))
    return (
        np.isin(groups, event_ids[:train_end]),
        np.isin(groups, event_ids[train_end:validation_end]),
        np.isin(groups, event_ids[validation_end:]),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_X: torch.Tensor,
    validation_y: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(
                model(validation_X.to(device)), validation_y.to(device)
            ).item()
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"Train loss: {train_loss / len(train_loader):.6f} - "
            f"Validation loss: {validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Không lưu được trạng thái tốt nhất.")
    model.load_state_dict(best_state)
    return model


def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    missing = set(FEATURE_COLS + ["UTC", "Fire Alarm"]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["time_to_alarm_sec"] = create_time_to_alarm_labels(df)
    event_ids = create_event_ids(df)
    train_rows = df[event_ids >= 0].iloc[: int(len(df) * 0.8)]
    scaler_X = MinMaxScaler().fit(train_rows[FEATURE_COLS])
    X = scaler_X.transform(df[FEATURE_COLS])
    joblib.dump(scaler_X, MODEL_DIR / "scaler_X.pkl")

    windows, times_to_alarm, groups = create_windows(
        X, df["UTC"].to_numpy(), df["time_to_alarm_sec"].to_numpy(), event_ids
    )
    train_mask, validation_mask, test_mask = split_by_event(groups)
    warning = (times_to_alarm > 0) & (times_to_alarm <= EARLY_WARNING_SECONDS)
    classifier_y = warning.astype(np.float32)
    regressor_mask = times_to_alarm > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = AlarmLSTM(1).to(device)
    train_X = torch.tensor(windows[train_mask], dtype=torch.float32)
    validation_X = torch.tensor(windows[validation_mask], dtype=torch.float32)
    class_train_y = torch.tensor(classifier_y[train_mask, None])
    class_validation_y = torch.tensor(classifier_y[validation_mask, None])
    positive = max(1, int(class_train_y.sum()))
    negative = max(1, len(class_train_y) - positive)
    class_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    classifier = train_model(
        classifier,
        DataLoader(TensorDataset(train_X, class_train_y), batch_size=64, shuffle=True),
        validation_X,
        class_validation_y,
        class_loss,
        device,
    )

    regression_train_mask = train_mask & regressor_mask
    regression_validation_mask = validation_mask & regressor_mask
    y_max = max(EARLY_WARNING_SECONDS, float(times_to_alarm[regression_train_mask].max()))
    scaler_y = MinMaxScaler().fit(times_to_alarm[regression_train_mask, None])
    joblib.dump(scaler_y, MODEL_DIR / "scaler_y.pkl")
    regressor = AlarmLSTM(1).to(device)
    reg_train_y = torch.tensor(
        scaler_y.transform(times_to_alarm[regression_train_mask, None]), dtype=torch.float32
    )
    reg_validation_y = torch.tensor(
        scaler_y.transform(times_to_alarm[regression_validation_mask, None]), dtype=torch.float32
    )
    regressor = train_model(
        regressor,
        DataLoader(
            TensorDataset(torch.tensor(windows[regression_train_mask], dtype=torch.float32), reg_train_y),
            batch_size=64,
            shuffle=True,
        ),
        torch.tensor(windows[regression_validation_mask], dtype=torch.float32),
        reg_validation_y,
        WeightedHuberLoss(),
        device,
    )

    classifier.eval()
    regressor.eval()
    with torch.no_grad():
        class_prob = torch.sigmoid(classifier(torch.tensor(windows[test_mask], dtype=torch.float32).to(device))).cpu().numpy().ravel()
        reg_test_mask = test_mask & regressor_mask
        reg_pred = regressor(torch.tensor(windows[reg_test_mask], dtype=torch.float32).to(device)).cpu().numpy()
    class_true = classifier_y[test_mask]
    class_pred = (class_prob >= 0.5).astype(np.float32)
    reg_true = times_to_alarm[reg_test_mask]
    reg_pred_seconds = scaler_y.inverse_transform(reg_pred).ravel()
    metrics = {
        "classification_test_windows": int(test_mask.sum()),
        "classification_positive_rate": float(class_true.mean()),
        "precision": precision_score(class_true, class_pred, zero_division=0),
        "recall": recall_score(class_true, class_pred, zero_division=0),
        "f1": f1_score(class_true, class_pred, zero_division=0),
        "pr_auc": average_precision_score(class_true, class_prob),
        "regression_test_windows": int(reg_test_mask.sum()),
        "regression_mae_seconds": float(np.mean(np.abs(reg_true - reg_pred_seconds))),
        "regression_rmse_seconds": float(np.sqrt(np.mean((reg_true - reg_pred_seconds) ** 2))),
        "regression_negative_ratio": float(np.mean(reg_pred_seconds < 0)),
        "early_warning_seconds": EARLY_WARNING_SECONDS,
        "max_regression_target_seconds": y_max,
    }
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "evaluation_metrics.csv", index=False)
    torch.save(classifier.state_dict(), MODEL_DIR / "early_warning_lstm.pt")
    torch.save(regressor.state_dict(), MODEL_DIR / "time_to_alarm_lstm.pt")
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
