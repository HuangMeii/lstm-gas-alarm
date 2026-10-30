import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "smoke_detection_iot.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
SEQ_LEN = 30
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
TARGET_COL = "time_to_alarm_sec"


def create_sliding_windows(
    X: np.ndarray, y: np.ndarray, timestamps: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_len):
        window_times = timestamps[i : i + seq_len + 1]
        if np.all(np.diff(window_times) == 1) and not np.isnan(y[i + seq_len]):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len])
    if not X_seq:
        raise ValueError("Không tạo được cửa sổ hợp lệ từ dữ liệu.")
    return np.asarray(X_seq), np.asarray(y_seq)


class FireAlarmLSTM(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return torch.nn.functional.softplus(self.fc(out[:, -1, :]))


def create_time_to_alarm_labels(df: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(df), np.nan, dtype=np.float32)
    timestamps = df["UTC"].to_numpy()
    alarms = df["Fire Alarm"].to_numpy()
    segment_starts = np.r_[0, np.flatnonzero(np.diff(timestamps) != 1) + 1]
    segment_ends = np.r_[segment_starts[1:], len(df)]

    for start, end in zip(segment_starts, segment_ends):
        next_alarm = None
        for i in range(end - 1, start - 1, -1):
            if alarms[i] == 1:
                next_alarm = timestamps[i]
            if next_alarm is not None:
                labels[i] = max(0.0, float(next_alarm - timestamps[i]))
    return labels


def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"1. Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    required_cols = set(FEATURE_COLS + ["Fire Alarm"])
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

    print("2. Creating UTC-based time_to_alarm_sec labels...")
    df[TARGET_COL] = create_time_to_alarm_labels(df)
    split_row = int(len(df) * 0.8)
    train_rows = df.iloc[:split_row]
    valid_train_rows = train_rows.dropna(subset=[TARGET_COL])

    scaler_X = MinMaxScaler().fit(valid_train_rows[FEATURE_COLS])
    scaler_y = MinMaxScaler().fit(valid_train_rows[[TARGET_COL]])
    X_scaled = scaler_X.transform(df[FEATURE_COLS])
    y_scaled = np.full((len(df), 1), np.nan, dtype=np.float32)
    valid_labels = df[TARGET_COL].notna()
    y_scaled[valid_labels] = scaler_y.transform(df.loc[valid_labels, [TARGET_COL]])
    joblib.dump(scaler_X, MODEL_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, MODEL_DIR / "scaler_y.pkl")

    X_win, y_win = create_sliding_windows(
        X_scaled, y_scaled, df["UTC"].to_numpy(), SEQ_LEN
    )
    train_size = int(len(X_win) * 0.8)
    validation_size = int(train_size * 0.2)
    X_train = torch.tensor(X_win[: train_size - validation_size], dtype=torch.float32)
    y_train = torch.tensor(y_win[: train_size - validation_size], dtype=torch.float32)
    X_validation = torch.tensor(
        X_win[train_size - validation_size : train_size], dtype=torch.float32
    )
    y_validation = torch.tensor(
        y_win[train_size - validation_size : train_size], dtype=torch.float32
    )
    X_test = torch.tensor(X_win[train_size:], dtype=torch.float32)
    y_test = torch.tensor(y_win[train_size:], dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FireAlarmLSTM(input_dim=len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print(f"3. Training the LSTM model on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(
                model(X_validation.to(device)), y_validation.to(device)
            ).item()
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"Train MSE: {average_loss:.6f} - Validation MSE: {validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping: validation loss stopped improving.")
                break

    if best_state is None:
        raise RuntimeError("Không lưu được trạng thái mô hình tốt nhất.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()
    actual_seconds = scaler_y.inverse_transform(y_test.numpy()).ravel()
    predicted_seconds = scaler_y.inverse_transform(predictions).ravel()
    train_seconds = scaler_y.inverse_transform(y_train.numpy()).ravel()
    baseline_predictions = np.full_like(actual_seconds, train_seconds.mean())
    metrics = {
        "mae_seconds": mean_absolute_error(actual_seconds, predicted_seconds),
        "rmse_seconds": mean_squared_error(actual_seconds, predicted_seconds) ** 0.5,
        "r2": r2_score(actual_seconds, predicted_seconds),
        "baseline_mae_seconds": mean_absolute_error(
            actual_seconds, baseline_predictions
        ),
        "test_windows": len(actual_seconds),
        "negative_prediction_ratio": float(np.mean(predicted_seconds < 0)),
    }
    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "evaluation_metrics.csv", index=False
    )
    print(
        f"Test MAE: {metrics['mae_seconds']:.2f}s - "
        f"RMSE: {metrics['rmse_seconds']:.2f}s - R2: {metrics['r2']:.4f}"
    )
    torch.save(model.state_dict(), MODEL_DIR / "lstm_gas_alarm_model.pt")
    print(
        "DONE: Saved lstm_gas_alarm_model.pt, scaler_X.pkl, and scaler_y.pkl."
    )


if __name__ == "__main__":
    main()
