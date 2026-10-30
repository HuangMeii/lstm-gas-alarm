from pathlib import Path
import sys
import json
import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nist_pipeline.config import MODEL_DIR, REPORT_DIR
from nist_pipeline.dataset import create_windows, load_experiments


class NistLSTM(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(2, 32, 1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


def train(model, loader, validation, loss_fn, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, state = float("inf"), None
    for _ in range(10):
        model.train()
        for X, y in loader:
            optimizer.zero_grad()
            loss_fn(model(X.to(device)), y.to(device)).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = loss_fn(model(validation[0].to(device)), validation[1].to(device)).item()
        if loss < best:
            best, state = loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if state is None:
        raise RuntimeError("NIST model did not produce a checkpoint.")
    model.load_state_dict(state)
    return model


def scenario_key(experiment_id: str) -> str:
    if "-" in experiment_id:
        parts = experiment_id.split("-")
        return "-".join(parts[:2])
    match = re.match(r"([A-Za-z]+)", experiment_id)
    return match.group(1) if match else experiment_id


def apply_debounce(
    probabilities: np.ndarray, times: np.ndarray, threshold: float
) -> np.ndarray:
    predictions = np.zeros(len(probabilities), dtype=bool)
    for i in range(len(probabilities)):
        start = max(0, i - 4)
        recent = probabilities[start : i + 1] >= threshold
        predictions[i] = recent.sum() >= 3
    return predictions


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (probabilities < upper)
        if selected.any():
            error += selected.mean() * abs(
                probabilities[selected].mean() - labels[selected].mean()
            )
    return float(error)


def main():
    torch.manual_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_experiments()
    X, y, groups, window_times = create_windows(data)
    experiments = np.unique(groups)
    scenarios = np.array([scenario_key(value) for value in experiments])
    scenario_values = np.unique(scenarios)
    rng = np.random.default_rng(42)
    rng.shuffle(scenario_values)
    train_scenarios, valid_scenarios, test_scenarios = np.array_split(
        scenario_values,
        [int(len(scenario_values) * 0.7), int(len(scenario_values) * 0.85)],
    )
    experiment_scenarios = dict(zip(experiments, scenarios))
    window_scenarios = np.array([experiment_scenarios[value] for value in groups])
    masks = [
        np.isin(window_scenarios, values)
        for values in (train_scenarios, valid_scenarios, test_scenarios)
    ]
    scaler = RobustScaler().fit(X[masks[0]].reshape(-1, X.shape[-1]))
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    joblib.dump(scaler, MODEL_DIR / "nist_feature_scaler.pkl")
    warning = np.isfinite(y) & (y > 0) & (y <= 300)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = NistLSTM(1).to(device)
    train_y = torch.tensor(warning[masks[0], None], dtype=torch.float32)
    valid_y = torch.tensor(warning[masks[1], None], dtype=torch.float32)
    positive = max(1, int(train_y.sum()))
    negative = max(1, len(train_y) - positive)
    classifier = train(
        classifier,
        DataLoader(TensorDataset(torch.tensor(X[masks[0]]), train_y), batch_size=128, shuffle=True),
        (torch.tensor(X[masks[1]]), valid_y),
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device)),
        device,
    )
    regression_train = masks[0] & np.isfinite(y) & (y > 0)
    regression_valid = masks[1] & np.isfinite(y) & (y > 0)
    target_scaler = RobustScaler().fit(y[regression_train, None])
    joblib.dump(target_scaler, MODEL_DIR / "nist_target_scaler.pkl")
    regressor = train(
        NistLSTM(1).to(device),
        DataLoader(
            TensorDataset(
                torch.tensor(X[regression_train]),
                torch.tensor(target_scaler.transform(y[regression_train, None]), dtype=torch.float32),
            ),
            batch_size=128,
            shuffle=True,
        ),
        (
            torch.tensor(X[regression_valid]),
            torch.tensor(target_scaler.transform(y[regression_valid, None]), dtype=torch.float32),
        ),
        nn.HuberLoss(),
        device,
    )
    with torch.no_grad():
        validation_logits = classifier(
            torch.tensor(X[masks[1]]).to(device)
        ).cpu().numpy().ravel()
        test_logits = classifier(
            torch.tensor(X[masks[2]]).to(device)
        ).cpu().numpy().ravel()
    calibrator = LogisticRegression().fit(
        validation_logits[:, None], warning[masks[1]].astype(int)
    )
    validation_probabilities = calibrator.predict_proba(
        validation_logits[:, None]
    )[:, 1]
    probabilities = calibrator.predict_proba(test_logits[:, None])[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_rows = []
    for threshold in thresholds:
        prediction = validation_probabilities >= threshold
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    warning[masks[1]], prediction, zero_division=0
                ),
                "recall": recall_score(
                    warning[masks[1]], prediction, zero_division=0
                ),
                "f1": f1_score(warning[masks[1]], prediction, zero_division=0),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    acceptable = threshold_table[threshold_table["recall"] >= 0.9]
    selected = (
        acceptable.sort_values(["precision", "threshold"], ascending=[False, True])
        .iloc[0]
        if not acceptable.empty
        else threshold_table.sort_values("f1", ascending=False).iloc[0]
    )
    selected_threshold = float(selected["threshold"])
    joblib.dump(calibrator, MODEL_DIR / "nist_probability_calibrator.pkl")
    threshold_table.to_csv(REPORT_DIR / "threshold_sweep.csv", index=False)
    (REPORT_DIR / "selected_threshold.json").write_text(
        json.dumps(
            {
                "threshold": selected_threshold,
                "selection_rule": "highest precision with validation recall >= 0.90",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    predicted_class = apply_debounce(
        probabilities, window_times[masks[2]], selected_threshold
    )
    calibration_metrics = {
        "brier_score": brier_score_loss(
            warning[masks[1]], validation_probabilities
        ),
        "expected_calibration_error": expected_calibration_error(
            warning[masks[1]], validation_probabilities
        ),
        "validation_precision_at_threshold": float(selected["precision"]),
        "validation_recall_at_threshold": float(selected["recall"]),
    }
    pd.DataFrame([calibration_metrics]).to_csv(
        REPORT_DIR / "calibration_metrics.csv", index=False
    )
    with torch.no_grad():
        reg_test = masks[2] & np.isfinite(y) & (y > 0)
        predictions = target_scaler.inverse_transform(
            regressor(torch.tensor(X[reg_test]).to(device)).cpu().numpy()
        ).ravel()
    actual_class = warning[masks[2]]
    actual_reg = y[reg_test]
    test_frame = pd.DataFrame(
        {
            "experiment_id": groups[masks[2]],
            "time_sec": window_times[masks[2]],
            "time_to_alarm_sec": y[masks[2]],
            "probability": probabilities,
            "prediction": predicted_class,
        }
    )
    experiment_rows = []
    for experiment_id, frame in test_frame.groupby("experiment_id"):
        has_alarm = frame["time_to_alarm_sec"].notna().any()
        predicted = bool(frame["prediction"].any())
        experiment_rows.append(
            {
                "experiment_id": experiment_id,
                "has_alarm": has_alarm,
                "predicted_alarm": predicted,
                "false_alarm": predicted and not has_alarm,
                "event_detected": predicted and has_alarm,
                "max_probability": float(frame["probability"].max()),
            }
        )
    experiment_table = pd.DataFrame(experiment_rows)
    negative_experiments = experiment_table[~experiment_table["has_alarm"]]
    false_alarm_rate = (
        float(negative_experiments["false_alarm"].mean())
        if len(negative_experiments)
        else float("nan")
    )
    false_alarm_metrics = {
        "threshold": selected_threshold,
        "negative_experiments": len(negative_experiments),
        "false_alarm_experiments": int(negative_experiments["false_alarm"].sum()),
        "experiment_false_alarm_rate": false_alarm_rate,
        "alarm_experiments": int(experiment_table["has_alarm"].sum()),
        "event_recall": float(
            experiment_table.loc[experiment_table["has_alarm"], "event_detected"].mean()
        ),
    }
    negative_duration_hours = 0.0
    for experiment_id in negative_experiments["experiment_id"]:
        frame = test_frame[test_frame["experiment_id"].eq(experiment_id)]
        if len(frame) > 1:
            negative_duration_hours += (
                frame["time_sec"].max() - frame["time_sec"].min()
            ) / 3600.0
    false_alarm_metrics["false_alarms_per_hour"] = (
        float(negative_experiments["false_alarm"].sum() / negative_duration_hours)
        if negative_duration_hours > 0
        else float("nan")
    )
    pd.DataFrame([false_alarm_metrics]).to_csv(
        REPORT_DIR / "false_alarm_metrics.csv", index=False
    )
    experiment_table.to_csv(REPORT_DIR / "experiment_predictions.csv", index=False)
    horizon_rows = []
    for lower, upper in [(0, 10), (10, 30), (30, 60), (60, 300), (300, np.inf)]:
        selected_horizon = reg_test & (y > lower) & (y <= upper)
        if selected_horizon.any():
            actual_horizon = y[selected_horizon]
            prediction_horizon = target_scaler.inverse_transform(
                regressor(
                    torch.tensor(X[selected_horizon]).to(device)
                ).detach().cpu().numpy()
            ).ravel()
            horizon_rows.append(
                {
                    "lower_seconds": lower,
                    "upper_seconds": upper,
                    "windows": int(selected_horizon.sum()),
                    "mae_seconds": float(
                        np.mean(np.abs(actual_horizon - prediction_horizon))
                    ),
                    "rmse_seconds": float(
                        np.sqrt(np.mean((actual_horizon - prediction_horizon) ** 2))
                    ),
                }
            )
    pd.DataFrame(horizon_rows).to_csv(
        REPORT_DIR / "regression_metrics_by_horizon.csv", index=False
    )
    metrics = {
        "experiments": len(experiments),
        "test_experiments": len(np.unique(groups[masks[2]])),
        "test_scenarios": len(test_scenarios),
        "unseen_test_scenarios": ",".join(sorted(test_scenarios)),
        "classification_windows": int(masks[2].sum()),
        "threshold": selected_threshold,
        "precision": precision_score(actual_class, predicted_class, zero_division=0),
        "recall": recall_score(actual_class, predicted_class, zero_division=0),
        "f1": f1_score(actual_class, predicted_class, zero_division=0),
        "pr_auc": average_precision_score(actual_class, probabilities),
        "regression_windows": int(reg_test.sum()),
        "regression_mae_seconds": float(np.mean(np.abs(actual_reg - predictions))),
        "experiment_false_alarm_rate": false_alarm_rate,
        "event_recall": false_alarm_metrics["event_recall"],
    }
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "evaluation_metrics.csv", index=False)
    torch.save(classifier.state_dict(), MODEL_DIR / "early_warning_lstm.pt")
    torch.save(regressor.state_dict(), MODEL_DIR / "time_to_alarm_lstm.pt")
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
