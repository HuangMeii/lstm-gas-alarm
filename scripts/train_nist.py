from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
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


def main():
    torch.manual_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_experiments()
    X, y, groups = create_windows(data)
    experiments = np.unique(groups)
    rng = np.random.default_rng(42)
    rng.shuffle(experiments)
    train_groups, valid_groups, test_groups = np.array_split(experiments, [int(len(experiments) * 0.7), int(len(experiments) * 0.85)])
    masks = [np.isin(groups, values) for values in (train_groups, valid_groups, test_groups)]
    scaler = RobustScaler().fit(X[masks[0]].reshape(-1, X.shape[-1]))
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    joblib.dump(scaler, MODEL_DIR / "nist_feature_scaler.pkl")
    warning = (y > 0) & (y <= 300)
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
    regression_train = masks[0] & (y > 0)
    regression_valid = masks[1] & (y > 0)
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
        probabilities = torch.sigmoid(classifier(torch.tensor(X[masks[2]]).to(device))).cpu().numpy().ravel()
        reg_test = masks[2] & (y > 0)
        predictions = target_scaler.inverse_transform(
            regressor(torch.tensor(X[reg_test]).to(device)).cpu().numpy()
        ).ravel()
    actual_class = warning[masks[2]]
    predicted_class = probabilities >= 0.5
    actual_reg = y[reg_test]
    metrics = {
        "experiments": len(experiments),
        "test_experiments": len(test_groups),
        "classification_windows": int(masks[2].sum()),
        "precision": precision_score(actual_class, predicted_class, zero_division=0),
        "recall": recall_score(actual_class, predicted_class, zero_division=0),
        "f1": f1_score(actual_class, predicted_class, zero_division=0),
        "pr_auc": average_precision_score(actual_class, probabilities),
        "regression_windows": int(reg_test.sum()),
        "regression_mae_seconds": float(np.mean(np.abs(actual_reg - predictions))),
    }
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "evaluation_metrics.csv", index=False)
    torch.save(classifier.state_dict(), MODEL_DIR / "early_warning_lstm.pt")
    torch.save(regressor.state_dict(), MODEL_DIR / "time_to_alarm_lstm.pt")
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
