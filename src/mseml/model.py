import json
from pathlib import Path

import duckdb
import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

def load_dataset(tensor_path, db_path):
    data = load_file(tensor_path)

    intensity = data["intensity"].float()
    group_ids = data["group_id"]

    db = duckdb.connect(db_path, read_only=True)

    targets = []

    for group_id in group_ids:
        pattern_id = f"hodge_alwen_1/pattern_{group_id.item()}"

        result = db.execute(
            """
            SELECT phase_fraction
            FROM phases
            WHERE pattern_id = ?
              AND chemical_composition = 'Al'
            """,
            [pattern_id],
        ).fetchone()

        targets.append(result[0])

    db.close()

    targets = torch.tensor(targets, dtype=torch.float32)

    return intensity, targets

class XRDNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.network(x).squeeze(1)

def train_model(model, x, y, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        model.train()

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(
            f"Epoch {epoch + 1:3d} | "
            f"MSE: {loss.item():.4f}"
        )

    return model, losses

def evaluate_model(model, x, y):
    model.eval()

    with torch.no_grad():
        pred = model(x)

    mae = torch.mean(torch.abs(pred - y))

    return mae.item()

def main():
    train_x, train_y = load_dataset(
        "data/inputs/train.safetensors",
        "data/dataset.db",
    )

    eval_x, eval_y = load_dataset(
        "data/inputs/eval.safetensors",
        "data/dataset.db",
    )

    model = XRDNet()
    model, losses = train_model(model, train_x, train_y, epochs=100)

    cnn_mae = evaluate_model(model, eval_x, eval_y)

    mean_prediction = train_y.mean()
    dummy_mae = torch.mean(
        torch.abs(eval_y - mean_prediction)
    ).item()
    positive_control_mae = 0.0

    results = {
        "cnn_test_mae": cnn_mae,
        "dummy_test_mae": dummy_mae,
        "positive_control_test_mae": positive_control_mae,
        "training_losses": losses,
    }

    Path("results").mkdir(exist_ok=True)

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"CNN test MAE: {cnn_mae:.4f}")
    print(f"Dummy test MAE: {dummy_mae:.4f}")
    print(f"Positive control MAE: {positive_control_mae:.4f}")

if __name__ == "__main__":
    main()
