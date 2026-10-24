import sys

import duckdb
import torch
from safetensors.torch import load_file


def main(db_path, train_path, eval_path):
    train = load_file(train_path)
    eval_data = load_file(eval_path)

    required = {
        "two_theta",
        "intensity",
        "mask",
        "group_id",
        "cv_fold",
    }

    if set(train) != required:
        raise RuntimeError(f"Unexpected train tensors: {set(train)}")

    if set(eval_data) != required:
        raise RuntimeError(f"Unexpected eval tensors: {set(eval_data)}")

    # Tensor dimensions must agree.
    for data, name in [(train, "train"), (eval_data, "eval")]:
        if data["two_theta"].shape != data["intensity"].shape:
            raise RuntimeError(f"{name}: two_theta/intensity shape mismatch")

        if data["mask"].shape != data["intensity"].shape:
            raise RuntimeError(f"{name}: mask shape mismatch")

        if len(data["group_id"]) != len(data["intensity"]):
            raise RuntimeError(f"{name}: group count mismatch")

        if len(data["cv_fold"]) != len(data["intensity"]):
            raise RuntimeError(f"{name}: fold count mismatch")

    # No train/eval group leakage.
    train_groups = set(train["group_id"].tolist())
    eval_groups = set(eval_data["group_id"].tolist())

    overlap = train_groups & eval_groups

    if overlap:
        raise RuntimeError(
            f"Train/eval leakage detected: {sorted(overlap)}"
        )

    # Evaluation samples must not belong to CV folds.
    if not torch.all(eval_data["cv_fold"] == -1):
        raise RuntimeError("Evaluation samples have CV fold assignments")

    # Training folds must be 0-4.
    if not torch.all(
        (train["cv_fold"] >= 0) & (train["cv_fold"] < 5)
    ):
        raise RuntimeError("Invalid training CV fold")

    # Every database pattern should appear exactly once.
    db = duckdb.connect(db_path, read_only=True)

    n_patterns = db.execute(
        "SELECT COUNT(*) FROM patterns"
    ).fetchone()[0]

    db.close()

    n_tensor_patterns = len(train_groups) + len(eval_groups)

    if n_tensor_patterns != n_patterns:
        raise RuntimeError(
            f"Database contains {n_patterns} patterns, "
            f"but tensors contain {n_tensor_patterns}"
        )

    print("[check] SafeTensor files load correctly")
    print("[check] Tensor shapes are consistent")
    print("[check] No train/eval group leakage")
    print("[check] Cross-validation folds are valid")
    print(f"[check] All {n_patterns} patterns accounted for")
    print("[check] PASS")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
