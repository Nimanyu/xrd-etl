import hashlib
import sys
from pathlib import Path

import duckdb
import torch
from safetensors.torch import save_file


SEED = 42
EVAL_FRACTION = 0.20
N_FOLDS = 5


def split_key(pattern_id):
    """Stable pseudo-random key for deterministic splitting."""
    value = f"{SEED}:{pattern_id}".encode()
    return hashlib.sha256(value).hexdigest()


def main(db_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = duckdb.connect(db_path, read_only=True)

    # Get pattern IDs deterministically.
    pattern_ids = [
        row[0]
        for row in db.execute(
            """
            SELECT pattern_id
            FROM patterns
            ORDER BY pattern_id
            """
        ).fetchall()
    ]

    if not pattern_ids:
        raise RuntimeError("No patterns found")

    # Deterministic group-level split.
    ordered_groups = sorted(pattern_ids, key=split_key)

    n_eval = max(1, round(len(ordered_groups) * EVAL_FRACTION))

    eval_ids = set(ordered_groups[:n_eval])
    train_ids = ordered_groups[n_eval:]

    # Assign CV folds only within training data.
    fold_map = {
        pattern_id: i % N_FOLDS
        for i, pattern_id in enumerate(train_ids)
    }

    def build_split(ids, is_eval, max_points):
        two_theta_tensor = torch.zeros(
            (len(ids), max_points), dtype=torch.float32
        )
        intensity_tensor = torch.zeros(
            (len(ids), max_points), dtype=torch.float32
        )
        mask_tensor = torch.zeros(
            (len(ids), max_points), dtype=torch.bool
        )
    
        group_ids = []
        folds = []
    
        group_lookup = {
            pattern_id: i
            for i, pattern_id in enumerate(pattern_ids)
        }
    
        for sample_index, pattern_id in enumerate(ids):
            points = db.execute(
                """
                SELECT two_theta, intensity
                FROM pattern_points
                WHERE pattern_id = ?
                ORDER BY point_index
                """,
                [pattern_id],
            ).fetchall()
    
            n = len(points)
    
            two_theta = torch.tensor(
                [row[0] for row in points],
                dtype=torch.float32,
            )
    
            intensity = torch.tensor(
                [row[1] for row in points],
                dtype=torch.float32,
            )
    
            two_theta_tensor[sample_index, :n] = two_theta
            intensity_tensor[sample_index, :n] = intensity
            mask_tensor[sample_index, :n] = True
    
            group_ids.append(group_lookup[pattern_id])
    
            if is_eval:
                folds.append(-1)
            else:
                folds.append(fold_map[pattern_id])
    
        return {
            "two_theta": two_theta_tensor,
            "intensity": intensity_tensor,
            "mask": mask_tensor,
            "group_id": torch.tensor(group_ids, dtype=torch.int64),
            "cv_fold": torch.tensor(folds, dtype=torch.int64),
        }

    max_points = db.execute(
        """
        SELECT MAX(n)
        FROM (
            SELECT pattern_id, COUNT(*) AS n
            FROM pattern_points
            GROUP BY pattern_id
        )
        """
    ).fetchone()[0]
    
    print(f"[tensors] Maximum points per pattern: {max_points}")

    train = build_split(
        train_ids,
        is_eval=False,
        max_points=max_points,
    )
    
    eval_data = build_split(
        sorted(eval_ids),
        is_eval=True,
        max_points=max_points,
    )

    save_file(train, output_dir / "train.safetensors")
    save_file(eval_data, output_dir / "eval.safetensors")

    print(f"[tensors] Train patterns: {len(train_ids)}")
    print(f"[tensors] Eval patterns: {len(eval_ids)}")
    print(f"[tensors] Train shape: {tuple(train['intensity'].shape)}")
    print(f"[tensors] Eval shape: {tuple(eval_data['intensity'].shape)}")

    db.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
