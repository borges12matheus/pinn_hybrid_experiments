import argparse
import hashlib
import json
import random
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Helper para a leitura dos parâmetros e paths
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg, Path(args.config)

def hash_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_deterministic(seed):
    torch.set_float32_matmul_precision("highest")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def load_or_create_split(
    df,
    seed,
    split_path,
    dataset_path,
    split_strategy,
    spatial_col,
    n_bins,
    val_frac,
    test_frac,
    version
):
    split_method = f"{split_strategy}_{spatial_col}_{version}"

    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError(
            "val_frac e test_frac devem ser >= 0 e sua soma deve ser menor que 1."
        )

    if split_path.exists():
        with open(split_path, "r") as f:
            payload = json.load(f)

        if payload.get("dataset_path") != str(dataset_path):
            raise ValueError(
                "O split salvo pertence a outro dataset."
            )
        if payload.get("split_method") != split_method:
            raise ValueError(
                "O split salvo pertence a outra convenção experimental."
            )

        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)

        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        if (
            len(all_idx) != len(df)
            or len(np.unique(all_idx)) != len(df)
            or set(all_idx) != set(range(len(df)))
        ):
            raise ValueError(
                "Split salvo não cobre exatamente o dataset atual."
            )

        if payload.get("val_frac") is None or not np.isclose(
            payload["val_frac"], val_frac
        ):
            raise ValueError("O split salvo usa outra fração de validação.")
        if payload.get("test_frac") is None or not np.isclose(
            payload["test_frac"], test_frac
        ):
            raise ValueError("O split salvo usa outra fração de teste.")

        return train_idx, val_idx, test_idx

    if spatial_col not in df.columns:
        raise KeyError(f"Coluna espacial ausente no dataset: {spatial_col}")

    spatial_values = df[spatial_col]
    if spatial_values.isna().any():
        raise ValueError(f"Coluna espacial com valores ausentes: {spatial_col}")

    bin_labels = pd.qcut(
        spatial_values,
        q=min(n_bins, max(1, spatial_values.nunique())),
        labels=False,
        duplicates="drop",
    )

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    for bin_id in np.unique(bin_labels):
        bin_idx = np.flatnonzero(bin_labels.to_numpy() == bin_id)
        if len(bin_idx) == 0:
            continue

        perm = rng.permutation(bin_idx)
        n_val = int(round(len(bin_idx) * val_frac))
        n_test = int(round(len(bin_idx) * test_frac))

        # Mantém pelo menos uma amostra no treino de cada bin.
        n_val = min(n_val, len(bin_idx) - 1)
        n_test = min(n_test, len(bin_idx) - n_val - 1)

        val_idx.extend(perm[:n_val].tolist())
        test_idx.extend(perm[n_val:n_val + n_test].tolist())
        train_idx.extend(perm[n_val + n_test:].tolist())

    train_idx = np.asarray(sorted(train_idx), dtype=np.int64)
    val_idx = np.asarray(sorted(val_idx), dtype=np.int64)
    test_idx = np.asarray(sorted(test_idx), dtype=np.int64)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "split_method": split_method,
        "split_strategy": split_strategy,
        "split_column": spatial_col,
        "split_bins": n_bins,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_samples": len(df),
        "dataset_path": str(dataset_path),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    with open(split_path, "w") as f:
        json.dump(payload, f, indent=2)

    return train_idx, val_idx, test_idx
