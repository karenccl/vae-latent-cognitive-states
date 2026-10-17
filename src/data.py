"""Windowed dataset with subject-level train/val/test split."""
import random
from pathlib import Path

import numpy as np
import yaml

from simulate import make_windows, simulate_dataset

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def build_dataset(cfg):
    """Return dict split -> {X (raw), Xs (log-RT standardised), y, states, subj}.

    Arrays have shape (n_subjects, n_windows, window_size, 2) so subject
    boundaries are preserved (needed by the HMM). Standardisation of log-RT
    uses training-set mean/SD only; correctness is left as 0/1.
    """
    subjects = simulate_dataset(cfg["n_subjects"], cfg["trials_per_subject"],
                                seed=cfg["seed"])
    X, y, S = [], [], []
    for s in subjects:
        xw, yw = make_windows(s, cfg["window_size"], cfg["n_states"])
        X.append(xw)
        y.append(yw)
        S.append(s["states"])
    X, y, S = np.stack(X), np.stack(y), np.stack(S)

    rng = np.random.default_rng(cfg["seed"] + 1)
    perm = rng.permutation(cfg["n_subjects"])
    n_tr = int(round(cfg["train_prop"] * cfg["n_subjects"]))
    n_va = int(round(cfg["val_prop"] * cfg["n_subjects"]))
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va],
           "test": perm[n_tr + n_va:]}

    train_rt = X[idx["train"]][..., 1]
    mean, sd = float(train_rt.mean()), float(train_rt.std())

    out = {"scaler": {"mean": mean, "sd": sd}}
    for name, ix in idx.items():
        Xs = X[ix].copy()
        Xs[..., 1] = (Xs[..., 1] - mean) / sd
        out[name] = {"X": X[ix], "Xs": Xs, "y": y[ix], "states": S[ix],
                     "subj": ix}
    return out


def flat(split):
    """Flatten (n_subj, n_win, ...) -> (n_subj*n_win, ...) for standardised X, y."""
    return (split["Xs"].reshape(-1, *split["Xs"].shape[2:]),
            split["y"].reshape(-1))
