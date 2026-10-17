import numpy as np

from data import build_dataset, load_config
from simulate import (TRANSITION_MATRIX, make_windows, simulate_dataset,
                      simulate_subject, stationary_distribution)


def _data(n_sub=100, n_trials=1000, seed=0):
    return simulate_dataset(n_sub, n_trials, seed=seed)


def test_stationary_proportions():
    ds = _data()
    states = np.concatenate([d["states"] for d in ds])
    emp = np.bincount(states, minlength=3) / len(states)
    assert np.allclose(emp, stationary_distribution(), atol=0.02)


def test_transition_frequencies():
    ds = _data()
    counts = np.zeros((3, 3))
    for d in ds:
        np.add.at(counts, (d["states"][:-1], d["states"][1:]), 1)
    emp = counts / counts.sum(1, keepdims=True)
    assert np.allclose(emp, TRANSITION_MATRIX, atol=0.02)


def test_lapsed_slower_than_engaged():
    ds = _data(20, 500)
    s = np.concatenate([d["states"] for d in ds])
    rt = np.concatenate([d["log_rt"] for d in ds])
    assert rt[s == 2].mean() > rt[s == 0].mean() + 0.5


def test_trial_counts():
    ds = simulate_dataset(7, 300, seed=1)
    assert len(ds) == 7
    for d in ds:
        assert all(len(d[k]) == 300 for k in ("states", "correct", "log_rt"))


def test_seed_reproducible():
    a = simulate_dataset(5, 300, seed=42)
    b = simulate_dataset(5, 300, seed=42)
    c = simulate_dataset(5, 300, seed=43)
    assert all(np.array_equal(x["log_rt"], y["log_rt"]) for x, y in zip(a, b))
    assert not np.array_equal(a[0]["log_rt"], c[0]["log_rt"])


def test_window_shapes_and_labels():
    d = simulate_subject(np.random.default_rng(0), n_trials=300)
    X, y = make_windows(d, 10)
    assert X.shape == (30, 10, 2) and y.shape == (30,)
    for i in range(30):
        seg = d["states"][i * 10:(i + 1) * 10]
        assert y[i] == np.bincount(seg, minlength=3).argmax()
        assert np.array_equal(X[i, :, 0], d["correct"][i * 10:(i + 1) * 10])


def test_split_by_subject_and_scaler():
    d = build_dataset(load_config())
    sets = [set(d[k]["subj"]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert [len(s) for s in sets] == [140, 30, 30]
    tr = d["train"]["Xs"][..., 1]
    assert abs(tr.mean()) < 1e-4 and abs(tr.std() - 1) < 1e-4
