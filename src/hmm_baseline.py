"""Gaussian-HMM baseline on [correct, standardised log-RT] trial sequences.

Limitation: correctness is binary but is modelled as a Gaussian feature.
The HMM is fit on trial-level sequences (one sequence per subject, windows
concatenated in order); inferred trial states are reduced to a window label by
majority vote, matching how the true window label is defined.
"""
import json
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from data import build_dataset, load_config, set_seeds, ROOT
from evaluate import recovery_metrics, shuffled_control, transition_error
from simulate import TRANSITION_MATRIX

warnings.filterwarnings("ignore")


def _seqs(split):
    """(n_subj, n_win, W, 2) -> concatenated trial array + lengths."""
    n_s, n_w, W, f = split["Xs"].shape
    return split["Xs"].reshape(-1, f), [n_w * W] * n_s


def fit_hmm(train_split, n_states=3, seed=2026, n_init=5):
    X, lengths = _seqs(train_split)
    best, best_ll = None, -np.inf
    for i in range(n_init):
        m = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, random_state=seed + i)
        m.fit(X, lengths)
        ll = m.score(X, lengths)
        if ll > best_ll:
            best, best_ll = m, ll
    return best


def infer_window_states(model, split, n_states=3):
    """Viterbi per subject -> majority-vote window labels, shape (n_subj, n_win)."""
    n_s, n_w, W, f = split["Xs"].shape
    out = np.empty((n_s, n_w), dtype=int)
    trial_states = np.empty((n_s, n_w * W), dtype=int)
    for i in range(n_s):
        ts = model.predict(split["Xs"][i].reshape(-1, f))
        trial_states[i] = ts
        out[i] = [np.bincount(ts[j * W:(j + 1) * W], minlength=n_states).argmax()
                  for j in range(n_w)]
    return out, trial_states


def trial_transition_mae(model, split, k=3):
    """MAE of the fitted HMM transmat vs the simulator's A, after matching."""
    _, ts = infer_window_states(model, split, k)
    true = split["states"]
    cm = np.zeros((k, k))
    np.add.at(cm, (true.ravel(), ts.ravel()), 1)
    r, c = linear_sum_assignment(-cm)
    perm = np.zeros(k, dtype=int)
    perm[r] = c                       # perm[true_state] = hmm_state
    A = model.transmat_[np.ix_(perm, perm)]
    return float(np.abs(A - TRANSITION_MATRIX).mean()), A


class BernoulliGaussianHMM:
    """HMM with Bernoulli(correct) x Gaussian(log-RT) emissions, fit by Baum-Welch.

    A better-specified baseline than GaussianHMM, which treats the binary
    correctness as a Gaussian feature. Sequences are stacked (S, T, 2).
    """

    def __init__(self, n_states=3, n_iter=200, tol=1e-4, seed=0):
        self.K, self.n_iter, self.tol, self.seed = n_states, n_iter, tol, seed

    def _log_emit(self, X):
        c, r = X[..., 0:1], X[..., 1:2]
        p = np.clip(self.p, 1e-4, 1 - 1e-4)
        return (c * np.log(p) + (1 - c) * np.log(1 - p)
                - 0.5 * np.log(2 * np.pi * self.var) - 0.5 * (r - self.mu) ** 2 / self.var)

    def _fb(self, X):
        S, T, _ = X.shape
        le = self._log_emit(X)
        logA, logpi = np.log(self.A), np.log(self.pi)
        la = np.empty((S, T, self.K)); lb = np.zeros((S, T, self.K))
        la[:, 0] = logpi + le[:, 0]
        for t in range(1, T):
            la[:, t] = le[:, t] + logsumexp(la[:, t - 1, :, None] + logA, axis=1)
        for t in range(T - 2, -1, -1):
            lb[:, t] = logsumexp(logA + (le[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
        ll = logsumexp(la[:, -1], axis=1)
        lg = la + lb - ll[:, None, None]
        return le, la, lb, lg, ll

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        flat_ = X.reshape(-1, 2)
        km = KMeans(self.K, n_init=1, random_state=int(rng.integers(1e6))).fit(flat_[:, [1]])
        lab = km.labels_
        self.mu = np.array([[flat_[lab == k, 1].mean() for k in range(self.K)]])
        self.var = np.array([[flat_[lab == k, 1].var() + 1e-3 for k in range(self.K)]])
        self.p = np.array([flat_[lab == k, 0].mean() for k in range(self.K)])
        self.A = np.full((self.K, self.K), 0.1 / (self.K - 1))
        np.fill_diagonal(self.A, 0.9)
        self.pi = np.full(self.K, 1 / self.K)
        prev = -np.inf
        for _ in range(self.n_iter):
            le, la, lb, lg, ll = self._fb(X)
            g = np.exp(lg)                                    # (S,T,K)
            lA = la[:, :-1, :, None] + np.log(self.A) + (le[:, 1:] + lb[:, 1:])[:, :, None, :]
            xi = np.exp(lA - ll[:, None, None, None])         # (S,T-1,K,K)
            self.pi = g[:, 0].mean(0)
            self.A = xi.sum((0, 1)); self.A /= self.A.sum(1, keepdims=True)
            w = g.sum((0, 1))
            self.p = (g * X[..., 0:1]).sum((0, 1)) / w
            self.mu = ((g * X[..., 1:2]).sum((0, 1)) / w)[None]
            self.var = ((g * (X[..., 1:2] - self.mu) ** 2).sum((0, 1)) / w)[None] + 1e-4
            tot = ll.sum()
            if tot - prev < self.tol * abs(tot):
                break
            prev = tot
        self.loglik_ = self._fb(X)[4].sum()
        return self

    def score(self, X):
        return self._fb(X)[4].sum()

    def predict_trials(self, X):
        """Posterior-marginal argmax per trial, shape (S, T)."""
        return self._fb(X)[3].argmax(-1)


def fit_bg_hmm(train_split, n_states=3, seed=2026, n_init=5):
    X = train_split["Xs"].reshape(train_split["Xs"].shape[0], -1, 2).astype(np.float64)
    best = None
    for i in range(n_init):
        m = BernoulliGaussianHMM(n_states, seed=seed + i).fit(X)
        if best is None or m.loglik_ > best.loglik_:
            best = m
    return best


def infer_bg_window_states(model, split, n_states=3):
    n_s, n_w, W, f = split["Xs"].shape
    ts = model.predict_trials(split["Xs"].reshape(n_s, -1, f).astype(np.float64))
    win = np.stack([[np.bincount(ts[i, j * W:(j + 1) * W], minlength=n_states).argmax()
                     for j in range(n_w)] for i in range(n_s)])
    return win, ts


def main():
    cfg = load_config()
    set_seeds(cfg["seed"])
    data = build_dataset(cfg)
    k = cfg["n_states"]
    res = {}

    gm = fit_hmm(data["train"], k, cfg["seed"], cfg["hmm_n_init"])
    bg = fit_bg_hmm(data["train"], k, cfg["seed"], cfg["hmm_n_init"])
    for tag, model, infer in (("hmm_gaussian", gm, infer_window_states),
                              ("hmm_bernoulli_gaussian", bg, infer_bg_window_states)):
        r = {}
        for name in ("val", "test"):
            pred, ts = infer(model, data[name])
            y = data[name]["y"]
            m = recovery_metrics(y.ravel(), pred.ravel())
            m["window_transition_mae"] = transition_error(y, pred)
            r[name] = m
            if name == "test":
                r["trial_state_acc_test"] = float((_match_trials(data[name]["states"], ts)))
                np.save(ROOT / "outputs" / f"{tag}_test_pred.npy", pred)
        res[tag] = r
    res["hmm_gaussian"]["trial_level_A_mae_test"], A = trial_transition_mae(gm, data["test"])
    res["hmm_gaussian"]["recovered_A_test"] = A.round(3).tolist()
    res["hmm_gaussian"]["state_means_std_space"] = gm.means_.round(3).tolist()
    # Bernoulli-Gaussian transition matrix, matched to true states
    ts_bg = infer_bg_window_states(bg, data["test"])[1]
    perm = _perm(data["test"]["states"], ts_bg)
    A_bg = bg.A[np.ix_(perm, perm)]
    res["hmm_bernoulli_gaussian"]["trial_level_A_mae_test"] = float(np.abs(A_bg - TRANSITION_MATRIX).mean())
    res["hmm_bernoulli_gaussian"]["recovered_A_test"] = A_bg.round(3).tolist()
    res["hmm_bernoulli_gaussian"]["p_correct"] = bg.p[perm].round(3).tolist()
    res["hmm_bernoulli_gaussian"]["logrt_mean_std_space"] = bg.mu[0][perm].round(3).tolist()
    res["shuffled_control_test"] = shuffled_control(
        data["test"]["y"].ravel(), np.random.default_rng(cfg["seed"]))

    (ROOT / "results" / "hmm_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def _perm(true, ts, k=3):
    cm = np.zeros((k, k))
    np.add.at(cm, (true.ravel(), ts.ravel()), 1)
    r, c = linear_sum_assignment(-cm)
    perm = np.zeros(k, dtype=int)
    perm[r] = c
    return perm


def _match_trials(true, ts, k=3):
    perm = _perm(true, ts, k)
    inv = np.argsort(perm)
    return (inv[ts] == true).mean()


if __name__ == "__main__":
    main()
