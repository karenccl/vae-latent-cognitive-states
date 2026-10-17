"""Hidden-state simulator: 3-state Markov chain emitting [correct, log_rt]."""
import numpy as np

STATE_NAMES = ["engaged", "exploratory", "lapsed"]

TRANSITION_MATRIX = np.array([
    [0.90, 0.07, 0.03],
    [0.10, 0.80, 0.10],
    [0.10, 0.20, 0.70],
])

STATE_PARAMS = {
    0: {"p_correct": 0.85, "log_rt_mean": 0.20, "log_rt_sd": 0.25},
    1: {"p_correct": 0.65, "log_rt_mean": 0.40, "log_rt_sd": 0.30},
    2: {"p_correct": 0.50, "log_rt_mean": 0.90, "log_rt_sd": 0.45},
}


def stationary_distribution(transition_matrix=TRANSITION_MATRIX):
    vals, vecs = np.linalg.eig(transition_matrix.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def simulate_subject(rng, n_trials=300, initial_state=0,
                     transition_matrix=None, state_params=None):
    if transition_matrix is None:
        transition_matrix = TRANSITION_MATRIX
    if state_params is None:
        state_params = STATE_PARAMS

    states = np.empty(n_trials, dtype=int)
    correct = np.empty(n_trials, dtype=int)
    log_rt = np.empty(n_trials)
    states[0] = initial_state

    for t in range(n_trials):
        s = states[t]
        params = state_params[s]
        correct[t] = rng.binomial(1, params["p_correct"])
        log_rt[t] = rng.normal(params["log_rt_mean"], params["log_rt_sd"])
        if t < n_trials - 1:
            states[t + 1] = rng.choice(3, p=transition_matrix[s])

    return {"states": states, "correct": correct, "log_rt": log_rt}


def simulate_dataset(n_subjects=200, n_trials=300, seed=2026, **kwargs):
    """List of per-subject dicts, generated from one seeded RNG."""
    rng = np.random.default_rng(seed)
    return [simulate_subject(rng, n_trials=n_trials, **kwargs)
            for _ in range(n_subjects)]


def make_windows(data, window_size=10, n_states=3):
    n = len(data["states"])
    X, y = [], []
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        X.append(np.column_stack([data["correct"][start:end],
                                  data["log_rt"][start:end]]))
        y.append(np.bincount(data["states"][start:end],
                             minlength=n_states).argmax())
    return np.array(X, dtype=np.float32), np.array(y)


def simulate_fixed_state_windows(state, n_windows, rng, window_size=10,
                                 state_params=None):
    """Ground-truth counterfactual: i.i.d. windows with the hidden state clamped."""
    p = (state_params or STATE_PARAMS)[state]
    correct = rng.binomial(1, p["p_correct"], size=(n_windows, window_size))
    log_rt = rng.normal(p["log_rt_mean"], p["log_rt_sd"],
                        size=(n_windows, window_size))
    return np.stack([correct, log_rt], axis=-1).astype(np.float32)


def plot_example_subject(path, seed=2026):
    """Figure 1: one simulated subject's true states, correctness and log-RT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = simulate_subject(np.random.default_rng(seed), n_trials=300)
    t = np.arange(300)
    fig, ax = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True)
    ax[0].step(t, d["states"], where="mid", color="k")
    ax[0].set_yticks([0, 1, 2]); ax[0].set_yticklabels(STATE_NAMES); ax[0].set_ylabel("hidden state")
    ax[1].vlines(t[d["correct"] == 1], 0.5, 1, color="tab:green", lw=0.8)
    ax[1].vlines(t[d["correct"] == 0], 0, 0.5, color="tab:red", lw=0.8)
    ax[1].set_yticks([0.25, 0.75]); ax[1].set_yticklabels(["error", "correct"]); ax[1].set_ylabel("choice")
    ax[2].plot(t, d["log_rt"], color="tab:blue", lw=0.8); ax[2].set_ylabel("log-RT"); ax[2].set_xlabel("trial")
    for w in range(0, 300, 10):
        for a in ax:
            a.axvline(w, color="grey", lw=0.3, alpha=0.4)
    fig.suptitle("Example simulated subject (grey lines: 10-trial windows)")
    fig.tight_layout(); fig.savefig(path, dpi=200)


if __name__ == "__main__":
    from data import ROOT
    plot_example_subject(ROOT / "figures" / "fig1_example_subject.png")
