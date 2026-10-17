"""Latent-space interventions ("what would the VAE decode under a different z?").

These are model-based counterfactuals compared with the simulator's own
clamped-state ground truth. They are NOT causal inference about cognition.
"""
import json
import warnings

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from data import ROOT, build_dataset, flat, load_config
from evaluate import lapsed_dimension
from simulate import STATE_NAMES, simulate_fixed_state_windows
from vae import WindowVAE

warnings.filterwarnings("ignore")


def load_model(tag, cfg):
    ck = torch.load(ROOT / "outputs" / f"{tag}.pt", weights_only=False)
    m = WindowVAE(2, cfg["hidden_dim"], cfg["latent_dim"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def decode(model, z, W):
    return model.decode(torch.as_tensor(z, dtype=torch.float32), W).numpy()


def unstandardise(x, scaler):
    x = x.copy()
    x[..., 1] = x[..., 1] * scaler["sd"] + scaler["mean"]
    return x


def behaviour_summary(w, rng=None):
    """w: (n, W, 2) in raw units. Decoded correctness is a continuous probability;
    for sequence statistics it is sampled as Bernoulli(clip(c, 0, 1))."""
    c, r = w[..., 0], w[..., 1]
    if rng is not None:
        b = (rng.random(c.shape) < np.clip(c, 0, 1)).astype(int)
    else:
        b = c.astype(int)
    prev, nxt = b[:, :-1].ravel(), b[:, 1:].ravel()
    return {"mean_correct": float(c.mean()),
            "mean_log_rt": float(r.mean()),
            "within_window_rt_sd": float(r.std(axis=1, ddof=1).mean()),
            "p_correct_given_correct": float(nxt[prev == 1].mean()) if (prev == 1).any() else np.nan,
            "p_correct_given_error": float(nxt[prev == 0].mean()) if (prev == 0).any() else np.nan}


def boot_ci(x, rng, n=2000):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    data = build_dataset(cfg)
    scaler = data["scaler"]
    W = cfg["window_size"]
    tag = json.loads((ROOT / "results" / "primary_model.json").read_text())["primary_by_val_ari"]
    model = load_model(tag, cfg)
    L = np.load(ROOT / "outputs" / f"{tag}_latents.npz")
    _, ytr = flat(data["train"])
    _, yte = flat(data["test"])
    mu_tr, mu_te = L["train_mu"], L["test_mu"]

    dim, effect = lapsed_dimension(mu_tr, ytr)          # training windows only
    target = {s: float(mu_tr[ytr == s, dim].mean()) for s in (0, 2)}
    print(f"model {tag}: lapsed-associated dim = z{dim}; "
          f"standardised lapsed-engaged effect per dim = {effect.round(2)}; "
          f"state-mean z{dim}: engaged {target[0]:.3f}, lapsed {target[2]:.3f}")

    conditions = {"observed (no intervention)": mu_te}
    for s, nm in ((2, "lapsed"), (0, "engaged")):
        z = mu_te.copy()
        z[:, dim] = target[s]
        conditions[f"z{dim} -> {nm} mean"] = z
        zf = np.tile(mu_tr[ytr == s].mean(0), (len(mu_te), 1))
        conditions[f"all z -> {nm} mean"] = zf

    decoded = {k: unstandardise(decode(model, z, W), scaler) for k, z in conditions.items()}

    n_gt = 5000
    truth = {nm: simulate_fixed_state_windows(s, n_gt, rng, W)
             for s, nm in ((0, "engaged"), (1, "exploratory"), (2, "lapsed"))}

    rows = []
    for k, w in decoded.items():
        rows.append({"source": "VAE decoded", "condition": k, **behaviour_summary(w, rng)})
    for nm, w in truth.items():
        rows.append({"source": "simulator, state clamped", "condition": nm, **behaviour_summary(w)})
    # observed test windows by true label, for reference
    Xte = flat({"Xs": data["test"]["X"], "y": data["test"]["y"]})[0]
    for s in (0, 2):
        rows.append({"source": "simulator, observed test windows",
                     "condition": f"true window label = {STATE_NAMES[s]}",
                     **behaviour_summary(Xte[yte == s])})

    # Wasserstein distances between pooled RT distributions (raw log-RT)
    wass = {}
    for dk, gk, nm in ((f"z{dim} -> lapsed mean", "lapsed", "lapsed"),
                       (f"z{dim} -> engaged mean", "engaged", "engaged"),
                       ("all z -> lapsed mean", "lapsed", "lapsed"),
                       ("all z -> engaged mean", "engaged", "engaged"),
                       ("observed (no intervention)", "lapsed", "lapsed"),
                       ("observed (no intervention)", "engaged", "engaged")):
        wass[f"{dk} vs {nm} truth"] = float(wasserstein_distance(
            decoded[dk][..., 1].ravel(), truth[gk][..., 1].ravel()))

    # direction test: lapsed minus engaged, decoded vs simulator
    dl, de = decoded[f"z{dim} -> lapsed mean"], decoded[f"z{dim} -> engaged mean"]
    paired_rt = (dl[..., 1].mean(1) - de[..., 1].mean(1))
    paired_c = (dl[..., 0].mean(1) - de[..., 0].mean(1))
    gt_rt = truth["lapsed"][..., 1].mean() - truth["engaged"][..., 1].mean()
    gt_c = truth["lapsed"][..., 0].mean() - truth["engaged"][..., 0].mean()
    direction = {
        "intervention_dim": dim,
        "decoded_rt_shift_lapsed_minus_engaged": float(paired_rt.mean()),
        "decoded_rt_shift_95ci_over_windows": boot_ci(paired_rt, rng),
        "decoded_correct_shift_lapsed_minus_engaged": float(paired_c.mean()),
        "decoded_correct_shift_95ci_over_windows": boot_ci(paired_c, rng),
        "simulator_rt_shift": float(gt_rt),
        "simulator_correct_shift": float(gt_c),
        "rt_longer_for_lapsed": bool(paired_rt.mean() > 0),
        "correct_lower_for_lapsed": bool(paired_c.mean() < 0),
        "magnitude_ratio_rt": float(paired_rt.mean() / gt_rt),
        "magnitude_ratio_correct": float(paired_c.mean() / gt_c),
    }

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "counterfactual_summary.csv", index=False)
    (ROOT / "results" / "counterfactual_results.json").write_text(
        json.dumps({"model": tag, "state_mean_z": target, "wasserstein_rt": wass,
                    "direction": direction}, indent=2))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(df.round(3).to_string())
    print(json.dumps({"wasserstein_rt": wass, "direction": direction}, indent=2))

    # ---- figure 5 ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"engaged": "tab:blue", "lapsed": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    bins = np.linspace(-0.8, 2.6, 60)
    for nm in ("engaged", "lapsed"):
        axes[0].hist(truth[nm][..., 1].ravel(), bins, density=True, histtype="step",
                     color=cols[nm], lw=2, label=f"simulator: {nm}")
        axes[0].hist(decoded[f"z{dim} -> {nm} mean"][..., 1].ravel(), bins, density=True,
                     color=cols[nm], alpha=0.35, label=f"VAE z{dim}->{nm}")
    axes[0].set_xlabel("log-RT"); axes[0].set_ylabel("density"); axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Decoded vs ground-truth RT")
    x = np.arange(2)
    for ax, key, ttl in ((axes[1], "mean_log_rt", "Mean log-RT"),
                         (axes[2], "mean_correct", "Mean correctness")):
        for j, nm in enumerate(("engaged", "lapsed")):
            g = df[(df.source == "simulator, state clamped") & (df.condition == nm)][key].item()
            v = df[(df.source == "VAE decoded") & (df.condition == f"z{dim} -> {nm} mean")][key].item()
            ax.bar(j - 0.2, g, 0.4, color=cols[nm])
            ax.bar(j + 0.2, v, 0.4, color=cols[nm], alpha=0.4, hatch="//")
        ax.set_xticks(x); ax.set_xticklabels(["engaged", "lapsed"]); ax.set_title(ttl)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="grey", label="simulator (solid)"),
                           Patch(facecolor="grey", alpha=0.4, hatch="//", label="VAE decoded (hatched)")],
                  frameon=False, fontsize=7)
    fig.suptitle(f"Latent intervention on z{dim} ({tag}): model counterfactual, not causal inference", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig5_counterfactuals.png", dpi=200)


if __name__ == "__main__":
    main()
