"""Shared recovery metrics (Hungarian-matched) and VAE evaluation / diagnostics."""
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             f1_score, normalized_mutual_info_score)


def hungarian_map(y_true, y_pred, k=3):
    """Return array m with m[pred_label] = matched true label."""
    cm = confusion_matrix(y_true, y_pred, labels=range(k))  # rows true, cols pred
    r, c = linear_sum_assignment(-cm)
    m = np.zeros(k, dtype=int)
    m[c] = r
    return m


def window_transition_matrix(y, k=3):
    """Row-normalised window-to-window transitions; y has shape (n_subj, n_win)."""
    counts = np.zeros((k, k))
    np.add.at(counts, (y[:, :-1].ravel(), y[:, 1:].ravel()), 1)
    return counts / np.maximum(counts.sum(1, keepdims=True), 1)


def recovery_metrics(y_true, y_pred, k=3):
    """ARI, NMI, macro-F1 (after Hungarian matching) on flat label vectors."""
    m = hungarian_map(y_true, y_pred, k)
    matched = m[y_pred]
    return {"ari": adjusted_rand_score(y_true, y_pred),
            "nmi": normalized_mutual_info_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, matched, average="macro",
                                 labels=list(range(k)))}


def transition_error(y_true_2d, y_pred_2d, k=3):
    """MAE between window-level transition matrices (matched pred vs true)."""
    m = hungarian_map(y_true_2d.ravel(), y_pred_2d.ravel(), k)
    return float(np.abs(window_transition_matrix(m[y_pred_2d], k)
                        - window_transition_matrix(y_true_2d, k)).mean())


def shuffled_control(y_true, rng, n_perm=100, k=3):
    """Metrics when the true labels are randomly permuted (negative control)."""
    res = []
    for _ in range(n_perm):
        res.append(recovery_metrics(y_true, rng.permutation(y_true), k))
    return {key: float(np.mean([r[key] for r in res])) for key in res[0]}


# ----------------------------------------------------------------- VAE side

def kmeans_recovery(z, y, seed, k=3):
    pred = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
    return pred, recovery_metrics(y, pred, k)


def probe(z_tr, y_tr, z_te, y_te):
    # z-scored so that tiny-scale (near-collapsed) latents are probed fairly
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(z_tr, y_tr)
    p = clf.predict(z_te)
    return {"acc": accuracy_score(y_te, p),
            "macro_f1": f1_score(y_te, p, average="macro")}


def collapse_diagnostics(mu, logvar, z_tr, y_tr, z_te, y_te, kl_floor=0.01):
    """Per-dimension posterior variance, KL and single-feature probe accuracy."""
    post_var = np.exp(logvar).mean(0)               # mean posterior sigma^2
    agg_var = mu.var(0)                             # spread of means across windows
    kl = 0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar).mean(0)
    probe_acc = np.array([
        probe(z_tr[:, [d]], y_tr, z_te[:, [d]], y_te)["acc"]
        for d in range(mu.shape[1])])
    chance = np.bincount(y_te).max() / len(y_te)    # majority-class accuracy
    collapsed = (kl < kl_floor) & (probe_acc < chance + 0.02)
    return {"post_var": post_var, "agg_mu_var": agg_var, "kl": kl,
            "probe_acc": probe_acc, "majority_acc": chance,
            "collapsed": collapsed}


def lapsed_dimension(z_tr, y_tr):
    """Latent dim with largest standardised lapsed-vs-engaged mean difference."""
    a, b = z_tr[y_tr == 2], z_tr[y_tr == 0]
    pooled = np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-12
    d = (a.mean(0) - b.mean(0)) / pooled
    return int(np.argmax(np.abs(d))), d


# ------------------------------------------------------------- main driver
def main():
    import json
    warnings.filterwarnings("ignore")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from data import ROOT, build_dataset, flat, load_config
    from simulate import STATE_NAMES

    cfg = load_config()
    data = build_dataset(cfg)
    seed = cfg["seed"]
    ytr, yva, yte = (flat(data[s])[1] for s in ("train", "val", "test"))
    hmm = json.loads((ROOT / "results" / "hmm_results.json").read_text())

    rows, detail, runs = [], [], {}
    for tag, label in (("hmm_gaussian", "HMM (Gaussian, spec)"),
                       ("hmm_bernoulli_gaussian", "HMM (Bernoulli-Gaussian)")):
        t = hmm[tag]["test"]
        rows.append({"model": label, "recon": "", "beta": "", "val_ari": hmm[tag]["val"]["ari"],
                     "test_ari": t["ari"], "test_nmi": t["nmi"], "test_macro_f1": t["macro_f1"],
                     "transition_error": t["window_transition_mae"],
                     "probe_acc": "", "probe_macro_f1": "",
                     "n_collapsed_dims": "", "total_kl": "", "latent_dims_active": ""})

    for recon in ("mean", "sum"):
        for beta in cfg["betas"]:
            tag = f"vae_{recon}_beta{beta:g}"
            L = np.load(ROOT / "outputs" / f"{tag}_latents.npz")
            mu = {s: L[f"{s}_mu"] for s in ("train", "val", "test")}
            _, m_va = kmeans_recovery(mu["val"], yva, seed)
            pred_te, m_te = kmeans_recovery(mu["test"], yte, seed)
            n_s, n_w = data["test"]["y"].shape
            terr = transition_error(data["test"]["y"], pred_te.reshape(n_s, n_w))
            pr = probe(mu["train"], ytr, mu["test"], yte)
            dg = collapse_diagnostics(L["test_mu"], L["test_logvar"], mu["train"], ytr,
                                      mu["test"], yte)
            rows.append({"model": "VAE", "recon": recon, "beta": beta, "val_ari": m_va["ari"],
                         "test_ari": m_te["ari"], "test_nmi": m_te["nmi"],
                         "test_macro_f1": m_te["macro_f1"], "transition_error": terr,
                         "probe_acc": pr["acc"], "probe_macro_f1": pr["macro_f1"],
                         "n_collapsed_dims": int(dg["collapsed"].sum()),
                         "total_kl": float(dg["kl"].sum()),
                         "latent_dims_active": int((dg["kl"] >= 0.01).sum())})
            for d in range(mu["test"].shape[1]):
                detail.append({"recon": recon, "beta": beta, "dim": d,
                               "posterior_var": dg["post_var"][d], "agg_mu_var": dg["agg_mu_var"][d],
                               "kl": dg["kl"][d], "single_dim_probe_acc": dg["probe_acc"][d],
                               "majority_acc": dg["majority_acc"], "collapsed": bool(dg["collapsed"][d])})
            runs[tag] = {"pred": pred_te, "mu": mu}

    summary = pd.DataFrame(rows)
    # primary VAE chosen on VALIDATION ARI only
    vae_rows = summary[summary.model == "VAE"]
    live = vae_rows[vae_rows.latent_dims_active > 0]     # exclude fully collapsed runs
    best = live.loc[live.val_ari.idxmax()]
    primary = f"vae_{best.recon}_beta{best.beta:g}"
    # spec-loss VAE with best val ARI, for the as-specified comparison
    spec = vae_rows[vae_rows.recon == "mean"]
    spec_best = spec.loc[spec.val_ari.idxmax()]
    summary.to_csv(ROOT / "results" / "summary.csv", index=False)
    pd.DataFrame(detail).to_csv(ROOT / "results" / "collapse_diagnostics.csv", index=False)
    rng = np.random.default_rng(seed)
    ctl = shuffled_control(yte, rng)
    (ROOT / "results" / "primary_model.json").write_text(json.dumps({
        "primary_by_val_ari": primary,
        "spec_loss_best_by_val_ari": f"vae_mean_beta{spec_best.beta:g}",
        "shuffled_control_test": ctl}, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summary.round(3).to_string())
    print("primary:", primary, "| shuffled control:", {k: round(v, 4) for k, v in ctl.items()})
    print(pd.DataFrame(detail).round(3).to_string())

    # ---- figures 2-4 ------------------------------------------------------
    fig_dir = ROOT / "figures"
    cols = ["test_ari", "test_nmi", "test_macro_f1"]
    short = {"HMM (Gaussian, spec)": "HMM\nGaussian\n(spec)",
             "HMM (Bernoulli-Gaussian)": "HMM\nBernoulli-\nGaussian"}
    labels = [short[r.model] if r.model in short else f"VAE {r.recon}\nbeta={r.beta:g}"
              for r in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.27
    for i, c in enumerate(cols):
        ax.bar(np.arange(len(summary)) + (i - 1) * w, summary[c].astype(float), w,
               label=c.replace("test_", "").upper())
    ax.set_xticks(range(len(summary))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test-set score"); ax.legend(frameon=False)
    ax.set_title("State recovery on held-out subjects (window level)")
    fig.text(0.99, 0.005, f"Shuffled-label control: ARI {ctl['ari']:.3f}, NMI {ctl['nmi']:.3f}",
             ha="right", va="bottom", fontsize=7, color="dimgrey")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_recovery_comparison.png", dpi=200); plt.close(fig)

    mu_te = runs[primary]["mu"]["test"]
    try:
        import umap
        emb = umap.UMAP(n_components=2, random_state=seed).fit_transform(mu_te)
        title = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        emb = TSNE(2, random_state=seed).fit_transform(mu_te); title = "t-SNE"
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for s, nm in enumerate(STATE_NAMES):
        ax.scatter(*emb[yte == s].T, s=6, alpha=0.6, label=nm)
    ax.legend(frameon=False, markerscale=2); ax.set_title(f"{title} of test latents ({primary})")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_latent_embedding.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, mu_te.shape[1], figsize=(3 * mu_te.shape[1], 3), sharey=False)
    for d, ax in enumerate(axes):
        for s, nm in enumerate(STATE_NAMES):
            ax.hist(mu_te[yte == s, d], bins=30, alpha=0.5, density=True, label=nm)
        ax.set_title(f"z{d}"); ax.set_xlabel("posterior mean")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Latent dimensions by true state ({primary})")
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_latent_distributions.png", dpi=200); plt.close(fig)

    # training curves
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for j, recon in enumerate(("mean", "sum")):
        for k, beta in enumerate(cfg["betas"]):
            h = json.loads((ROOT / "results" / f"vae_{recon}_beta{beta:g}_history.json").read_text())
            ax = axes[j, k]
            ax.plot(h["train_total"], label="train"); ax.plot(h["val_total"], label="val")
            ax.axvline(h["best_epoch"], color="grey", ls=":")
            ax.set_title(f"recon={recon}, beta={beta:g}", fontsize=9)
            ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / "figS1_training_curves.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
