"""Re-train the primary VAE configuration under other model seeds (same dataset)."""
import warnings

import numpy as np
import pandas as pd

from data import ROOT, build_dataset, flat, load_config
from evaluate import kmeans_recovery
from train_vae import train_one
import torch

warnings.filterwarnings("ignore")


def main(seeds=(1, 2, 3, 4), beta=0.1, recon="mean"):
    cfg = load_config()
    data = build_dataset(cfg)
    yva, yte = flat(data["val"])[1], flat(data["test"])[1]
    rows = []
    for sd in seeds:
        c = dict(cfg, seed=cfg["seed"] + 100 * sd)
        # dataset stays fixed (built from the original seed); only init/shuffle vary
        model = train_one(c, data, beta, recon, save=False)
        out = {}
        for name, y in (("val", yva), ("test", yte)):
            X = torch.as_tensor(flat(data[name])[0])
            with torch.no_grad():
                mu, _ = model.encode(X)
            _, m = kmeans_recovery(mu.numpy(), y, c["seed"])
            out[name] = m
        rows.append({"model_seed": c["seed"], "val_ari": out["val"]["ari"],
                     "test_ari": out["test"]["ari"], "test_nmi": out["test"]["nmi"],
                     "test_macro_f1": out["test"]["macro_f1"]})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "seed_robustness.csv", index=False)
    print(df.describe().loc[["mean", "std"]].round(3))


if __name__ == "__main__":
    main()
