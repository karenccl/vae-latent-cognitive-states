"""Train the windowed VAE for each beta; early-stop on validation loss."""
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import ROOT, build_dataset, flat, load_config, set_seeds
from vae import WindowVAE, vae_loss


def tag(beta, recon):
    return f"vae_{recon}_beta{beta:g}"


@torch.no_grad()
def encode_all(model, X):
    model.eval()
    x = torch.as_tensor(X)
    mu, logvar = model.encode(x)
    return mu.numpy(), logvar.numpy(), model.decode(mu, x.size(1)).numpy()


def train_one(cfg, data, beta, recon="mean", save=True):
    set_seeds(cfg["seed"])
    Xtr, _ = flat(data["train"])
    Xva, _ = flat(data["val"])
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(TensorDataset(torch.as_tensor(Xtr)), shuffle=True,
                        batch_size=cfg["batch_size"], generator=g)
    xva = torch.as_tensor(Xva)

    model = WindowVAE(2, cfg["hidden_dim"], cfg["latent_dim"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    hist = {"train_total": [], "train_recon": [], "train_kl": [],
            "val_total": [], "val_recon": [], "val_kl": []}
    best, best_state, best_epoch, bad = np.inf, None, 0, 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        sums = np.zeros(3); n = 0
        for (xb,) in loader:
            x_hat, mu, logvar, _ = model(xb)
            loss, rec, kl = vae_loss(xb, x_hat, mu, logvar, beta, recon)
            opt.zero_grad(); loss.backward(); opt.step()
            sums += np.array([loss.item(), rec.item(), kl.item()]) * len(xb)
            n += len(xb)
        for k, v in zip(("total", "recon", "kl"), sums / n):
            hist[f"train_{k}"].append(float(v))

        model.eval()
        with torch.no_grad():
            torch.manual_seed(cfg["seed"] + epoch)   # fixed noise for val comparability
            x_hat, mu, logvar, _ = model(xva)
            vl, vr, vk = vae_loss(xva, x_hat, mu, logvar, beta, recon)
        for k, v in zip(("total", "recon", "kl"), (vl, vr, vk)):
            hist[f"val_{k}"].append(float(v))

        if vl.item() < best - 1e-6:
            best, best_epoch, bad = vl.item(), epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    if not save:
        return model
    t = tag(beta, recon)
    out = ROOT / "outputs"
    torch.save({"state_dict": best_state, "beta": beta, "recon": recon,
                "config": cfg, "best_epoch": best_epoch}, out / f"{t}.pt")
    arrays = {}
    for name in ("train", "val", "test"):
        X, _ = flat(data[name])
        mu, logvar, rec = encode_all(model, X)
        arrays.update({f"{name}_mu": mu, f"{name}_logvar": logvar, f"{name}_recon": rec})
    np.savez(out / f"{t}_latents.npz", **arrays)
    (ROOT / "results" / f"{t}_history.json").write_text(
        json.dumps({"beta": beta, "recon": recon, "best_epoch": best_epoch,
                    "epochs_run": len(hist["val_total"]), **hist}))
    print(f"{t}: best epoch {best_epoch}, epochs run {len(hist['val_total'])}, "
          f"val recon {hist['val_recon'][best_epoch]:.4f}, "
          f"val KL {hist['val_kl'][best_epoch]:.4f}", flush=True)


def main():
    cfg = load_config()
    data = build_dataset(cfg)
    recons = sys.argv[1:] or ["mean"]       # e.g. `python train_vae.py mean sum`
    for recon in recons:
        for beta in cfg["betas"]:
            train_one(cfg, data, beta, recon)


if __name__ == "__main__":
    main()
