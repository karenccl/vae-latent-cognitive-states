# VAE recovery of latent cognitive states and counterfactual simulation

Simulation study: can a windowed GRU-VAE trained on choice/RT sequences recover
known hidden states (engaged, exploratory, lapsed) better than an HMM baseline,
and do interventions on its latent space decode to behaviour consistent with the
simulator's own state-specific distributions? See [report.md](report.md) for
results and interpretation.

> **Counterfactual is not causal.** Latent interventions answer "what would the
> *model* decode under a different latent value?". They are compared with the
> simulator's clamped-state ground truth, but they do not identify causal
> mechanisms in real cognition. Nothing here is real-data validation.

## Installation

```bash
conda env create -f environment.yml
conda activate vae-cognitive-states
```

All parameters live in `config.yaml` (seed 2026, 200 subjects x 300 trials, windows of 10, latent_dim 4, betas 0.1/1/5).

## Run order

```bash
python -m pytest                      # simulator + windowing + split tests
cd src
python simulate.py                    # figure 1 (example subject)
python hmm_baseline.py                # both HMM baselines -> results/hmm_results.json
python train_vae.py mean sum          # 3 betas x 2 reconstruction scalings -> outputs/, results/*_history.json
python evaluate.py                    # summary.csv, collapse_diagnostics.csv, figures 2-4, S1
python counterfactuals.py             # counterfactual tables + figure 5
python seed_robustness.py             # optional: primary config under 4 other model seeds
```

## Outputs

| File | Content |
|---|---|
| `results/summary.csv` | HMMs and all 6 VAE runs: val/test ARI, NMI, macro-F1, transition error, probe, collapse indicators |
| `results/collapse_diagnostics.csv` | per-latent-dimension posterior variance, KL, single-dimension probe |
| `results/hmm_results.json`, `primary_model.json`, `seed_robustness.csv` | baseline details, model selection, seed check |
| `results/counterfactual_*.{csv,json}` | decoded vs simulator behaviour, Wasserstein distances, intervention direction |
| `outputs/` | checkpoints (`.pt`), latents/reconstructions (`.npz`), HMM test predictions |
| `figures/` | fig1-5 plus `figS1_training_curves.png` |

## Methodological choices that deviate from or extend the brief

1. **Two HMMs.** The brief's `GaussianHMM` (binary correctness modelled as Gaussian) degenerates: its states latch onto correct/incorrect and outlier RTs rather than cognitive states. It is reported as specified, and a Bernoulli-Gaussian HMM (own Baum-Welch in `hmm_baseline.py`, correct emission model) is added as the fair baseline. Both are fit on trial sequences per subject; trial states are majority-voted into window labels.
2. **Transition error** is the MAE between window-level transition matrices of matched predicted labels and of true window labels (same definition for HMM and VAE). Trial-level MAE versus the simulator's A is reported additionally for the HMMs.
3. **Reconstruction scaling.** The brief's loss (MSE averaged over elements + beta x KL summed over latents) makes KL dominate; beta >= 1 collapses completely. A sensitivity variant with summed squared error (`recon=sum`) is also run.
4. **Primary VAE** is selected on *validation* K-means ARI among runs with at least one active latent dimension (KL >= 0.01 nats). Test data are used for final evaluation only.
5. **Probes are z-scored.** K-means is scale-invariant and a collapsed latent can still carry a tiny-scale signal, so cluster ARI alone can look better than the KL suggests; see limitations.
6. Counterfactual correctness is decoded as a probability and sampled as Bernoulli for sequence statistics; state-mean latent values come from training windows only.

## Known limitations

- One simulated dataset; model-seed variation is checked (primary test ARI 0.30 +/- 0.01 across five seeds) but dataset-seed variation is not.
- The deterministic MSE decoder shrinks toward the mean: decoded RT variability is far below the simulator's, and counterfactual shifts reach ~35-42% of the true shift.
- "Posterior collapse" is a continuum: at KL ~ 1e-4 nats latents can still be linearly decodable (see `collapse_diagnostics.csv`).
- Window labels are majority votes, so mixed windows are inherently ambiguous; this caps all recovery metrics.
- Bootstrap CIs in the counterfactual file are over test windows of a deterministic decoder and understate uncertainty.
