# VAE recovery of latent cognitive states: report

## Abstract

I simulated 200 subjects x 300 trials from a three-state (engaged, exploratory, lapsed) first-order Markov process emitting binary correctness and log-RT, windowed the data into non-overlapping 10-trial windows, and split by subject (70/15/15). A GRU-VAE (4-d latent; beta 0.1, 1, 5; two reconstruction scalings) was compared with two HMM baselines on held-out subjects. K-means on the VAE latents recovered the window-level states (test ARI 0.30, NMI 0.22, macro-F1 0.51 for the validation-selected beta = 0.1 model; 0.30 +/- 0.01 across five model seeds) better than the specified Gaussian HMM (ARI 0.16) but clearly worse than a correctly specified Bernoulli-Gaussian HMM (ARI 0.47). The brief's loss collapsed completely at beta >= 1. Setting the lapsed-associated latent dimension (z1) to the lapsed-window mean, versus the engaged mean, produced decoded behaviour in the correct direction (longer RT, lower correctness) but only ~35-42% of the simulator's true shift, with strongly shrunken RT variability. These are model-based counterfactuals, not causal inference.

## Results

| Model | Test ARI | NMI | Macro-F1 | Window transition MAE |
|---|---:|---:|---:|---:|
| Shuffled-label control | 0.001 | 0.002 | 0.34 | - |
| HMM, Gaussian (as specified) | 0.158 | 0.105 | 0.372 | 0.247 |
| HMM, Bernoulli-Gaussian | 0.471 | 0.357 | 0.749 | 0.041 |
| VAE, spec loss, beta = 0.1 (primary) | 0.301 | 0.217 | 0.512 | 0.153 |
| VAE, spec loss, beta = 1 / 5 | 0.268 / 0.053 | 0.254 / 0.053 | 0.645 / 0.418 | 0.068 / 0.107 |
| VAE, summed loss, beta = 0.1 / 1 / 5 | 0.127 / 0.249 / 0.316 | 0.123 / 0.174 / 0.285 | 0.439 / 0.526 / 0.656 | 0.114 / 0.114 / 0.077 |

Full table: `results/summary.csv`; figures in `figures/`.

**Success criteria**
- *Primary (VAE ARI > HMM ARI):* met against the specified Gaussian HMM (0.30 vs 0.16), **not met** against the Bernoulli-Gaussian HMM (0.30 vs 0.47). The Gaussian HMM's weakness is an artefact of modelling binary correctness as Gaussian, so the honest summary is that a discrete-state model with the right emission distributions beats the VAE on this simulator. Per the brief, that is a valid finding: the simulator is a discrete-state process, so a discrete-state model is the more appropriate one.
- *Secondary (a latent dimension separates lapsed from engaged):* met. In the primary model z1 has a standardised lapsed-engaged difference of 3.2 SD (single-dimension probe accuracy 0.72 vs 0.57 majority baseline).
- *Counterfactual direction:* met. Intervening on z1 toward the lapsed mean versus the engaged mean shifts decoded mean log-RT by +0.30 (simulator: +0.70) and decoded correctness by -0.12 (simulator: -0.35). Wasserstein distance of decoded to true lapsed RTs is 0.39 (engaged: 0.19). The magnitude is under-recovered and decoded within-window RT SD (0.02-0.08) is far below the simulator's (0.24-0.44), because a deterministic MSE decoder outputs conditional means.

**Posterior collapse.** With the brief's loss, beta = 1 and 5 collapse (total KL < 1e-4 nats, posterior variance ~ 1). beta = 0.1 keeps three active dimensions (KL 0.41, 0.06, 0.09 nats) and one collapsed. Summed-reconstruction runs keep more dimensions active at low beta. Near-collapsed latents can still be linearly decodable (z-scored probes), so cluster ARI on tiny-variance latents should not be read as healthy latent use.

## Interpretation and caveats

The VAE learns state-related structure (probe accuracy 0.73 vs 0.57 baseline, ARI far above the shuffle control), but its unsupervised clusters are less state-aligned than a properly specified HMM. Window labels are majority votes over a Markov chain with 70-90% self-transition, so mixed windows cap every method. Results rest on one simulated dataset (model-seed variation only). Counterfactual claims are about the decoder's behaviour, not about causal structure in cognition.

## Comprehension checks (brief answers)

1. The HMM shows whether the VAE beats a simple, correctly structured probabilistic model; without it, any ARI is uninterpretable.
2. Posterior collapse: the posterior equals the prior in some/all dimensions (variance ~ 1, KL ~ 0, no state information); detected here via per-dimension variance, KL and single-dimension probes.
3. The intervention edits a learned representation and reads out the decoder; it does not manipulate the generative process that produced real behaviour.
4. Worse-than-HMM recovery suggests a continuous, unconstrained latent is an inefficient fit to a discrete-switching process without structure or priors that encourage clustering.
5. Trials within a subject are autocorrelated and subject-specific; trial-level splits leak.
6. Latent/cluster labels are arbitrary permutations; matching is required before label-based scores.

## CV line

Built a reproducible simulation pipeline and GRU-VAE/HMM comparison (PyTorch, hmmlearn) recovering latent cognitive states from choice/RT sequences, with posterior-collapse diagnostics and latent-intervention counterfactuals benchmarked against simulator ground truth.
