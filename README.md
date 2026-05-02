# Replication code for "Fractional parallel trends"

This repository contains the Monte Carlo code and simulation output for the
online appendix of:

> Zambrano, E. "Fractional parallel trends." *Economics Letters* (forthcoming).

The paper studies a fractional-difference relaxation of the parallel-trends
restriction in difference-in-differences. The simulation is an illustration of
the maintained identifying restriction, not evidence of general robustness.

## Contents

| File | Description |
| --- | --- |
| `run_fractional_did_mc.py` | Reproduces the appendix table. |
| `fractional_did_mc_results.csv` | Main table (oracle fractional DiD at K = 30). |
| `fractional_did_mc_results_with_K_robustness.csv` | Same design, K ∈ {20, 30, 40}. |

## Reproducing the table

Requirements: Python ≥ 3.9, NumPy, pandas. From this directory:

```
python run_fractional_did_mc.py
```

The script runs 5,000 replications in chunks of 1,000 and writes both CSV
files to the working directory. The seed is fixed at `20260502`; output is
deterministic up to NumPy's `default_rng` implementation.

## Design summary

- n = 300 units, T₀ = 80 pre-treatment periods, one post-treatment date.
- Untreated outcome: `Y_it(0) = α_i + λ_t + u_it`, where
  `(1 − B)^{d₀} u_it = ε_it` with `d₀ = 0.45`.
- Estimators: levels (d = 0), oracle fractional DiD (d = d₀), standard
  first-difference DiD (d = 1).
- Two assignment mechanisms: selection on the persistent pre-treatment state
  (DGP A) and on the future untreated innovation (DGP B).

See the paper's online appendix for the full design and interpretation.

## License

MIT — see `LICENSE`.

## Citation

```
@article{zambrano_fractional_parallel_trends,
  author  = {Zambrano, Eduardo},
  title   = {Fractional parallel trends},
  journal = {Economics Letters},
  year    = {forthcoming}
}
```
