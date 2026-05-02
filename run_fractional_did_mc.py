"""Monte Carlo illustration for fractional parallel trends.

The script reproduces the online appendix table for the Economics Letters note.
It uses 5,000 replications and reports the oracle fractional DiD estimator for
K in {20, 30, 40}. The main table uses K=30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fracdiff_weights(d: float, K: int) -> np.ndarray:
    """Weights for (1-B)^d through lag K."""
    pi = np.empty(K + 1, dtype=float)
    pi[0] = 1.0
    for k in range(1, K + 1):
        pi[k] = pi[k - 1] * ((k - 1 - d) / k)
    return pi


def fracint_weights(d: float, L: int) -> np.ndarray:
    """Weights for (1-B)^(-d) through lag L."""
    psi = np.empty(L + 1, dtype=float)
    psi[0] = 1.0
    for k in range(1, L + 1):
        psi[k] = psi[k - 1] * ((k - 1 + d) / k)
    return psi


def summarize(estimates: np.ndarray, true_effect: float = 1.0) -> tuple[float, float, float]:
    """Return mean estimate, bias, and RMSE."""
    mean = float(np.mean(estimates))
    bias = mean - true_effect
    rmse = float(np.sqrt(np.mean((estimates - true_effect) ** 2)))
    return mean, bias, rmse


def grouped_difference(values: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Treated-control mean difference for each replication."""
    C = ~G
    nG = G.sum(axis=1)
    nC = C.sum(axis=1)
    return (values * G).sum(axis=1) / nG - (values * C).sum(axis=1) / nC


def run_mc(
    R: int = 5_000,
    seed: int = 20260502,
    n: int = 300,
    d0: float = 0.45,
    K_values: tuple[int, ...] = (20, 30, 40),
    K_main: int = 30,
    L: int = 150,
    tau: float = 1.0,
    batch_size: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Monte Carlo and return main and robustness results.

    The oracle fractional DiD estimator uses the true memory parameter d0.
    All estimators are evaluated at the first post-treatment date. The code
    simulates only the linear combinations of shocks needed for those estimators.
    """
    rng = np.random.default_rng(seed)
    Kmax = max(K_values)
    psi = fracint_weights(d0, L)
    psi_rev = psi[::-1]
    pi_by_K = {K: fracdiff_weights(d0, K) for K in K_values}
    conv_by_K = {K: np.convolve(pi_by_K[K], psi) for K in K_values}
    lam_window = np.linspace(0.0, 0.2, Kmax + 1)  # dates post-Kmax,...,post

    scenarios = [
        "A. selection on persistent pre-treatment state",
        "B. selection on future untreated innovation",
    ]
    stored: dict[tuple[str, str, int | None], list[np.ndarray]] = {}
    for scenario_label in scenarios:
        stored[(scenario_label, "Levels (d=0)", None)] = []
        stored[(scenario_label, "Standard DiD (d=1)", None)] = []
        for K in K_values:
            stored[(scenario_label, "Oracle fractional DiD (d=d0)", K)] = []

    completed = 0
    while completed < R:
        b = min(batch_size, R - completed)
        # Let the first post-treatment date be T.  eps index L+Kmax is eps_T.
        # We draw eps_{T-(L+Kmax)},...,eps_T so that all required lagged
        # fractionally integrated states can be computed as finite sums.
        eps = rng.normal(size=(b, n, L + Kmax + 1))
        alpha = rng.normal(scale=1.0, size=(b, n))

        u_current = np.tensordot(eps[:, :, Kmax : Kmax + L + 1], psi_rev, axes=([2], [0]))
        u_previous = np.tensordot(eps[:, :, Kmax - 1 : Kmax + L], psi_rev, axes=([2], [0]))

        Y0_current = alpha + lam_window[Kmax] + u_current
        Y0_previous = alpha + lam_window[Kmax - 1] + u_previous

        persistent_state_score = u_previous + 0.25 * rng.normal(size=(b, n))
        future_innovation_score = eps[:, :, L + Kmax] + 0.25 * rng.normal(size=(b, n))
        scores = {
            scenarios[0]: persistent_state_score,
            scenarios[1]: future_innovation_score,
        }

        for scenario_label in scenarios:
            score = scores[scenario_label]
            G = score >= np.median(score, axis=1, keepdims=True)

            Y_current = Y0_current + G * tau
            levels = grouped_difference(Y_current, G)
            did = grouped_difference(Y_current - Y0_previous, G)
            stored[(scenario_label, "Levels (d=0)", None)].append(levels)
            stored[(scenario_label, "Standard DiD (d=1)", None)].append(did)

            for K, pi in pi_by_K.items():
                # Filtered untreated component at T, using estimator truncation K.
                c = conv_by_K[K]
                u_filtered = np.tensordot(
                    eps[:, :, Kmax - K : L + Kmax + 1], c[::-1], axes=([2], [0])
                )
                alpha_filtered = alpha * pi.sum()
                lambda_filtered = float(np.dot(lam_window[Kmax - K : Kmax + 1][::-1], pi))
                # At the first treated date, only the current term is treated and pi_0=1.
                Y_filtered = alpha_filtered + lambda_filtered + u_filtered + G * tau
                frac = grouped_difference(Y_filtered, G)
                stored[(scenario_label, "Oracle fractional DiD (d=d0)", K)].append(frac)

        completed += b

    rows = []
    for (scenario_label, estimator, K), chunks in stored.items():
        est = np.concatenate(chunks)
        mean, bias, rmse = summarize(est, tau)
        rows.append(
            {
                "DGP": scenario_label,
                "Estimator": estimator,
                "K": "" if K is None else K,
                "Mean estimate": mean,
                "Bias": bias,
                "RMSE": rmse,
                "Replications": R,
            }
        )
    all_results = pd.DataFrame(rows)

    order = {
        "Levels (d=0)": 0,
        "Oracle fractional DiD (d=d0)": 1,
        "Standard DiD (d=1)": 2,
    }
    all_results["_order"] = all_results["Estimator"].map(order)
    all_results = all_results.sort_values(["DGP", "_order", "K"]).drop(columns="_order")
    main_results = all_results[(all_results["K"].eq("")) | (all_results["K"].eq(K_main))].copy()
    return main_results.reset_index(drop=True), all_results.reset_index(drop=True)


def combine_summary_tables(tables: list[pd.DataFrame], true_effect: float = 1.0) -> pd.DataFrame:
    """Combine equal-design Monte Carlo summary tables by weighted moments."""
    cat = pd.concat(tables, ignore_index=True)
    rows = []
    for keys, g in cat.groupby(["DGP", "Estimator", "K"], dropna=False, sort=False):
        reps = g["Replications"].to_numpy(dtype=float)
        total_reps = int(reps.sum())
        mean = float(np.sum(g["Mean estimate"].to_numpy(dtype=float) * reps) / total_reps)
        mse = float(np.sum((g["RMSE"].to_numpy(dtype=float) ** 2) * reps) / total_reps)
        K_value = keys[2]
        rows.append(
            {
                "DGP": keys[0],
                "Estimator": keys[1],
                "K": "" if pd.isna(K_value) or K_value == "" else int(K_value),
                "Mean estimate": mean,
                "Bias": mean - true_effect,
                "RMSE": mse ** 0.5,
                "Replications": total_reps,
            }
        )
    out = pd.DataFrame(rows)
    order = {
        "Levels (d=0)": 0,
        "Oracle fractional DiD (d=d0)": 1,
        "Standard DiD (d=1)": 2,
    }
    out["_order"] = out["Estimator"].map(order)
    return out.sort_values(["DGP", "_order", "K"]).drop(columns="_order").reset_index(drop=True)


def run_mc_chunked(
    total_R: int = 5_000,
    chunk_R: int = 1_000,
    seed: int = 20260502,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 5,000-replication design in chunks and combine summaries.

    Chunking keeps memory use modest and makes the script robust on ordinary laptops.
    """
    if total_R % chunk_R != 0:
        raise ValueError("total_R must be an integer multiple of chunk_R")
    robust_tables = []
    for m in range(total_R // chunk_R):
        _, robust = run_mc(R=chunk_R, seed=seed + m, **kwargs)
        robust_tables.append(robust)
        print(f"completed chunk {m + 1}/{total_R // chunk_R}", flush=True)
    robustness = combine_summary_tables(robust_tables)
    K_main = kwargs.get("K_main", 30)
    main = robustness[(robustness["K"].eq("")) | (robustness["K"].eq(K_main))].reset_index(drop=True)
    return main, robustness


if __name__ == "__main__":
    main, robustness = run_mc_chunked(total_R=5_000, chunk_R=1_000)
    pd.set_option("display.width", 140)
    print("Main table (K=30 for oracle fractional DiD):")
    print(main.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nAll results including K robustness:")
    print(robustness.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    main.to_csv("fractional_did_mc_results.csv", index=False)
    robustness.to_csv("fractional_did_mc_results_with_K_robustness.csv", index=False)
