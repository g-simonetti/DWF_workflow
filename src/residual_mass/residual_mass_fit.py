#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")

# -----------------------------------------------------------------------------
# Ensure src/ is importable so we can import autocorr_time.tau_int
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from autocorr_time.tau_int import compute_tau_from_file
except Exception as e:
    raise ImportError(
        "Failed to import compute_tau_from_file from src/autocorr_time/tau_int.py.\n"
        "Expected repo layout with 'src/autocorr_time/tau_int.py'."
    ) from e


# ============================================================
# File Readers
# ============================================================
def read_mres_file(filename: str) -> np.ndarray:
    """Read real part of wardIdentity/PJ5q from an mres HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f["wardIdentity/PJ5q"][:]["re"]


def read_ptll_file(filename: str, n_elems: int | None = None) -> np.ndarray:
    """Read real part of meson/meson_1/corr from a pt_ll HDF5 file."""
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        if n_elems is None or len(data) == n_elems:
            return data["re"]
    raise ValueError(f"corr dataset does not have {n_elems} entries in {filename}")


# ============================================================
# Folding
# ============================================================
def fold_correlator(data: np.ndarray) -> np.ndarray:
    r"""
    Fold correlator according to

        C_fold(t) = 1/2 * [ C(t) + C(Nt - t) ]

    using periodic indexing, so Nt - t is interpreted mod Nt.
    """
    arr = np.asarray(data)
    T = arr.shape[-1]
    idx = (-np.arange(T)) % T
    return 0.5 * (arr + arr[..., idx])


# ============================================================
# Bootstrap ratio of ensemble means
# ============================================================
def bootstrap_ratio_of_means(
    data_num: np.ndarray,
    data_den: np.ndarray,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
):
    r"""
    Compute the ratio of ensemble means of folded correlators:

        Rbar(t) = <num_fold(t)> / <den_fold(t)>

    and estimate its bootstrap uncertainty.
    """
    if rng is None:
        rng = np.random.default_rng()

    num = fold_correlator(data_num)
    den = fold_correlator(data_den)

    if num.shape != den.shape:
        raise ValueError(f"Shape mismatch: numerator {num.shape}, denominator {den.shape}")

    if num.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (Ncfg, T), got {num.shape}")

    Ncfg, T = num.shape
    if Ncfg < 2:
        raise ValueError(f"Need at least 2 configurations, got {Ncfg}")

    num_mean = num.mean(axis=0)
    den_mean = den.mean(axis=0)

    if np.any(den_mean == 0):
        bad = np.where(den_mean == 0)[0]
        raise ZeroDivisionError(f"Zero ensemble-mean denominator encountered at times {bad.tolist()}")

    ratio_mean = num_mean / den_mean

    ratios_boot = np.empty((n_boot, T), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, Ncfg, size=Ncfg)
        num_b = num[idx].mean(axis=0)
        den_b = den[idx].mean(axis=0)

        if np.any(den_b == 0):
            bad = np.where(den_b == 0)[0]
            raise ZeroDivisionError(
                f"Zero bootstrap denominator encountered in replica {b} at times {bad.tolist()}"
            )

        ratios_boot[b] = num_b / den_b

    ratio_err = ratios_boot.std(axis=0, ddof=1)
    return ratio_mean, ratio_err, ratios_boot


# ============================================================
# Correlated constant fit from bootstrap covariance
# ============================================================
def correlated_constant_fit(
    ratio_mean: np.ndarray,
    ratios_boot: np.ndarray,
    t_vals: np.ndarray,
    tmin: int,
    tmax: int,
    rcond: float = 1e-12,
):
    mask = (t_vals >= tmin) & (t_vals <= tmax)
    if not np.any(mask):
        raise ValueError(f"No points in plateau range [{tmin}, {tmax}]")

    y = np.asarray(ratio_mean[mask], dtype=np.float64)
    Yb = np.asarray(ratios_boot[:, mask], dtype=np.float64)

    if Yb.ndim != 2:
        raise ValueError(f"Expected bootstrap array of shape (Nb, np), got {Yb.shape}")

    Nb, np_ = Yb.shape
    if Nb < 2:
        raise ValueError("Need at least 2 bootstrap replicas to estimate covariance.")
    if np_ < 1:
        raise ValueError("Plateau window is empty.")

    y_boot_mean = Yb.mean(axis=0)

    if np_ == 1:
        m = float(y[0])
        sigma_m = float(np.sqrt(np.cov(Yb[:, 0], ddof=1)))
        red_chi2 = None
        cov = np.array([[sigma_m**2]], dtype=np.float64)
        return m, sigma_m, red_chi2, cov, y, y_boot_mean

    cov = np.cov(Yb, rowvar=False, ddof=1)
    cov_inv = np.linalg.pinv(cov, rcond=rcond)

    one = np.ones(np_, dtype=np.float64)
    denom = float(one @ cov_inv @ one)
    if denom <= 0:
        raise np.linalg.LinAlgError(
            "Non-positive denominator in correlated fit. Covariance may be singular or ill-conditioned."
        )

    m = float((one @ cov_inv @ y) / denom)
    sigma_m = float(np.sqrt(1.0 / denom))

    resid = y - m * one
    chi2 = float(resid @ cov_inv @ resid)
    dof = np_ - 1
    red_chi2 = chi2 / dof if dof > 0 else None

    return m, sigma_m, red_chi2, cov, y, y_boot_mean


# ============================================================
# Trajectory numbers: ordering + selection
# ============================================================
_TRAJ_RE = re.compile(r".*\.(\d+)\.h5$")


def traj_number_from_path(path: str) -> int:
    base = os.path.basename(path)
    m = _TRAJ_RE.match(base)
    if not m:
        raise ValueError(f"Cannot extract trajectory number from filename: {path}")
    return int(m.group(1))


def build_number_map(files: list[str]) -> dict[int, str]:
    """Map trajectory-number -> filepath. If duplicates exist, keep the first in sorted order."""
    out: dict[int, str] = {}
    for f in sorted(files):
        n = traj_number_from_path(f)
        if n not in out:
            out[n] = f
    return out


def select_numbers_by_delta(numbers_sorted: list[int], delta_traj_ps: int) -> list[int]:
    """
    numbers_sorted must already be therm-cut and sorted.
    Keep arithmetic progression start=numbers_sorted[0], then start + k*delta_traj_ps (if present).

    If delta_traj_ps <= 0, return full list.
    """
    if not numbers_sorted:
        return []

    delta_traj_ps = int(delta_traj_ps)
    if delta_traj_ps <= 0:
        return list(numbers_sorted)

    start = numbers_sorted[0]
    num_set = set(numbers_sorted)
    last = numbers_sorted[-1]

    keep: list[int] = []
    k = 0
    while True:
        n = start + k * delta_traj_ps
        if n in num_set:
            keep.append(n)
        if n > last:
            break
        k += 1

    return sorted(set(keep))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute residual mass from HDF5 data using an optional correlated constant fit."
    )
    parser.add_argument("input_dir", help="mesons directory containing mres.*.h5 and pt_ll.*.h5")

    parser.add_argument("--label", default="", help="yes → include β, am0 label on plot")
    parser.add_argument("--mres_out", required=True, help="Output JSON file (m_res.json)")
    parser.add_argument("--plot_file", required=True, help="Output plot file")
    parser.add_argument("--plot_styles", default="")

    # plateau args may be passed as -1 by the workflow when missing
    parser.add_argument("--plateau_start", type=float, default=-1)
    parser.add_argument("--plateau_end", type=float, default=-1)

    parser.add_argument("--therm", type=int, required=True)
    parser.add_argument("--delta_traj_ps", type=int, required=True)

    parser.add_argument("--beta", type=float, default=np.nan)
    parser.add_argument("--mass", type=float, default=np.nan)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--a5", type=float, required=True)
    parser.add_argument("--m5", type=float, required=True)
    parser.add_argument("--mpv", type=float, required=True)

    parser.add_argument("--Nt", type=int, required=True)
    parser.add_argument("--Ns", type=int, required=True)
    parser.add_argument("--Ls", type=int, required=True)

    parser.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap replicas")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")

    args = parser.parse_args()

    # Treat negative plateau bounds as "missing"
    have_plateau = (args.plateau_start is not None) and (args.plateau_end is not None)
    if have_plateau:
        have_plateau = (args.plateau_start >= 0) and (args.plateau_end >= 0)

    plateau_start = int(round(args.plateau_start)) if have_plateau else None
    plateau_end = int(round(args.plateau_end)) if have_plateau else None

    if have_plateau and plateau_end < plateau_start:
        raise ValueError(
            f"Invalid plateau range: plateau_start={plateau_start}, plateau_end={plateau_end}"
        )

    rng = np.random.default_rng(args.seed)

    # ----------------------------------------------------------
    # Find files
    # ----------------------------------------------------------
    mres_files = glob.glob(os.path.join(args.input_dir, "mres.*.h5"))
    ptll_files = glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5"))
    if not mres_files or not ptll_files:
        raise FileNotFoundError(f"Missing mres.*.h5 or pt_ll.*.h5 files in {args.input_dir}")

    mres_map = build_number_map(mres_files)
    ptll_map = build_number_map(ptll_files)

    common_numbers = sorted(set(mres_map.keys()) & set(ptll_map.keys()))
    if not common_numbers:
        raise FileNotFoundError("No matching trajectory numbers between mres.*.h5 and pt_ll.*.h5")

    # ----------------------------------------------------------
    # FULL series: apply ONLY therm cut, keep strict ordering
    # ----------------------------------------------------------
    therm = int(args.therm)
    full_numbers = [n for n in common_numbers if n > therm]
    if len(full_numbers) < 2:
        raise ValueError(
            f"Too few matched configurations after therm cut. "
            f"common={len(common_numbers)}, full(after therm)={len(full_numbers)}, therm={therm}"
        )

    mres_full_files = [mres_map[n] for n in full_numbers]
    ptll_full_files = [ptll_map[n] for n in full_numbers]

    # ----------------------------------------------------------
    # MEAS series (outputs): apply delta_traj_ps AFTER therm cut
    # ----------------------------------------------------------
    meas_numbers = select_numbers_by_delta(full_numbers, delta_traj_ps=int(args.delta_traj_ps))
    if len(meas_numbers) < 2:
        raise ValueError(
            f"Too few configs after measurement thinning. "
            f"full(after therm)={len(full_numbers)}, meas(after delta)={len(meas_numbers)}, "
            f"delta_traj_ps={args.delta_traj_ps}"
        )

    mres_meas_files = [mres_map[n] for n in meas_numbers]
    ptll_meas_files = [ptll_map[n] for n in meas_numbers]

    # ----------------------------------------------------------
    # Read MEAS data in NUMBER order
    # ----------------------------------------------------------
    mres_meas = np.array([read_mres_file(f) for f in mres_meas_files])
    n_times = mres_meas.shape[1]
    ptll_meas = np.array([read_ptll_file(f, n_elems=n_times) for f in ptll_meas_files])

    min_len_meas = min(len(mres_meas), len(ptll_meas))
    mres_meas = mres_meas[:min_len_meas]
    ptll_meas = ptll_meas[:min_len_meas]
    used_numbers_meas = meas_numbers[:min_len_meas]

    if min_len_meas < 2:
        raise ValueError(f"Need at least 2 MEAS configurations after reading, got {min_len_meas}")

    # ----------------------------------------------------------
    # Read FULL pt_ll series (for tau_int), NO delta thinning, in NUMBER order
    # ----------------------------------------------------------
    ptll_full = np.array([read_ptll_file(f, n_elems=n_times) for f in ptll_full_files])
    used_numbers_full = full_numbers[: len(ptll_full)]

    if len(used_numbers_full) < 2:
        raise ValueError(
            f"Need at least 2 FULL configurations for tau_int after reading, got {len(used_numbers_full)}"
        )

    # ----------------------------------------------------------
    # Bootstrap ratio-of-means using folded correlators (MEAS ensemble)
    # ----------------------------------------------------------
    ratio_mean, ratio_err, ratios_boot = bootstrap_ratio_of_means(
        mres_meas,
        ptll_meas,
        n_boot=int(args.n_boot),
        rng=rng,
    )

    # ----------------------------------------------------------
    # Optional correlated constant fit on plateau window
    # ----------------------------------------------------------
    t_vals = np.arange(n_times)

    avg = None
    err = None
    red_chi2 = None
    cov_plateau = None
    y_plateau = None
    y_boot_mean = None

    if have_plateau:
        avg, err, red_chi2, cov_plateau, y_plateau, y_boot_mean = correlated_constant_fit(
            ratio_mean=ratio_mean,
            ratios_boot=ratios_boot,
            t_vals=t_vals,
            tmin=plateau_start,
            tmax=plateau_end,
        )

    # ----------------------------------------------------------
    # tau_int for UNFOLDED pt_ll:
    #   - plateau_start if plateau exists
    #   - otherwise T/2
    #
    # Always store in tau_int_ptll and always use ptll_tau_int in JSON
    # ----------------------------------------------------------
    tau_t = int(plateau_start) if have_plateau else int(n_times // 2)

    if tau_t < 0 or tau_t >= n_times:
        raise ValueError(f"tau_int time slice t={tau_t} out of range [0, {n_times - 1}]")

    out_json_dir = os.path.dirname(os.path.abspath(args.mres_out)) or "."
    tau_out_dir = os.path.join(out_json_dir, "tau_int_ptll")
    os.makedirs(tau_out_dir, exist_ok=True)

    tau_series_path = os.path.join(tau_out_dir, "ptll_series.txt")
    with open(tau_series_path, "w") as f:
        f.write(f"# traj_number\tpt_ll_re(t={tau_t})\n")
        for n, y in zip(used_numbers_full, ptll_full[:, tau_t]):
            f.write(f"{n}\t{float(y):.16e}\n")

    ptll_tau, ptll_tau_err, Nb_est, Nbs_est, found = compute_tau_from_file(
        input_file=tau_series_path,
        out_dir=tau_out_dir,
        therm=0,
        plot_styles=args.plot_styles if args.plot_styles else None,
        base_name="tau_int",
    )

    # ----------------------------------------------------------
    # Write JSON output
    # ----------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.mres_out)), exist_ok=True)

    payload = {
        "parameters": {
            "beta": float(args.beta) if np.isfinite(args.beta) else None,
            "mass": float(args.mass) if np.isfinite(args.mass) else None,
            "Nt": int(args.Nt),
            "Ns": int(args.Ns),
            "Ls": int(args.Ls),
            "alpha": float(args.alpha),
            "a5": float(args.a5),
            "m5": float(args.m5),
            "mpv": float(args.mpv),
        },
        "analysis_settings": {
            "plateau_start": int(plateau_start) if have_plateau else None,
            "plateau_end": int(plateau_end) if have_plateau else None,
            "therm": int(therm),
            "delta_traj_ps": int(args.delta_traj_ps),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed) if args.seed is not None else None,
            "ratio_definition": "ratio_of_folded_ensemble_means",
            "covariance_estimator": "bootstrap",
            "fit_method": "correlated_constant_fit" if have_plateau else None,
            "fit_error_estimator": "GLS_analytic" if have_plateau else None,
            "tau_int_series": "unfolded_ptll",
            "fit_performed": bool(have_plateau),
        },
        "ensembles": {
            "meas": {
                "n_cfg": int(min_len_meas),
                "traj_start": int(used_numbers_meas[0]),
                "traj_end": int(used_numbers_meas[-1]),
                "traj_numbers": [int(x) for x in used_numbers_meas],
            },
            "full": {
                "n_cfg": int(len(used_numbers_full)),
                "traj_start": int(used_numbers_full[0]),
                "traj_end": int(used_numbers_full[-1]),
                "traj_numbers": [int(x) for x in used_numbers_full],
            },
        },
        "mres_series": {
            "t": [int(t) for t in t_vals],
            "mres": [float(x) for x in ratio_mean],
            "mres_err": [float(x) for x in ratio_err],
            "folded": True,
        },
        "mres_extract": {
            "ptll_tau_int": {
                "t": int(tau_t),
                "tau_int": float(ptll_tau),
                "tau_int_err": float(ptll_tau_err),
                "Nb_est": int(Nb_est),
                "Nbs_est": int(Nbs_est),
                "found": bool(found),
                "tau_int_dir": str(tau_out_dir),
                "series_file": str(tau_series_path),
                "folded": False,
            },
        },
    }

    if have_plateau:
        plateau_mask = (t_vals >= plateau_start) & (t_vals <= plateau_end)
        payload["mres_extract"].update(
            {
                "value": float(avg),
                "error": float(err),
                "reduced_chi2": float(red_chi2) if red_chi2 is not None else None,
                "plateau_start": int(plateau_start),
                "plateau_end": int(plateau_end),
                "n_plateau_points": int(np.sum(plateau_mask)),
                "plateau_t": [int(x) for x in t_vals[plateau_mask]],
                "plateau_y": [float(x) for x in y_plateau],
                "plateau_y_boot_mean": [float(x) for x in y_boot_mean],
                "covariance_matrix": cov_plateau.tolist(),
            }
        )

    with open(args.mres_out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # ----------------------------------------------------------
    # Plot (MEAS ensemble)
    # ----------------------------------------------------------
    if args.plot_styles:
        parts = [p.strip() for p in str(args.plot_styles).split(",") if p.strip()]
        if parts:
            plt.style.use(parts)

    t_plot_max = n_times // 2
    plot_mask = t_vals <= t_plot_max
    t_plot = t_vals[plot_mask]
    ratio_mean_plot = ratio_mean[plot_mask]
    ratio_err_plot = ratio_err[plot_mask]

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = rf"$\beta={args.beta},\ am_0={args.mass}$" if args.label == "yes" else None
    title_str = (
        rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
        rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
    )
    ax.set_title(title_str, fontsize=10)

    ax.errorbar(
        t_plot,
        ratio_mean_plot,
        yerr=ratio_err_plot,
        fmt="o",
        color="C4",
        label=data_label,
    )

    if have_plateau:
        fit_label = rf"$am_{{\rm res}}^{{\rm extract}} = {avg:.5f}\,\pm\,{err:.5f}$"
        ax.axvspan(plateau_start, plateau_end, color="C2", alpha=0.2, label="Plateau range")
        ax.fill_between(
            [plateau_start, plateau_end],
            [avg - err, avg - err],
            [avg + err, avg + err],
            color="C1",
            alpha=0.25,
            linewidth=0,
        )
        ax.hlines(avg, plateau_start, plateau_end, color="C1", linestyle="--", label=fit_label)

    ax.set_xlim(-0.5, t_plot_max + 0.5)
    ax.set_xlabel("$t/a$")
    ax.set_ylabel("$am_{\\rm res}$")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if have_plateau or data_label:
        ax.legend()

    os.makedirs(os.path.dirname(os.path.abspath(args.plot_file)), exist_ok=True)
    plt.savefig(args.plot_file, dpi=300)
    plt.close(fig)

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------
    print(
        f"✓ FULL series (tau_int input): n_cfg={len(used_numbers_full)} "
        f"(numbers {used_numbers_full[0]} → {used_numbers_full[-1]}, therm>{therm})"
    )
    print(
        f"✓ MEAS series (outputs): n_cfg={min_len_meas} "
        f"(numbers {used_numbers_meas[0]} → {used_numbers_meas[-1]}, "
        f"delta_traj_ps={args.delta_traj_ps})"
    )

    if have_plateau:
        print(
            f"✓ Correlated plateau fit: am_res^extract = {avg:.6g} ± {err:.3g}"
            + (f", chi2/dof = {red_chi2:.3g}" if red_chi2 is not None else "")
        )
    else:
        print("✓ No valid plateau_start/plateau_end provided: skipped correlated plateau fit")

    print(f"✓ Saved plot → {args.plot_file}")
    print(f"✓ Saved JSON → {args.mres_out}")
    print(f"✓ tau_int outputs written in → {tau_out_dir}")
    print(
        f"✓ pt_ll tau_int at t={tau_t} (FULL unfolded series; Berg/2): "
        f"{ptll_tau:.6g} ± {ptll_tau_err:.3g}  "
        f"(Nb={Nb_est}, Nbs={Nbs_est}, found={bool(found)})"
    )


if __name__ == "__main__":
    main()