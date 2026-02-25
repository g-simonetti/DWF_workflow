#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import glob
import os, sys
import re
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

plt.style.use("tableau-colorblind10")

# -----------------------------------------------------------------------------
# Ensure src/ is importable so we can import autocorr_time.tau_int
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]  # if this file is src/residual_mass/residual_mass_plateau.py
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
# Bootstrap ratio
# ============================================================
def bootstrap_ratio(data1: np.ndarray, data2: np.ndarray, n_boot: int = 2000):
    ratios_cfg = data1 / data2
    Ncfg, T = ratios_cfg.shape
    ratios_boot = np.empty((n_boot, T))

    for b in range(n_boot):
        idx = np.random.randint(0, Ncfg, size=Ncfg)
        ratios_boot[b] = ratios_cfg[idx].mean(axis=0)

    ratio_mean = ratios_cfg.mean(axis=0)
    ratio_err = ratios_boot.std(axis=0, ddof=1)
    return ratio_mean, ratio_err, ratios_cfg


# ============================================================
# Bootstrap plateau fit
# ============================================================
def bootstrap_plateau(ratios_cfg, t_vals, tmin, tmax, n_boot=2000):
    """
    Bootstrap over configurations to get plateau value and error.
    No covariance matrix needed.
    """
    mask = (t_vals >= tmin) & (t_vals <= tmax)
    if not np.any(mask):
        raise ValueError(f"No points in plateau range {tmin}–{tmax}")

    Ncfg, T = ratios_cfg.shape
    A = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, Ncfg, size=Ncfg)
        Rb = ratios_cfg[idx].mean(axis=0)
        A[b] = Rb[mask].mean()

    return A.mean(), A.std(ddof=1)


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
    parser = argparse.ArgumentParser(description="Compute residual mass from HDF5 data (Snakemake rule m_res).")
    parser.add_argument("input_dir", help="mesons directory containing mres.*.h5 and pt_ll.*.h5")

    # snakemake-expected args
    parser.add_argument("--label", default="", help="yes → include β, am0 label on plot")
    parser.add_argument("--mres_out", required=True, help="Output JSON file (m_res.json)")
    parser.add_argument("--plot_file", required=True, help="Output plot file")
    parser.add_argument("--plot_styles", default="")

    parser.add_argument("--plateau_start", type=float, required=True)
    parser.add_argument("--plateau_end", type=float, required=True)
    parser.add_argument("--therm", type=int, required=True)
    parser.add_argument("--delta_traj_ps", type=int, required=True)

    parser.add_argument("--beta", type=float, default=np.nan)
    parser.add_argument("--mass", type=float, default=np.nan)

    # title args
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--a5", type=float, required=True)
    parser.add_argument("--m5", type=float, required=True)
    parser.add_argument("--mpv", type=float, required=True)

    args = parser.parse_args()

    plateau_start = int(round(args.plateau_start))
    plateau_end = int(round(args.plateau_end))

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
    # Read MEAS data (ratio/plateau/plot) in NUMBER order
    # ----------------------------------------------------------
    mres_meas = np.array([read_mres_file(f) for f in mres_meas_files])
    n_times = mres_meas.shape[1]
    ptll_meas = np.array([read_ptll_file(f, n_elems=n_times) for f in ptll_meas_files])

    min_len_meas = min(len(mres_meas), len(ptll_meas))
    mres_meas = mres_meas[:min_len_meas]
    ptll_meas = ptll_meas[:min_len_meas]
    used_numbers_meas = meas_numbers[:min_len_meas]

    # ----------------------------------------------------------
    # Read FULL pt_ll series (for tau_int), NO delta thinning, in NUMBER order
    # ----------------------------------------------------------
    ptll_full = np.array([read_ptll_file(f, n_elems=n_times) for f in ptll_full_files])
    used_numbers_full = full_numbers[: len(ptll_full)]

    # ----------------------------------------------------------
    # Bootstrap Ratio (MEAS ensemble)
    # ----------------------------------------------------------
    ratio_mean, ratio_err, ratios_cfg = bootstrap_ratio(mres_meas, ptll_meas)

    # ----------------------------------------------------------
    # Plateau Fit via Bootstrap (MEAS ensemble)
    # ----------------------------------------------------------
    t_vals = np.arange(n_times)
    avg, err = bootstrap_plateau(ratios_cfg, t_vals, plateau_start, plateau_end)

    # ----------------------------------------------------------
    # tau_int for pt_ll at t=plateau_start using external module (FULL ensemble)
    # ----------------------------------------------------------
    if plateau_start < 0 or plateau_start >= n_times:
        raise ValueError(f"plateau_start={plateau_start} out of range [0, {n_times-1}]")

    out_json_dir = os.path.dirname(os.path.abspath(args.mres_out)) or "."
    tau_out_dir = os.path.join(out_json_dir, "tau_int_ptll_tstart")
    os.makedirs(tau_out_dir, exist_ok=True)

    tau_series_path = os.path.join(tau_out_dir, "ptll_series_tstart.txt")
    with open(tau_series_path, "w") as f:
        f.write("# traj_number\tpt_ll_re(t=plateau_start)\n")
        for n, y in zip(used_numbers_full, ptll_full[:, plateau_start]):
            f.write(f"{n}\t{float(y):.16e}\n")

    ptll_tau, ptll_tau_err, Nb_est, Nbs_est, found = compute_tau_from_file(
        input_file=tau_series_path,
        out_dir=tau_out_dir,
        therm=0,  # already therm-cut by construction
        plot_styles=args.plot_styles if args.plot_styles else None,
        base_name="tau_int",
    )

    # ----------------------------------------------------------
    # Write JSON output (single artifact)
    # ----------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.mres_out)), exist_ok=True)

    payload = {
        "parameters": {
            "beta": float(args.beta) if np.isfinite(args.beta) else None,
            "mass": float(args.mass) if np.isfinite(args.mass) else None,
            "alpha": float(args.alpha),
            "a5": float(args.a5),
            "m5": float(args.m5),
            "mpv": float(args.mpv),
        },
        "analysis_settings": {
            "plateau_start": int(plateau_start),
            "plateau_end": int(plateau_end),
            "therm": int(therm),
            "delta_traj_ps": int(args.delta_traj_ps),
            "n_boot_ratio": 2000,
            "n_boot_plateau": 2000,
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
        },
        # requested naming: mres_extract (NOT mres_fit)
        "mres_extract": {
            "value": float(avg),
            "error": float(err),
            "reduced_chi2": None,
            "plateau_start": int(plateau_start),
            "plateau_end": int(plateau_end),
            "ptll_tau_int_tstart": {
                "t": int(plateau_start),
                "tau_int": float(ptll_tau),
                "tau_int_err": float(ptll_tau_err),
                "Nb_est": int(Nb_est),
                "Nbs_est": int(Nbs_est),
                "found": bool(found),
                "tau_int_dir": str(tau_out_dir),
                "series_file": str(tau_series_path),
            },
        },
    }

    with open(args.mres_out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # ----------------------------------------------------------
    # Plot (MEAS ensemble)
    # ----------------------------------------------------------
    if args.plot_styles:
        parts = [p.strip() for p in str(args.plot_styles).split(",") if p.strip()]
        if parts:
            plt.style.use(parts)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = rf"$\beta={args.beta},\ am_0={args.mass}$" if args.label == "yes" else None
    fit_label = rf"$am_{{\rm res}}^{{\rm extract}} = {avg:.5f}\,\pm\,{err:.5f}$"

    title_str = (
        rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
        rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
    )
    ax.set_title(title_str, fontsize=10)

    ax.errorbar(t_vals, ratio_mean, yerr=ratio_err, fmt="o", color="C4", label=data_label)
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

    ax.set_xlabel("$t/a$")
    ax.set_ylabel("$am_{\\rm res}$")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if data_label or fit_label:
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
        f"(numbers {used_numbers_meas[0]} → {used_numbers_meas[-1]}, delta_traj_ps={args.delta_traj_ps})"
    )
    print(f"✓ Saved plot → {args.plot_file}")
    print(f"✓ Saved JSON → {args.mres_out}")
    print(f"✓ tau_int outputs written in → {tau_out_dir}")
    print(
        f"✓ pt_ll tau_int at t={plateau_start} (FULL series; Berg/2): "
        f"{ptll_tau:.6g} ± {ptll_tau_err:.3g}  (Nb={Nb_est}, Nbs={Nbs_est}, found={bool(found)})"
    )


if __name__ == "__main__":
    main()