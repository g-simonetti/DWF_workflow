#!/usr/bin/env python3
"""
Wilson flow analysis -> single JSON output + plots.

Writes ONE JSON file containing:
- wflow_table: t, t2E_mean, W_mean, W_err (from W-selection)
- topcharge_series: cfg_id, Q_t0 (full chain)
- topcharge_w0_series: cfg_id, Q_tw0, w0_sq_used (full chain)
- summary: w0, w0_err, Qw0_mean, Qw0_err
- tau_int: w0 and Q(w0) tau_int results (full chain finite series)
- selection info: therm, delta_traj_w, delta_traj_q, counts, ranges

Plots:
- W(t)
- Q(t=0) histogram+timeseries, only for cfg_id >= therm
- Q(w0) histogram+timeseries, only for cfg_id >= therm
- Optional: for a chosen configuration, Q(t) vs Wilson flow time

Plotting info:
- Q(t=0) and Q(w0) plots only show configurations with cfg_id >= therm
- horizontal guide lines are drawn at integer Q values
"""

import argparse
import glob
import os
import re
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")

PLOT_DATA_COLOR = "C4"
PLOT_REFERENCE_COLOR = "C1"
PLOT_HIGHLIGHT_COLOR = "C2"
PLOT_GUIDE_COLOR = "gray"

# -----------------------------------------------------------------------------
# Ensure src/ is importable so we can import autocorr_time.tau_int
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_BOOTSTRAP_DIR = _SRC_DIR / "bootstrap"
if str(_BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_DIR))

try:
    from autocorr_time.tau_int import compute_tau_from_file
except Exception as e:
    raise ImportError(
        "Failed to import compute_tau_from_file from src/autocorr_time/tau_int.py.\n"
        "Expected repo layout with 'src/autocorr_time/tau_int.py'."
    ) from e

try:
    from bootstrap import bootstrap_from_path, bootstrap_to_jsonable
except Exception as e:
    raise ImportError(
        "Failed to import bootstrap helpers from src/bootstrap/bootstrap.py."
    ) from e

# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------
CFGID_PAT = re.compile(r"Reading configuration:\s*.*?cfg_ckpoint\.(\d+)")
CLOVER_PAT = re.compile(
    r"Energy density \(cloverleaf\)\s*:\s*(\d+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)"
)
TOPCHARGE_PAT = re.compile(r"Top\.\s*charge\s*:\s*(\d+)\s+([0-9eE.+-]+)")


def parse_log_file(path: str):
    """Parse one Wilson flow output file."""
    list_cfg, list_t, list_t2E, list_Q = [], [], [], []
    step_to_t, step_to_t2E, step_to_Q = {}, {}, {}
    current_cfgid = None

    def finalize_current_config():
        nonlocal step_to_t, step_to_t2E, step_to_Q, current_cfgid

        if not step_to_t2E:
            current_cfgid = None
            step_to_t, step_to_t2E, step_to_Q = {}, {}, {}
            return

        common_steps = sorted([s for s in step_to_t2E.keys() if s in step_to_t])
        if common_steps:
            t_arr = np.array([step_to_t[s] for s in common_steps], dtype=float)
            t2E_arr = np.array([step_to_t2E[s] for s in common_steps], dtype=float)

            Q_arr = np.full_like(t_arr, np.nan, dtype=float)
            for j, s in enumerate(common_steps):
                if s in step_to_Q:
                    Q_arr[j] = float(step_to_Q[s])

            list_t.append(t_arr)
            list_t2E.append(t2E_arr)
            list_Q.append(Q_arr)
            list_cfg.append(int(current_cfgid) if current_cfgid is not None else -1)

        step_to_t, step_to_t2E, step_to_Q = {}, {}, {}
        current_cfgid = None

    with open(path, "r") as f:
        for line in f:
            mcfg = CFGID_PAT.search(line)
            if mcfg:
                finalize_current_config()
                current_cfgid = int(mcfg.group(1))
                continue

            m = CLOVER_PAT.search(line)
            if m:
                step = int(m.group(1))
                step_to_t[step] = float(m.group(2))
                step_to_t2E[step] = float(m.group(3))
                continue

            mq = TOPCHARGE_PAT.search(line)
            if mq:
                step = int(mq.group(1))
                step_to_Q[step] = float(mq.group(2))
                continue

    finalize_current_config()
    return np.array(list_cfg, dtype=int), list_t, list_t2E, list_Q


def load_all_logs(input_path: str):
    """Load and merge all Wilson flow logs."""
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "wflow*.out")))
        if not files:
            raise RuntimeError(f"No wflow*.out files in directory '{input_path}'")
    else:
        files = [input_path]

    all_cfg, all_t, all_t2E, all_Q = [], [], [], []
    for p in files:
        cfg, lt, lt2E, lQ = parse_log_file(p)
        all_cfg.extend(cfg.tolist())
        all_t.extend(lt)
        all_t2E.extend(lt2E)
        all_Q.extend(lQ)

    if not all_t:
        raise RuntimeError(f"No Wilson flow trajectories found in '{input_path}'")

    t_ref = all_t[0]
    for t_arr in all_t[1:]:
        if len(t_arr) != len(t_ref) or not np.allclose(
            t_arr, t_ref, rtol=1e-12, atol=1e-12
        ):
            raise RuntimeError("Inconsistent flow times across configurations.")

    cfg_ids = np.array(all_cfg, dtype=int)
    t2E = np.vstack(all_t2E)
    Q = np.vstack(all_Q)

    if np.any(cfg_ids < 0):
        bad = int(np.sum(cfg_ids < 0))
        raise RuntimeError(f"Failed to parse cfg_ckpoint.<N> for {bad} configurations.")

    order = np.argsort(cfg_ids)
    return cfg_ids[order], t_ref, t2E[order], Q[order]


def numerical_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Numerical derivative dy/dx."""
    return np.gradient(y, x)


def solve_for_w0_squared(t: np.ndarray, W: np.ndarray, W0: float) -> float:
    """Solve W(t)=W0 by linear interpolation between first bracketing points."""
    diff = W - W0
    idx = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if len(idx) == 0:
        raise RuntimeError("No sign change in W(t) - W0; cannot solve for w0^2.")
    i = int(idx[0])
    t0, t1 = float(t[i]), float(t[i + 1])
    W0_i, W1 = float(W[i]), float(W[i + 1])
    if W1 == W0_i:
        return 0.5 * (t0 + t1)
    a = (W0 - W0_i) / (W1 - W0_i)
    return t0 + a * (t1 - t0)


def select_indices_by_cfgid(cfg_ids: np.ndarray, therm: int, delta_traj: int) -> np.ndarray:
    """Select configurations by cfg_id >= therm and stride in cfg_id space."""
    therm = int(therm)
    delta_traj = max(1, int(delta_traj))
    cfg_ids = np.asarray(cfg_ids, dtype=int)
    mask = (cfg_ids >= therm) & (((cfg_ids - therm) % delta_traj) == 0)
    return np.where(mask)[0].astype(int)


def interp_nan_safe(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """1D linear interpolation that ignores NaNs and returns NaN if invalid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    if np.count_nonzero(m) < 2:
        return np.nan
    xf, yf = x[m], y[m]
    xmin, xmax = float(np.min(xf)), float(np.max(xf))
    if not (xmin <= float(x0) <= xmax):
        return np.nan
    return float(np.interp(float(x0), xf, yf))


def integer_q_range(values: np.ndarray):
    """Return a symmetric integer Q range centered on zero."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    qabs = int(np.ceil(np.max(np.abs(vals))))
    return -qabs, qabs


def add_integer_horizontal_lines(ax, qmin: int, qmax: int):
    """Draw horizontal guide lines at integer Q."""
    for qint in range(qmin, qmax + 1):
        ax.axhline(
            qint, ls=":", color=PLOT_GUIDE_COLOR, linewidth=0.45, alpha=0.65, zorder=0
        )


def set_symmetric_q_ylim(ax, values: np.ndarray):
    """Set symmetric Q-axis limits around zero with a small padding."""
    q_range = integer_q_range(values)
    if q_range is None:
        return

    _, qmax = q_range
    q_abs_max = max(0.5, float(qmax) + 0.5)
    ax.set_ylim(-q_abs_max, q_abs_max)


def add_horizontal_histogram(ax, values: np.ndarray, q_range):
    """Draw a horizontal histogram with a zero-centered Gaussian overlay."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or q_range is None:
        return

    qmin, qmax = q_range
    bins = np.arange(qmin - 0.5, qmax + 1.5, 1.0)

    ax.hist(
        vals,
        bins=bins,
        density=True,
        orientation="horizontal",
        color=PLOT_DATA_COLOR,
        alpha=0.25,
    )

    sigma = float(np.sqrt(np.mean(vals**2)))
    sigma = max(sigma, 1.0e-6)
    y_dense = np.linspace(bins[0], bins[-1], 400)
    x_dense = np.exp(-0.5 * (y_dense / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    ax.plot(x_dense, y_dense, color=PLOT_REFERENCE_COLOR, linewidth=1.2)


def analyze(
    input_path: str,
    label: str,
    W0_reference: float,
    wflow_out: str,
    plot_wflow: list[str],
    plot_q0: list[str],
    plot_qw0: list[str],
    plot_q_cfg: list[str] | None,
    q_cfg_id: int | None,
    plot_styles: str | None,
    beta: float,
    mass: float,
    n_bootstrap: int,
    therm: int,
    delta_traj_w: int,
    delta_traj_q: int,
):
    """Main analysis routine."""
    if plot_styles and str(plot_styles).lower() != "none":
        parts = [p.strip() for p in str(plot_styles).split(",") if p.strip()]
        if parts:
            plt.style.use(parts)

    cfg_ids, t, t2E_conf, Q_conf = load_all_logs(input_path)
    n_conf, n_t = t2E_conf.shape
    print(f"Loaded {n_conf} configurations, {n_t} flow-time points each.")
    print(f"[cfg] cfg_id range: {int(cfg_ids.min())} .. {int(cfg_ids.max())}")

    # -------------------------------------------------------------------------
    # Per-configuration W(t)
    # -------------------------------------------------------------------------
    W_conf = np.zeros_like(t2E_conf)
    for i in range(n_conf):
        dFdt = numerical_derivative(t2E_conf[i], t)
        W_conf[i] = t * dFdt

    # -------------------------------------------------------------------------
    # Full-chain w0, q(t=0), q(w0)
    # -------------------------------------------------------------------------
    w0_sq_all = np.full(n_conf, np.nan, dtype=float)
    n_fail_all = 0
    for i in range(n_conf):
        try:
            w0_sq_all[i] = solve_for_w0_squared(t, W_conf[i], W0_reference)
        except RuntimeError:
            n_fail_all += 1

    w0_all = np.full(n_conf, np.nan, dtype=float)
    m_w0 = np.isfinite(w0_sq_all) & (w0_sq_all > 0)
    w0_all[m_w0] = np.sqrt(w0_sq_all[m_w0])

    q0_all = Q_conf[:, 0].astype(float)

    q_w0_all = np.full(n_conf, np.nan, dtype=float)
    for i in range(n_conf):
        if np.isfinite(w0_sq_all[i]):
            q_w0_all[i] = interp_nan_safe(t, Q_conf[i], w0_sq_all[i])

    # -------------------------------------------------------------------------
    # tau_int(w0) from full finite series
    # -------------------------------------------------------------------------
    tau_w0 = tau_w0_err = np.nan
    Nb_w0 = Nbs_w0 = np.nan
    found_w0 = False
    cfg_ids_tau_w0 = cfg_ids[np.isfinite(w0_all)]
    w0_series_full = w0_all[np.isfinite(w0_all)]
    n_tau_used_w0 = int(w0_series_full.size)

    # tau_int(Qw0)
    tau_q = tau_q_err = np.nan
    Nb_q = Nbs_q = np.nan
    found_q = False
    cfg_ids_tau_q = cfg_ids[np.isfinite(q_w0_all)]
    q_w0_series_full = q_w0_all[np.isfinite(q_w0_all)]
    n_tau_used_q = int(q_w0_series_full.size)

    out_dir = os.path.dirname(os.path.abspath(wflow_out)) or "."

    if n_tau_used_w0 >= 4:
        tau_out_dir = os.path.join(out_dir, "tau_int_w0")
        os.makedirs(tau_out_dir, exist_ok=True)
        w0_series_file = os.path.join(tau_out_dir, "w0_series.txt")
        with open(w0_series_file, "w") as f:
            f.write("# cfg_id   w0\n")
            for cfg, val in zip(cfg_ids_tau_w0, w0_series_full):
                f.write(f"{int(cfg)}  {float(val):.16g}\n")
        tau_w0, tau_w0_err, Nb_w0, Nbs_w0, found_w0 = compute_tau_from_file(
            input_file=w0_series_file,
            out_dir=tau_out_dir,
            therm=0,
            plot_styles=plot_styles,
            base_name="w0_tau_int",
        )

    if n_tau_used_q >= 4:
        tau_out_dir_q = os.path.join(out_dir, "tau_int_Qw0")
        os.makedirs(tau_out_dir_q, exist_ok=True)
        q_series_file = os.path.join(tau_out_dir_q, "Qw0_series.txt")
        with open(q_series_file, "w") as f:
            f.write("# cfg_id   Qw0\n")
            for cfg, val in zip(cfg_ids_tau_q, q_w0_series_full):
                f.write(f"{int(cfg)}  {float(val):.16g}\n")
        tau_q, tau_q_err, Nb_q, Nbs_q, found_q = compute_tau_from_file(
            input_file=q_series_file,
            out_dir=tau_out_dir_q,
            therm=0,
            plot_styles=plot_styles,
            base_name="Qw0_tau_int",
        )

    # -------------------------------------------------------------------------
    # W-selection for W(t) + bootstrap w0
    # -------------------------------------------------------------------------
    idx_sel_w = select_indices_by_cfgid(cfg_ids, therm=therm, delta_traj=delta_traj_w)
    n_plot_used_w = int(idx_sel_w.size)
    if n_plot_used_w == 0:
        raise RuntimeError(
            f"W selection empty: therm={therm}, delta_traj_w={delta_traj_w}"
        )

    t2E_sel = t2E_conf[idx_sel_w]
    W_sel = W_conf[idx_sel_w]
    cfg_ids_sel_w = cfg_ids[idx_sel_w]
    t2E_mean = t2E_sel.mean(axis=0)
    W_mean = W_sel.mean(axis=0)

    w0_sq_central = solve_for_w0_squared(t, W_mean, W0_reference)
    w0_central = np.sqrt(w0_sq_central) if w0_sq_central > 0 else np.nan

    bootstrap_w = bootstrap_from_path(input_path, cfg_ids_sel_w, n_bootstrap)
    idx_boot_w = np.asarray(bootstrap_w["boot_idx"], dtype=int)
    W_boot = np.zeros((n_bootstrap, n_t), dtype=float)
    w0_sq_boot = np.full(n_bootstrap, np.nan, dtype=float)
    w0_boot = np.full(n_bootstrap, np.nan, dtype=float)

    for b in range(n_bootstrap):
        take = idx_boot_w[b]
        W_b = W_sel[take].mean(axis=0)
        W_boot[b] = W_b
        try:
            w0_sq_b = solve_for_w0_squared(t, W_b, W0_reference)
            w0_sq_boot[b] = w0_sq_b
            w0_boot[b] = np.sqrt(w0_sq_b) if w0_sq_b > 0 else np.nan
        except RuntimeError:
            w0_sq_boot[b] = w0_sq_central
            w0_boot[b] = w0_central

    W_err = W_boot.std(axis=0, ddof=1)
    w0_err = np.nanstd(w0_boot, ddof=1)
    w0_boot_samples = []
    for b in range(n_bootstrap):
        w0_boot_samples.append(
            {
                "index": int(b),
                "w0": float(w0_boot[b]) if np.isfinite(w0_boot[b]) else None,
            }
        )

    # -------------------------------------------------------------------------
    # Q-selection for Q(w0) bootstrap
    # -------------------------------------------------------------------------
    idx_sel_q = select_indices_by_cfgid(cfg_ids, therm=therm, delta_traj=delta_traj_q)
    n_plot_used_q = int(idx_sel_q.size)
    if n_plot_used_q == 0:
        raise RuntimeError(
            f"Q selection empty: therm={therm}, delta_traj_q={delta_traj_q}"
        )

    cfg_ids_sel_q = cfg_ids[idx_sel_q]
    q_w0_sel = q_w0_all[idx_sel_q]
    q_w0_sel_finite = q_w0_sel[np.isfinite(q_w0_sel)]
    q_w0_mean = (
        float(np.mean(q_w0_sel_finite)) if q_w0_sel_finite.size > 0 else np.nan
    )

    bootstrap_q = bootstrap_from_path(input_path, cfg_ids_sel_q, n_bootstrap)
    idx_boot_q = np.asarray(bootstrap_q["boot_idx"], dtype=int)
    q_w0_boot = np.full(n_bootstrap, np.nan, dtype=float)
    for b in range(n_bootstrap):
        take_q = idx_boot_q[b]
        qw0_take = q_w0_sel[take_q]
        qw0_take = qw0_take[np.isfinite(qw0_take)]
        q_w0_boot[b] = float(np.mean(qw0_take)) if qw0_take.size > 0 else np.nan
    q_w0_err = np.nanstd(q_w0_boot, ddof=1)

    # -------------------------------------------------------------------------
    # JSON output
    # -------------------------------------------------------------------------
    payload = {
        "inputs": {
            "W0_reference": float(W0_reference),
            "beta": float(beta),
            "mass": float(mass),
            "therm": int(therm),
            "delta_traj_w": int(delta_traj_w),
            "delta_traj_q": int(delta_traj_q),
            "n_bootstrap": int(n_bootstrap),
            "q_cfg_id": int(q_cfg_id) if q_cfg_id is not None else None,
        },
        "counts": {
            "n_total": int(n_conf),
            "n_tau_used_w0": int(n_tau_used_w0),
            "n_tau_used_Qw0": int(n_tau_used_q),
            "n_plot_used_w": int(n_plot_used_w),
            "n_plot_used_q": int(n_plot_used_q),
            "n_fail_w0_cross": int(n_fail_all),
        },
        "ranges": {
            "cfg_id_min": int(cfg_ids.min()),
            "cfg_id_max": int(cfg_ids.max()),
            "flow_t_min": float(np.min(t)),
            "flow_t_max": float(np.max(t)),
        },
        "bootstrap": {
            "w0": {
                **bootstrap_to_jsonable(bootstrap_w),
                "samples": w0_boot_samples,
            },
            "Qw0": bootstrap_to_jsonable(bootstrap_q),
        },
        "wflow_table": {
            "t": [float(x) for x in t],
            "t2E_mean": [float(x) for x in t2E_mean],
            "W_mean": [float(x) for x in W_mean],
            "W_err": [float(x) for x in W_err],
        },
        "topcharge_series": {
            "cfg_id": [int(x) for x in cfg_ids],
            "Q_t0": [float(x) for x in q0_all],
        },
        "topcharge_w0_series": {
            "cfg_id": [int(x) for x in cfg_ids],
            "Q_tw0": [float(x) for x in q_w0_all],
            "w0_sq_used": [float(x) for x in w0_sq_all],
        },
        "summary": {
            "w0": float(w0_central) if np.isfinite(w0_central) else None,
            "w0_err": float(w0_err) if np.isfinite(w0_err) else None,
            "Qw0_mean": float(q_w0_mean) if np.isfinite(q_w0_mean) else None,
            "Qw0_err": float(q_w0_err) if np.isfinite(q_w0_err) else None,
        },
        "tau_int": {
            "w0": {
                "tau_int": float(tau_w0) if np.isfinite(tau_w0) else None,
                "tau_int_err": float(tau_w0_err) if np.isfinite(tau_w0_err) else None,
                "Nb": float(Nb_w0) if np.isfinite(Nb_w0) else None,
                "Nbs": float(Nbs_w0) if np.isfinite(Nbs_w0) else None,
                "found": bool(found_w0),
                "out_dir": os.path.join(out_dir, "tau_int_w0"),
            },
            "Qw0": {
                "tau_int": float(tau_q) if np.isfinite(tau_q) else None,
                "tau_int_err": float(tau_q_err) if np.isfinite(tau_q_err) else None,
                "Nb": float(Nb_q) if np.isfinite(Nb_q) else None,
                "Nbs": float(Nbs_q) if np.isfinite(Nbs_q) else None,
                "found": bool(found_q),
                "out_dir": os.path.join(out_dir, "tau_int_Qw0"),
            },
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(wflow_out)) or ".", exist_ok=True)
    with open(wflow_out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    print(f"[json] wrote {wflow_out}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # W(t)
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    data_label = (
        rf"$\beta={beta},\ am_0={mass}$"
        if str(label).lower() == "yes"
        else r"$\langle W(t)\rangle$"
    )
    ax.errorbar(
        t,
        W_mean,
        yerr=W_err,
        fmt="o",
        linestyle="-",
        label=data_label,
        color=PLOT_DATA_COLOR,
    )
    ax.axhline(
        W0_reference,
        ls="--",
        label=rf"$W_0 = {W0_reference}$",
        color=PLOT_HIGHLIGHT_COLOR,
    )
    ax.axvline(
        w0_sq_central,
        ls="--",
        label=rf"$w_0^2/a^2 = {w0_sq_central:.3g}$",
        color=PLOT_REFERENCE_COLOR,
    )
    ax.set_xlabel(r"Flow time $t/a^2$")
    ax.set_ylabel(r"$\mathcal{W}(t)$")
    ax.set_ylim(-0.05, 1.45)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.legend()
    for pf in plot_wflow:
        os.makedirs(os.path.dirname(pf) or ".", exist_ok=True)
        fig.savefig(pf, dpi=300)
    plt.close(fig)

    # Q(t=0) -- only thermalised configurations
    mask_q0 = cfg_ids >= int(therm)
    cfg_ids_q0 = cfg_ids[mask_q0]
    q0_plot = q0_all[mask_q0]
    title_str = rf"$\beta={beta},\ am_0={mass}$"

    fig = plt.figure(figsize=(3.5, 2.5), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1], sharey=ax)
    ax.set_title(title_str, fontsize=10)

    q0_rng = integer_q_range(q0_plot)
    if q0_rng is not None:
        add_integer_horizontal_lines(ax, q0_rng[0], q0_rng[1])
        add_integer_horizontal_lines(axh, q0_rng[0], q0_rng[1])
    set_symmetric_q_ylim(ax, q0_plot)

    ax.plot(cfg_ids_q0, q0_plot, marker="o", linestyle="-", color=PLOT_DATA_COLOR)
    ax.set_xlabel("HMC time")
    ax.set_ylabel(r"$Q(t_{\rm flow} = 0)$")

    q0_f = q0_plot[np.isfinite(q0_plot)]
    if q0_f.size > 0 and q0_rng is not None:
        add_horizontal_histogram(axh, q0_f, q0_rng)

    axh.set_xlabel("$P(Q)$")
    axh.tick_params(axis="y", labelleft=False)

    for pf in plot_q0:
        os.makedirs(os.path.dirname(pf) or ".", exist_ok=True)
        fig.savefig(pf, dpi=300)
    plt.close(fig)

    # Q(w0) -- only thermalised configurations
    mask_qw0 = cfg_ids >= int(therm)
    cfg_ids_qw0 = cfg_ids[mask_qw0]
    q_w0_plot = q_w0_all[mask_qw0]

    fig = plt.figure(figsize=(3.5, 2.5), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1], sharey=ax)
    ax.set_title(title_str, fontsize=10)

    qw0_rng = integer_q_range(q_w0_plot)
    if qw0_rng is not None:
        add_integer_horizontal_lines(ax, qw0_rng[0], qw0_rng[1])
        add_integer_horizontal_lines(axh, qw0_rng[0], qw0_rng[1])
    set_symmetric_q_ylim(ax, q_w0_plot)

    ax.plot(cfg_ids_qw0, q_w0_plot, marker="o", linestyle="-", color=PLOT_DATA_COLOR)
    ax.set_xlabel("HMC time")
    ax.set_ylabel(r"$Q(t_{\rm flow}=w_0^2)$")

    qw0_f = q_w0_plot[np.isfinite(q_w0_plot)]
    if qw0_f.size > 0 and qw0_rng is not None:
        add_horizontal_histogram(axh, qw0_f, qw0_rng)

    axh.set_xlabel("$P(Q)$")
    axh.tick_params(axis="y", labelleft=False)

    for pf in plot_qw0:
        os.makedirs(os.path.dirname(pf) or ".", exist_ok=True)
        fig.savefig(pf, dpi=300)
    plt.close(fig)

    # Optional: Q(t) for a chosen configuration vs Wilson flow time
    if q_cfg_id is not None and plot_q_cfg:
        matches = np.where(cfg_ids == int(q_cfg_id))[0]
        if matches.size == 0:
            raise RuntimeError(
                f"Requested q_cfg_id={q_cfg_id}, but this configuration was not found."
            )

        i_cfg = int(matches[0])
        q_cfg = Q_conf[i_cfg]

        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
        ax.set_title(title_str, fontsize=10)

        q_rng = integer_q_range(q_cfg)
        if q_rng is not None:
            add_integer_horizontal_lines(ax, q_rng[0], q_rng[1])
        set_symmetric_q_ylim(ax, q_cfg)

        ax.plot(
            t,
            q_cfg,
            marker="o",
            linestyle="-",
            label=rf"cfg {q_cfg_id}",
            color=PLOT_DATA_COLOR,
        )

        if np.isfinite(w0_sq_all[i_cfg]):
            ax.axvline(
                w0_sq_all[i_cfg],
                ls="--",
                color=PLOT_REFERENCE_COLOR,
                label=rf"$w_0^2/a^2 = {w0_sq_all[i_cfg]:.3g}$",
            )

        ax.set_xlabel(r"Flow time $t/a^2$")
        ax.set_ylabel(r"$Q(t)$")
        ax.legend()

        for pf in plot_q_cfg:
            os.makedirs(os.path.dirname(pf) or ".", exist_ok=True)
            fig.savefig(pf, dpi=300)
        plt.close(fig)

    if n_fail_all > 0:
        print(f"[w0] Note: {n_fail_all}/{n_conf} configs failed to cross W0.")


def main():
    ap = argparse.ArgumentParser(
        description="Wilson flow with Q(t) -> single JSON + plots."
    )
    ap.add_argument(
        "input",
        help="Input directory (containing wflow*.out) or single log file",
    )

    ap.add_argument("--label", default="")
    ap.add_argument("--W0", type=float, required=True)

    ap.add_argument("--wflow_out", required=True)

    ap.add_argument("--plot_wflow", nargs="+", required=True)
    ap.add_argument("--plot_q0", nargs="+", required=True)
    ap.add_argument("--plot_qw0", nargs="+", required=True)

    ap.add_argument(
        "--plot_q_cfg",
        nargs="+",
        default=None,
        help="Output file(s) for Q(t) vs flow-time plot of a single configuration",
    )
    ap.add_argument(
        "--q_cfg_id",
        type=int,
        default=None,
        help="Configuration ID for single-config Q(t) plot",
    )

    ap.add_argument("--plot_styles", default=None)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--n_boot", type=int, default=200)

    ap.add_argument("--therm", type=int, default=0)
    ap.add_argument("--delta_traj_w", type=int, default=1)
    ap.add_argument("--delta_traj_q", type=int, default=1)

    args = ap.parse_args()

    if (args.plot_q_cfg is None) != (args.q_cfg_id is None):
        ap.error("--plot_q_cfg and --q_cfg_id must be provided together")

    analyze(
        input_path=args.input,
        label=args.label,
        W0_reference=args.W0,
        wflow_out=args.wflow_out,
        plot_wflow=args.plot_wflow,
        plot_q0=args.plot_q0,
        plot_qw0=args.plot_qw0,
        plot_q_cfg=args.plot_q_cfg,
        q_cfg_id=args.q_cfg_id,
        plot_styles=args.plot_styles,
        beta=args.beta,
        mass=args.mass,
        n_bootstrap=args.n_boot,
        therm=args.therm,
        delta_traj_w=args.delta_traj_w,
        delta_traj_q=args.delta_traj_q,
    )


if __name__ == "__main__":
    main()
