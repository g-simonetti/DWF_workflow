#!/usr/bin/env python3
"""
Wilson flow analysis with multiple trajectories per log file and bootstrap errors.

Assumptions:
  * Each "Reading configuration" line marks the beginning of a new gauge configuration.
  * For each configuration, the log contains lines:
      [WilsonFlow] Energy density (cloverleaf) : step  t  t2E
      [WilsonFlow] Top. charge                 : step  Q
  * We treat each configuration (trajectory) as one member of the ensemble.

Outputs:
  1) --output_file:
       # t   t2E_mean   Q_mean   W_mean   W_err
  2) --summary_file (default: wflow_summary.txt):
       # w0^2  err_w0^2  Q(w0^2)  Q_err(w0^2)  tau_int_Q  tau_int_Q_err
       (single data line)
  3) --plot_file: W(t) with bootstrap error bars, W0 line, and w0^2 line.
"""

import argparse
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Default plotting style similar to the mres script
plt.style.use("tableau-colorblind10")

# ---------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------

READ_CFG_PAT = re.compile(r"Reading configuration")
CLOVER_PAT = re.compile(
    r"Energy density \(cloverleaf\)\s*:\s*(\d+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)"
)
TOP_PAT = re.compile(
    r"Top\. charge\s*:\s*(\d+)\s+([0-9eE.+-]+)"
)


# ---------------------------------------------------------------------
# Parsing a single log file with multiple trajectories
# ---------------------------------------------------------------------

def parse_log_file(path):
    """
    Parse a single log file that may contain multiple trajectories.

    Each trajectory is delimited by a 'Reading configuration' line.

    Returns:
        list_t    : list of 1D np.array flow times per config
        list_t2E  : list of 1D np.array t^2 E(t) per config
        list_Q    : list of 1D np.array Q(t) per config
    """
    list_t = []
    list_t2E = []
    list_Q = []

    # per-configuration temporary accumulators
    step_to_t = {}
    step_to_t2E = {}
    step_to_Q = {}

    def finalize_current_config():
        """Finalize current trajectory if it has data and push to lists."""
        nonlocal step_to_t, step_to_t2E, step_to_Q, list_t, list_t2E, list_Q

        if not step_to_t2E:
            return  # no data

        # keep only steps that have t, t2E, and Q
        common_steps = sorted(
            s for s in step_to_t2E.keys()
            if s in step_to_t and s in step_to_Q
        )
        if not common_steps:
            # no overlapping t2E and Q for this config
            step_to_t = {}
            step_to_t2E = {}
            step_to_Q = {}
            return

        t_arr = np.array([step_to_t[s] for s in common_steps], float)
        t2E_arr = np.array([step_to_t2E[s] for s in common_steps], float)
        Q_arr = np.array([step_to_Q[s] for s in common_steps], float)

        list_t.append(t_arr)
        list_t2E.append(t2E_arr)
        list_Q.append(Q_arr)

        # reset for next configuration
        step_to_t = {}
        step_to_t2E = {}
        step_to_Q = {}

    with open(path, "r") as f:
        for line in f:
            # Start of a new configuration
            if READ_CFG_PAT.search(line):
                # finalize previous one if present
                finalize_current_config()
                continue

            m_clov = CLOVER_PAT.search(line)
            if m_clov:
                step = int(m_clov.group(1))
                t_val = float(m_clov.group(2))
                t2E_val = float(m_clov.group(3))
                step_to_t[step] = t_val
                step_to_t2E[step] = t2E_val
                continue

            m_top = TOP_PAT.search(line)
            if m_top:
                step = int(m_top.group(1))
                Q_val = float(m_top.group(2))
                step_to_Q[step] = Q_val
                continue

    # finalize last configuration at EOF
    finalize_current_config()

    return list_t, list_t2E, list_Q


# ---------------------------------------------------------------------
# Load all logs (multi-file + multi-trajectory)
# ---------------------------------------------------------------------

def load_all_logs(input_path):
    """
    Load all configurations from a directory (or a single log file).

    Returns:
        t        : 1D np.array of common flow times
        t2E_conf : 2D np.array of shape (N_conf, N_t)
        Q_conf   : 2D np.array of shape (N_conf, N_t)
    """
    if os.path.isdir(input_path):

        files = glob.glob(os.path.join(input_path, "wflow*.out"))

        # Sort by key
        word = "wflow"
        ftype = ".out"
        files = sorted(files, key = lambda s : int(os.path.basename(s)[len(word) : -len(ftype)]))

        if not files:
            raise RuntimeError(f"No wflow-*.out files in directory '{input_path}'")
    else:
        files = [input_path]

    all_t = []
    all_t2E = []
    all_Q = []

    for path in files:
        list_t, list_t2E, list_Q = parse_log_file(path)
        all_t.extend(list_t)
        all_t2E.extend(list_t2E)
        all_Q.extend(list_Q)

    if not all_t:
        raise RuntimeError(f"No Wilson flow trajectories found in '{input_path}'")

    # enforce a common t grid
    t_ref = all_t[0]
    for t_arr in all_t[1:]:
        if len(t_arr) != len(t_ref) or not np.allclose(t_arr, t_ref, rtol=1e-12, atol=1e-12):
            raise RuntimeError("Inconsistent flow times across configurations.")

    t2E_conf = np.vstack(all_t2E)
    Q_conf = np.vstack(all_Q)

    return t_ref, t2E_conf, Q_conf


# ---------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------

def numerical_derivative(y, x):
    return np.gradient(y, x)


def solve_for_w0_squared(t, W, W0):
    """
    Solve W(t) = W0 for t via sign change; returns w0^2.
    """
    diff = W - W0
    idx = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if len(idx) == 0:
        raise RuntimeError("No sign change in W(t) - W0; cannot solve for w0^2.")
    i = idx[0]
    t0, t1 = t[i], t[i+1]
    W0_i, W1 = W[i], W[i+1]
    if W1 == W0_i:
        return 0.5 * (t0 + t1)
    alpha = (W0 - W0_i) / (W1 - W0_i)
    return t0 + alpha * (t1 - t0)


# ---------------------------------------------------------------------
# Autocorrelation (Madras–Sokal style) for Q(w0^2)
# ---------------------------------------------------------------------

def integrated_autocorrelation_time(x, c=5.0, M_max=50):
    """
    Madras–Sokal automatic window selection for tau_int and its error.

    x: 1D array of observable values along the Markov chain (here Q(w0^2)).
    Returns:
        tau_int, tau_int_err
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return 0.5, 0.0

    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.5, 0.0

    # Normalized autocorrelation function with decreasing statistics at large lag
    acf = np.correlate(x, x, mode="full")[n - 1:] / (var * np.arange(n, 0, -1))

    candidate_M = range(1, min(M_max, n - 1) + 1)
    tau_int_M = np.array([0.5 + np.sum(acf[1:M + 1]) for M in candidate_M])

    # Automatic window: first M such that M >= c * tau_int(M), or largest M
    valid = [M for M, tau in zip(candidate_M, tau_int_M) if M >= c * tau]
    M_selected = min(valid) if valid else candidate_M[-1]

    tau = 0.5 + np.sum(acf[1:M_selected + 1])
    tau_err = np.sqrt(2.0 * (M_selected + 1) / n) * tau
    return float(tau), float(tau_err)


def bootstrap_indices(n_conf, n_bootstrap):
    return np.random.randint(0, n_conf, size=(n_bootstrap, n_conf))


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze(input_path, label, W0_reference,
            output_file, summary_file, plot_files,
            plot_styles, beta, mass, n_bootstrap):

    # Allow user to override default style if requested
    if plot_styles and plot_styles.lower() != "none":
        try:
            plt.style.use(plot_styles)
        except Exception:
            pass

    # Load configurations from logs
    t, t2E_conf, Q_conf = load_all_logs(input_path)
    n_conf, n_t = t2E_conf.shape
    print(f"Loaded {n_conf} configurations, {n_t} flow-time points each.")

    # Per-config W_c(t) = t * d/dt [ t^2 E_c(t) ]
    W_conf = np.zeros_like(t2E_conf)
    for i in range(n_conf):
        dFdt = numerical_derivative(t2E_conf[i], t)
        W_conf[i] = t * dFdt

    # Ensemble averages
    t2E_mean = t2E_conf.mean(axis=0)
    Q_mean = Q_conf.mean(axis=0)
    W_mean = W_conf.mean(axis=0)

    # Solve for w0^2 from ensemble-averaged W
    w0_sq_central = solve_for_w0_squared(t, W_mean, W0_reference)

    # Ensemble-average Q(t) and Q(w0^2)
    Q_w0_central = float(np.interp(w0_sq_central, t, Q_mean))

    # Per-config Q_c(w0^2) for tau_int
    # This is the time series along the Markov chain of the topological charge
    # at flow time t = w0^2.
    Q_w0_conf = np.array(
        [np.interp(w0_sq_central, t, Q_conf[i]) for i in range(n_conf)],
        float,
    )

    tau_int_central, tau_int_err = integrated_autocorrelation_time(Q_w0_conf)

    # -----------------------------------------------------------------
    # Bootstrap (for W, w0^2, and Q(w0^2) errors)
    # -----------------------------------------------------------------
    np.random.seed()
    idx = bootstrap_indices(n_conf, n_bootstrap)

    W_boot = np.zeros((n_bootstrap, n_t))
    w0_sq_boot = np.zeros(n_bootstrap, float)
    Q_w0_boot = np.zeros(n_bootstrap, float)

    for b in range(n_bootstrap):
        take = idx[b]

        t2E_b = t2E_conf[take].mean(axis=0)
        Q_b = Q_conf[take].mean(axis=0)

        dFdt_b = numerical_derivative(t2E_b, t)
        W_b = t * dFdt_b
        W_boot[b] = W_b

        # Solve for w0^2 from W_b
        try:
            w0_sq_b = solve_for_w0_squared(t, W_b, W0_reference)
        except RuntimeError:
            # If a bootstrap sample fails to cross W0, fall back to central value
            w0_sq_b = w0_sq_central
        w0_sq_boot[b] = w0_sq_b

        # Q(w0^2) from Q_b
        Q_w0_boot[b] = float(np.interp(w0_sq_b, t, Q_b))

    # Bootstrap errors
    W_err = W_boot.std(axis=0, ddof=1)
    w0_sq_err = w0_sq_boot.std(ddof=1)
    Q_w0_err = Q_w0_boot.std(ddof=1)

    # -----------------------------------------------------------------
    # Write detailed output file
    # -----------------------------------------------------------------
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write("# t   t2E_mean   Q_mean   W_mean   W_err\n")
        for i in range(n_t):
            f.write(
                f"{t[i]:.10g}  "
                f"{t2E_mean[i]:.10g}  "
                f"{Q_mean[i]:.10g}  "
                f"{W_mean[i]:.10g}  "
                f"{W_err[i]:.10g}\n"
            )

    # -----------------------------------------------------------------
    # Write summary file (single line, with header comment)
    # -----------------------------------------------------------------
    if os.path.dirname(summary_file):
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    with open(summary_file, "w") as f:
        f.write("# w0^2  err_w0^2  Q(w0^2)  Q_err(w0^2)  tau_int_Q  tau_int_Q_err\n")
        f.write(
            f"{w0_sq_central:.10g}  "
            f"{w0_sq_err:.10g}  "
            f"{Q_w0_central:.10g}  "
            f"{Q_w0_err:.10g}  "
            f"{tau_int_central:.10g}  "
            f"{tau_int_err:.10g}\n"
        )

    # -----------------------------------------------------------------
    # Plot W(t) with bootstrap error bars, with mres-like styling
    # -----------------------------------------------------------------
    # Small figure, constrained layout, scientific notation as in your example
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    # Optional label string, if you decide to use it (e.g., "yes" -> include beta,m)
    if label == "yes":
        data_label = rf"$\beta={beta},\ am_0={mass}$"
    else:
        data_label = r"$\langle W(t)\rangle$"

    ax.errorbar(t, W_mean, yerr=W_err, fmt="o", linestyle="-",label=data_label, color="C4")

    ax.axhline(W0_reference, ls="--", label=rf"$W_0 = {W0_reference}$", color="C0")
    ax.axvline(w0_sq_central, ls="--", label=rf"$t/a = w_0^2/a^2 = {w0_sq_central:.3g}$", color="C1")

    ax.set_xlabel(r"Flow time $t/a$")
    ax.set_ylabel(r"$\mathcal{W}(t/a)$")
    ax.set_ylim(-0.05,1.45)
    #ax.set_title(rf"Wilson flow ($\beta={beta}$, $m={mass}$)")

    # Scientific notation for cleaner axis (like in your m_res plot)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax.legend()

    for pf in plot_files:
        if os.path.dirname(pf):
            os.makedirs(os.path.dirname(pf), exist_ok=True)
        fig.savefig(pf, dpi=300)

    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wilson flow analysis with multiple trajectories per log file."
    )
    parser.add_argument("input", help="Input directory or single log file")
    parser.add_argument("--label", default="", help="yes → include β, m label on plot")
    parser.add_argument("--W0", type=float, required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--summary_file",
        default="wflow_summary.txt",
        help="Summary output file (default: wflow_summary.txt)",
    )
    parser.add_argument("--plot_file", nargs="+", required=True)
    parser.add_argument("--plot_styles", default=None)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)

    args = parser.parse_args()

    analyze(
        input_path=args.input,
        label=args.label,
        W0_reference=args.W0,
        output_file=args.output_file,
        summary_file=args.summary_file,
        plot_files=args.plot_file,
        plot_styles=args.plot_styles,
        beta=args.beta,
        mass=args.mass,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
