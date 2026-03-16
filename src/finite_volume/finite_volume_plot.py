#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")


# ============================================================
# Readers
# ============================================================
def read_spectrum_json(filename):
    """
    Read am_ps mean and sdev from spectrum.json.

    Expected structure:
      standard_fit -> PP -> am_ps -> mean
      standard_fit -> PP -> am_ps -> sdev
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not read JSON file: {filename}\n{e}")

    try:
        am_ps = data["results"]["bootstrap_fit"]["PP"]["am_ps"]["mean"]
        am_ps_err = data["results"]["bootstrap_fit"]["PP"]["am_ps"]["sdev"]
    except KeyError as e:
        raise ValueError(
            f"Missing key in spectrum JSON '{filename}': {e}\n"
            "Expected: standard_fit -> PP -> am_ps -> mean/sdev"
        )

    return float(am_ps), float(am_ps_err)


# ============================================================
# Extract beta and mass from directory structure
# ============================================================
def extract_beta_mass_from_path(path):
    parts = path.split(os.sep)
    beta = None
    mass = None

    for p in parts:
        if p.startswith("B"):
            try:
                beta = float(p[1:])
            except ValueError:
                pass

        elif p.startswith("M") and mass is None:
            # this catches M{mass}, not M5{...}, because mass is set only once
            try:
                mass = float(p[1:])
            except ValueError:
                pass

    return beta, mass


# ============================================================
# Extract Ns from path structure
# ============================================================
def extract_Ns_from_path(path):
    parts = path.split(os.sep)

    for p in parts:
        if p.startswith("Ns"):
            try:
                return float(p[2:])
            except ValueError:
                raise ValueError(f"Could not parse Ns from path component '{p}'")

    raise ValueError(f"Could not extract Ns from path: {path}")


# ============================================================
# Label placement
# ============================================================
def place_label(ax, x, y, text, pos):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    dx = 0.08 * np.cos(np.pi * pos) * width
    dy = 0.08 * np.sin(np.pi / 2 * pos) * height

    xt = x + dx
    yt = y + dy

    ax.text(xt, yt, text, fontsize=7)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Finite-volume plot from spectrum.json files."
    )
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--m_ps", nargs="+", required=True,
                        help="Input spectrum.json files")
    parser.add_argument("--fv_out", required=True,
                        help="Output JSON summary")
    parser.add_argument("--fv_plot", required=True,
                        help="Output plot file")
    parser.add_argument("--beta", required=True)
    parser.add_argument("--mass", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    BETA = float(args.beta)
    MASS = float(args.mass)

    X = []
    Y = []
    Yerr = []
    selected_files = []

    # ========================================================
    # Collect data from all matching spectrum.json files
    # ========================================================
    for spec_path in args.m_ps:
        beta, mass = extract_beta_mass_from_path(spec_path)
        Ns = extract_Ns_from_path(spec_path)

        if beta is None or mass is None:
            raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

        # Keep this extra check for robustness, even if Snakemake already filtered
        if not (np.isclose(beta, BETA) and np.isclose(mass, MASS)):
            continue

        am_ps, am_ps_err = read_spectrum_json(spec_path)

        X.append(Ns)
        Y.append(am_ps)
        Yerr.append(am_ps_err)
        selected_files.append(spec_path)

    if len(X) == 0:
        raise ValueError(
            f"No input files matched beta={BETA}, mass={MASS} among provided --m_ps files."
        )

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    Yerr = np.array(Yerr, dtype=float)

    idx = np.argsort(X)
    X = X[idx]
    Y = Y[idx]
    Yerr = Yerr[idx]
    selected_files = [selected_files[i] for i in idx]

    Ns = np.copy(X)

    # Use largest-Ns value as infinite-volume proxy
    m_ps_inf = Y[-1]
    X_plot = Ns * m_ps_inf  # m_PS^inf * L

    fit_result = None

    # ========================================================
    # Fit
    # ========================================================
    def fit_func(L, A):
        return m_ps_inf * (
            1.0 + A * np.exp(-m_ps_inf * L) / (m_ps_inf * L) ** 1.5
        )

    if len(Ns) >= 2:
        try:
            popt, pcov = curve_fit(
                fit_func,
                Ns,
                Y,
                sigma=Yerr,
                absolute_sigma=True,
                maxfev=10000,
            )
            fit_result = {
                "A_mean": float(popt[0]),
                "A_sdev": float(np.sqrt(pcov[0, 0])) if pcov.size else None,
            }
        except Exception as e:
            print(f"WARNING: fit failed: {e}")
            popt = None
            pcov = None
    else:
        popt = None
        pcov = None
        print("WARNING: fewer than 2 points; skipping fit.")

    # ========================================================
    # Save JSON summary
    # ========================================================
    out_data = {
        "label": args.label,
        "beta": BETA,
        "mass": MASS,
        "m_ps_inf": float(m_ps_inf),
        "Ns": [float(v) for v in Ns],
        "m_ps": [float(v) for v in Y],
        "m_ps_err": [float(v) for v in Yerr],
        "m_ps_inf_L": [float(v) for v in X_plot],
        "input_files": selected_files,
        "fit": fit_result,
    }

    os.makedirs(os.path.dirname(args.fv_out), exist_ok=True)
    with open(args.fv_out, "w") as f:
        json.dump(out_data, f, indent=2)

    # ========================================================
    # Plot
    # ========================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.5), layout="constrained")

    if args.label:
        legend_label = rf"{args.label}: $\beta={BETA},\, am_0={MASS}$"
    else:
        legend_label = rf"$\beta={BETA},\, am_0={MASS}$"

    ax.errorbar(
        X_plot,
        Y,
        yerr=Yerr,
        ls="none",
        marker="o",
        label=legend_label,
    )

    if popt is not None:
        xx = np.linspace(np.min(X_plot), np.max(X_plot), 200)
        ns_fit = xx / m_ps_inf
        ax.plot(xx, fit_func(ns_fit, *popt), "k--")

    ax.set_ylabel(r"$am_{PS}$")
    ax.set_xlabel(r"$m_{PS}^{\infty} L$")

    # Point labels
    if len(X_plot) == 1:
        place_label(ax, X_plot[0], Y[0], rf"$N_s={int(Ns[0])}$", 0.5)
    else:
        for i in range(len(X_plot)):
            pos = i / (len(X_plot) - 1)
            place_label(ax, X_plot[i], Y[i], rf"$N_s={int(Ns[i])}$", pos)

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.legend()

    os.makedirs(os.path.dirname(args.fv_plot), exist_ok=True)
    plt.savefig(args.fv_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved JSON  → {args.fv_out}")
    print(f"✓ Saved plot  → {args.fv_plot}")


if __name__ == "__main__":
    main()