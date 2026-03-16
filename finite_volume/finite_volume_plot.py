#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import TABLEAU_COLORS

plt.style.use("tableau-colorblind10")

colours = list(TABLEAU_COLORS)

# ============================================================
# Readers
# ============================================================
def read_spectrum_file(filename):
    """Reads am_ps, am_ps_err, am_v, am_v_err from spectrum.txt.
       If file is empty (after skipping comments), returns None and prints a warning."""
    try:
        df = pd.read_csv(filename, sep=r"\s+", comment="#", header=None)
    except Exception as e:
        raise ValueError(f"Could not read spectrum file: {filename}\n{e}")

    if df.empty:
        print(f"WARNING: spectrum file '{filename}' is empty. Skipping.")
        return None

    expected_cols = [
        "am_ps",
        "am_ps_err",
        "am_v",
        "am_v_err",
        "chi2_ps",
        "chi2_v",
        "plateau_start_ps",
        "plateau_end_ps",
        "plateau_start_v",
        "plateau_end_v",
    ]

    if df.shape[1] >= len(expected_cols):
        df = df.iloc[:, : len(expected_cols)]
        df.columns = expected_cols
    else:
        raise ValueError(
            f"spectrum file '{filename}' has only {df.shape[1]} columns, "
            f"expected at least {len(expected_cols)}."
        )

    return df


def read_metadata(metadata_csv):
    """Reads ensembles.csv (must contain beta and mass)."""
    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError(f"Metadata file '{metadata_csv}' contains no rows.")
    return df


# ============================================================
# Extract β and mass from directory structure
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
            try:
                mass = float(p[1:])
            except ValueError:
                pass

    return beta, mass


# ============================================================
# Extract number of spatial points Ns from path structure
# ============================================================
def extract_Ns_from_path(path):
    parts = path.split(os.sep)
    beta = None
    mass = None

    for p in parts:
        if p.startswith("Ns"):
            try:
                Ns = float(p[2:])
            except ValueError:
                # pass
                print(f"Value Error {p}")
                exit(1)

    return Ns


# ============================================================
# Error propagation (m w0)^2
# ============================================================
def mw0_sq_and_error(am, am_err, w0_sq, w0_sq_err):
    X = (am**2) * w0_sq
    dX_dam = 2 * am * w0_sq
    dX_dw0 = am**2
    var = (dX_dam**2) * (am_err**2) + (dX_dw0**2) * (w0_sq_err**2)
    return X, np.sqrt(var)


# ============================================================
# Label placement
# ============================================================
def place_label(fig, ax, x, y, text, pos):

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    dx = 0.08 * np.cos(np.pi * pos) * width 
    dy = 0.08 * np.sin(np.pi / 2 * pos) *  height 

    xt = x + dx
    yt = y + dy

    ax.text(xt, yt, text, fontsize=7)#,
            #ha="left",
            #va="bottom")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot (a*m_PS, m_PS^inf * L)  with error bars.")
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--spectrum", nargs="+", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--am", required=True)
    parser.add_argument("--beta", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    metadata = read_metadata(args.metadata_csv)

    X, Y, Yerr = [], [], []
    BETA, MASS = float(args.beta), float(args.am)


    # For every ensemble
    for spec_path in args.spectrum:

        beta, mass = extract_beta_mass_from_path(spec_path)
        Ns = extract_Ns_from_path(spec_path)
        if beta is None or mass is None:
            raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

        # Check that the ensembles with this id exist in the metadata
        meta_row = metadata[(metadata["beta"] == beta) & (metadata["mass"] == mass) & (metadata["Ns"] == Ns)]
        if meta_row.empty:
            raise ValueError(f"No metadata match for beta={beta}, mass={mass} from {spec_path}")

        if (beta == BETA) & (mass == MASS):

            print(spec_path)
            # Read the spectrum intermediary data file
            # with the analysis results of the ensemble
            spec_df = read_spectrum_file(spec_path)
            if spec_df is None:
                continue
    
            for _, row in spec_df.iterrows():
                
                am_ps, am_ps_err = row["am_ps"], row["am_ps_err"]

                Y.append(am_ps)    
                Yerr.append(am_ps_err)
    
                
            X.append(Ns)

    # Get plot points as arrays
    X, Y, Yerr = map(np.array, (X, Y, Yerr))

    idx = np.argsort(X)
    X = X[idx]; Y = Y[idx]; Yerr = Yerr[idx]; # sort plot points

    Ns = np.copy(X)
    X = X * Y[-1] # m_PS^inf * L

    m_ps = Y[-1]
    def fit_func(Ns, A, mps_inf):
        #return m_ps * (1 + A * (m_ps)**(3/2) / np.sqrt(Ns) * np.exp(-m_ps * Ns))
        return mps_inf * (1 + A * 1/(mps_inf *  Ns)**(3/2.) * np.exp(-mps_inf * Ns))

    fit_from = 0
    popt, pcov = curve_fit(fit_func, Ns[fit_from:], Y[fit_from:], sigma=Yerr[fit_from:])


    # ============================================================
    # Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.5), layout="constrained")
    used_labels = True #set()

    label = rf"$\beta = {BETA}\, , am_0 = {MASS}$"
    ax.errorbar(X, Y, yerr=Yerr, ls="none", marker="o", label=label)

    xx = np.linspace(np.min(X), np.max(X))
    ns = xx / Y[-1]
    plt.plot(xx, fit_func(ns, *popt), "--", c=colours[0])

    # Plot hline for m_ps^inf
    #plt.hlines(m_ps, *ax.get_xlim(), color="black", label=r"$m_{\mathrm{PS}}^{\mathrm{inf}}$")
    plt.hlines(popt[1], *ax.get_xlim(), color="black", label=r"$m_{\mathrm{PS}}^{\mathrm{inf}}$")

    ax.set_ylabel(r"$am_{\mathrm{PS}}$")
    ax.set_xlabel(r"$m_{\mathrm{PS}}^{\mathrm{inf}} L$")

    # Make N_s labels on points
    for i in range(len(X)):

        pos = i / (len(X)-1)
        Ns_label = rf"$N_s={int(Ns[i])}$" 
        place_label(fig, ax, X[i], Y[i], Ns_label, pos)
    

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    if used_labels:
        ax.legend()


    plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved plot → {args.output_file}")


if __name__ == "__main__":
    main()
