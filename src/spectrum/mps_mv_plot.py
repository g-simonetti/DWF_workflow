#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")


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


def read_wflow_summary(filename):
    """Robust reader for wflow summary files."""
    try:
        df = pd.read_csv(filename, sep=r"\s+", comment="#", header=None)
    except Exception as e:
        raise ValueError(f"Could not read wflow file: {filename}\n{e}")

    if df.empty:
        raise ValueError(f"ERROR: wflow file '{filename}' contains no data rows.")

    if df.shape[1] < 2:
        raise ValueError(
            f"ERROR: wflow file '{filename}' does not have at least 2 numeric columns."
        )

    w0_sq = float(df.iloc[0, 0])
    w0_sq_err = float(df.iloc[0, 1])
    return w0_sq, w0_sq_err


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
# Error propagation (m w0)^2
# ============================================================
def mw0_sq_and_error(am, am_err, w0_sq, w0_sq_err):
    X = (am**2) * w0_sq
    dX_dam = 2 * am * w0_sq
    dX_dw0 = am**2
    var = (dX_dam**2) * (am_err**2) + (dX_dw0**2) * (w0_sq_err**2)
    return X, np.sqrt(var)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot (m_ps w0)^2 vs (m_v w0)^2 with error bars.")
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--spectrum", nargs="+", required=True)
    parser.add_argument("--wflow", nargs="+", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    metadata = read_metadata(args.metadata_csv)

    unique_betas = sorted(metadata["beta"].unique())
    unique_masses = sorted(metadata["mass"].unique())

    beta_colors = {b: f"C{i % 10}" for i, b in enumerate(unique_betas)}
    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "x"]
    mass_markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(unique_masses)}

    if len(args.spectrum) != len(args.wflow):
        raise ValueError("Number of --spectrum files must equal number of --wflow files.")

    X, Xerr, Y, Yerr = [], [], [], []
    BETA, MASS = [], []

    for spec_path, wflow_path in zip(args.spectrum, args.wflow):

        beta, mass = extract_beta_mass_from_path(spec_path)
        if beta is None or mass is None:
            raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

        meta_row = metadata[(metadata["beta"] == beta) & (metadata["mass"] == mass)]
        if meta_row.empty:
            raise ValueError(f"No metadata match for beta={beta}, mass={mass} from {spec_path}")

        spec_df = read_spectrum_file(spec_path)
        if spec_df is None:
            continue

        w0_sq, w0_sq_err = read_wflow_summary(wflow_path)

        for _, row in spec_df.iterrows():
            am_ps, am_ps_err = row["am_ps"], row["am_ps_err"]
            am_v, am_v_err   = row["am_v"],  row["am_v_err"]

            # x = (m_ps w0)^2
            x, xerr = mw0_sq_and_error(am_ps, am_ps_err, w0_sq, w0_sq_err)
            # y = (m_v w0)^2
            y, yerr = mw0_sq_and_error(am_v,  am_v_err,  w0_sq, w0_sq_err)

            X.append(x);    Xerr.append(xerr)
            Y.append(y);    Yerr.append(yerr)
            BETA.append(beta); MASS.append(mass)

    X, Xerr, Y, Yerr = map(np.array, (X, Xerr, Y, Yerr))
    BETA, MASS = np.array(BETA), np.array(MASS)

    # ============================================================
    # Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.5), layout="constrained")
    used_labels = set()

    for x, xe, y, ye, b, m in zip(X, Xerr, Y, Yerr, BETA, MASS):
        color = beta_colors[b]
        marker = mass_markers[m]

        label = rf"$\beta={b},\,am_0={m}$"
        if label in used_labels:
            label = None
        else:
            used_labels.add(label)

        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=marker, linestyle="none", color=color, label=label)

    ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
    ax.set_ylabel(r"$(m_{\rm V} w_0)^2$")

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    if used_labels:
        ax.legend()

    plt.savefig(args.output_file, dpi=300)
    plt.close()

    print(f"✓ Saved plot → {args.output_file}")


if __name__ == "__main__":
    main()
