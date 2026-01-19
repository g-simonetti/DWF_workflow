#!/usr/bin/env python3
"""
spectrum.py — compute pseudoscalar and vector effective masses using cosh,
bootstrap errors, and plateau fits controlled by Snakemake parameters.
"""

import argparse
import glob
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")


# ===============================================================
# SAFE ATTRIBUTE DECODER
# ===============================================================
def _decode_attr(x):
    """
    Safely decode HDF5 string attributes.

    Handles all formats:
      - bytes → str
      - str → str
      - numpy bytes_ or str_
      - numpy scalar → str
      - 0-dim numpy array → unwrap
      - 1-element array → unwrap element
      - multi-element array → return list of strings
    """
    import numpy as _np

    # bytes → utf8
    if isinstance(x, bytes):
        return x.decode("utf-8")

    # native Python string
    if isinstance(x, str):
        return x

    # numpy scalar (str_, bytes_, others)
    if _np.isscalar(x):
        if isinstance(x.item(), bytes):
            return x.item().decode("utf-8")
        return str(x.item())

    # numpy array
    if isinstance(x, _np.ndarray):
        if x.ndim == 0:           # scalar array
            return _decode_attr(x.item())
        if x.size == 1:           # single element array
            return _decode_attr(x.flat[0])
        # multi-element → return list of strings
        return [_decode_attr(e) for e in x.tolist()]

    # fallback: convert to string
    return str(x)


# ===============================================================
# PSEUDOSCALAR READER (meson_1)
# ===============================================================
def read_ps_corr(filename):
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        return data["re"]


# ===============================================================
# VECTOR READER: average over GammaX/Y/Z channels
# ===============================================================
def read_vec_corr(filename, gammas=("GammaX", "GammaY", "GammaZ")):
    vec = []

    with h5py.File(filename, "r") as f:
        mg = f["meson"]

        for key in mg:
            g = mg[key]
            if "gamma_src" not in g.attrs:
                continue  # skip groups without gamma information

            # decode safely
            gamma_src = _decode_attr(g.attrs["gamma_src"])
            gamma_snk = _decode_attr(g.attrs["gamma_snk"])

            # unwrap lists into single strings
            if isinstance(gamma_src, list):
                gamma_src = gamma_src[0]
            if isinstance(gamma_snk, list):
                gamma_snk = gamma_snk[0]

            # channel selection
            if gamma_src == gamma_snk and gamma_src in gammas:
                corr = g["corr"][:]
                vec.append(corr["re"])

    if not vec:
        raise ValueError(f"No GammaX/Y/Z vector channels found in {filename}")

    return np.array(vec).mean(axis=0)


# ===============================================================
# Cosh effective mass
# ===============================================================
def eff_mass_cosh(C):
    C = np.asarray(C)
    T = len(C)
    t = np.arange(1, T - 1)

    R = (C[t - 1] + C[t + 1]) / (2 * C[t])
    R = np.maximum(R, 1.0)  # numerical safety
    return t, np.arccosh(R)


# ===============================================================
# Reduced chi² for plateau
# ===============================================================
def plateau_chi2(y, yerr, avg):
    chi2 = np.sum(((y - avg) / yerr)**2)
    dof = len(y) - 1
    return chi2 / dof if dof > 0 else np.nan


# ===============================================================
# Bootstrap effective mass + plateau fit
# ===============================================================
def bootstrap_fit(corr_data, t_start, t_end, n_boot=1000):
    corr_data = np.asarray(corr_data)
    N, T = corr_data.shape

    # Use the first configuration to determine times
    t_vals, _ = eff_mass_cosh(corr_data[0])
    nt = len(t_vals)

    mask = (t_vals >= t_start) & (t_vals <= t_end)
    if not np.any(mask):
        raise ValueError(
            f"Plateau range {t_start}-{t_end} outside effective-mass range "
            f"{t_vals[0]}-{t_vals[-1]}"
        )

    m_eff_samples = np.zeros((n_boot, nt))
    m_plateau = np.zeros(n_boot)

    rng = np.random.default_rng()

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        C_mean = corr_data[idx].mean(axis=0)
        _, m_eff = eff_mass_cosh(C_mean)

        m_eff_samples[b] = m_eff
        m_plateau[b] = m_eff[mask].mean()

    m_eff_mean = m_eff_samples.mean(axis=0)
    m_eff_err = m_eff_samples.std(axis=0, ddof=1)

    mass = m_plateau.mean()
    mass_err = m_plateau.std(ddof=1)

    chi2 = plateau_chi2(m_eff_mean[mask], m_eff_err[mask], mass)

    return t_vals, m_eff_mean, m_eff_err, mass, mass_err, chi2


# ===============================================================
# Plotting
# ===============================================================
def plot_effmass(t, m, err, t0, t1, mfit, errfit, outfile, label, beta, mass, channel):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = None
    if label == "yes":
        data_label = rf"{channel}: $\beta={beta},\ am_0={mass}$"

    fit_label = rf"$am_{{\rm {channel}}} = {mfit:.5f}\,\pm\,{errfit:.5f}$"

    ax.errorbar(t, m, yerr=err, fmt="o", label=data_label, color="C4")
    ax.axvspan(t0, t1, alpha=0.2, color="C2", label="Plateau")
    ax.hlines(mfit, t0, t1, linestyles="--", color="C1", label=fit_label)

    ax.set_xlabel(r"$t/a$")
    ax.set_ylabel(r"$am_{\rm eff}$")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    #ax.set_ylim(0.35,1.7)

    if data_label or fit_label:
        ax.legend()

    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✓ Saved {channel} plot: {outfile}")


# ===============================================================
# Main
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--label", default="")
    parser.add_argument("--output_file1", required=True)
    parser.add_argument("--plot_file1", required=True)
    parser.add_argument("--plot_file2", required=True)
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--plateau_start_ps", type=float, required=True)
    parser.add_argument("--plateau_end_ps", type=float, required=True)
    parser.add_argument("--plateau_start_v", type=float, required=True)
    parser.add_argument("--plateau_end_v", type=float, required=True)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    ps_start = int(round(args.plateau_start_ps))
    ps_end = int(round(args.plateau_end_ps))
    v_start = int(round(args.plateau_start_v))
    v_end = int(round(args.plateau_end_v))

    # Collect correlator files
    files = sorted(glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5")))
    if not files:
        raise FileNotFoundError("No pt_ll.*.h5 files found.")

    ps_corrs = []
    v_corrs = []

    for f in files:
        ps_corrs.append(read_ps_corr(f))
        v_corrs.append(read_vec_corr(f))

    ps_corrs = np.array(ps_corrs)
    v_corrs = np.array(v_corrs)

    # Fit
    t_ps, m_ps, e_ps, M_PS, dM_PS, chi2_PS = bootstrap_fit(ps_corrs, ps_start, ps_end, args.n_boot)
    t_v, m_v, e_v, M_V, dM_V, chi2_V = bootstrap_fit(v_corrs, v_start, v_end, args.n_boot)

    # Write output
    with open(args.output_file1, "w") as f:
        f.write(
            "#am_ps\tam_ps_err\tam_v\tam_v_err\tchi2_ps\tchi2_v\t"
            "plateau_start_ps\tplateau_end_ps\tplateau_start_v\tplateau_end_v\n"
        )
        f.write(
            f"{M_PS:.6e}\t{dM_PS:.6e}\t"
            f"{M_V:.6e}\t{dM_V:.6e}\t"
            f"{chi2_PS:.6e}\t{chi2_V:.6e}\t"
            f"{ps_start}\t{ps_end}\t"
            f"{v_start}\t{v_end}\n"
        )

    print(f"✓ Saved spectrum file → {args.output_file1}")

    # Plots
    plot_effmass(t_ps, m_ps, e_ps, ps_start, ps_end, M_PS, dM_PS,
                 args.plot_file1, args.label, args.beta, args.mass, "PS")

    plot_effmass(t_v, m_v, e_v, v_start, v_end, M_V, dM_V,
                 args.plot_file2, args.label, args.beta, args.mass, "V")


if __name__ == "__main__":
    main()
