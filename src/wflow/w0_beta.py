import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import re

plt.style.use("tableau-colorblind10")


# ============================================================
#  2-loop coefficients for Sp(2N)
# ============================================================
def compute_b0_b1(N, Nf):
    fac1 = 1.0 / (4 * np.pi)**2
    fac2 = 1.0 / (4 * np.pi)**4
    b0 = fac1 * ((11/3)*(N+1) - (2/3)*Nf)
    b1 = fac2 * ((34/3)*(N+1)**2 -
                 (2/3)*Nf*(5*(N+1) + 0.75*(2*N+1)))
    return b0, b1


# ============================================================
#  Theory
# ============================================================
def w0_over_a_theory(beta, N, lam, b0, b1):
    prefactor = b0 * (4 * N / beta)
    power = b1 / (2 * b0 * b0)
    return lam * prefactor**power * np.exp(beta / (8 * N * b0))


# ============================================================
#  Extract β and mass from paths
# ============================================================
beta_regex = re.compile(r"/B(\d+(\.\d+)?)/")
mass_regex = re.compile(r"/M(\d+(\.\d+)?)/")

def extract_beta(path):
    m = beta_regex.search(path)
    if not m:
        raise RuntimeError(f"Cannot extract beta from path: {path}")
    return float(m.group(1))

def extract_mass(path):
    m = mass_regex.search(path)
    if not m:
        raise RuntimeError(f"Cannot extract mass from path: {path}")
    return float(m.group(1))


# ============================================================
#                       MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot ratio vs beta, colored by mass.")
    parser.add_argument("input", nargs="+", help="summary.txt files")
    parser.add_argument("--Nc", required=True, type=int)
    parser.add_argument("--nF", required=True, type=int)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--output_filename", required=True)
    parser.add_argument("--plot_styles", default=None)
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    # Gauge group parameters
    N = args.Nc // 2
    Nf = args.nF
    lam = 1.0

    b0, b1 = compute_b0_b1(N, Nf)

    # Optional metadata
    if os.path.exists(args.metadata):
        with open(args.metadata) as f:
            metadata = yaml.safe_load(f)

    # -------------------------------------------------------------
    #   Read all data points
    # -------------------------------------------------------------
    betas = []
    masses = []
    w0_vals = []
    w0_errs = []

    for fname in args.input:
        beta = extract_beta(fname)
        mass = extract_mass(fname)

        arr = np.loadtxt(fname)
        w0_sq = arr[0]
        w0_sq_err = arr[1]

        w0 = np.sqrt(w0_sq)
        w0_err = 0.5 * w0_sq_err / w0

        betas.append(beta)
        masses.append(mass)
        w0_vals.append(w0)
        w0_errs.append(w0_err)

    betas = np.array(betas)
    masses = np.array(masses)
    w0_vals = np.array(w0_vals)
    w0_errs = np.array(w0_errs)

    # Sort by beta for nicer plotting
    idx = np.argsort(betas)
    betas, masses, w0_vals, w0_errs = betas[idx], masses[idx], w0_vals[idx], w0_errs[idx]

    # -------------------------------------------------------------
    #   Theory @ same β points
    # -------------------------------------------------------------
    w0_th = w0_over_a_theory(betas, N, lam, b0, b1)

    ratio = w0_vals / w0_th
    ratio_err = w0_errs / w0_th

    # -------------------------------------------------------------
    #   Color by mass (assume 2 distinct masses)
    # -------------------------------------------------------------
    unique_masses = np.unique(masses)
    if len(unique_masses) == 1:
        colors = {unique_masses[0]: "tab:blue"}
    elif len(unique_masses) == 2:
        colors = {
            unique_masses[0]: "tab:blue",
            unique_masses[1]: "tab:orange",
        }
    else:
        # fallback: assign many colors
        colormap = plt.cm.get_cmap("tab10", len(unique_masses))
        colors = {m: colormap(i) for i, m in enumerate(unique_masses)}

    # -------------------------------------------------------------
    #                       PLOT
    # -------------------------------------------------------------
    plt.figure(figsize=(3.5, 2), layout="constrained")

    plt.figure(figsize=(3.5, 2), layout="constrained")

    unique_masses = np.unique(masses)

    for m in unique_masses:
        mask = masses == m
        beta_m = betas[mask]
        R_m = w0_vals[mask]
        dR_m = w0_errs[mask]

        plt.errorbar(beta_m,R_m,yerr=dR_m,fmt="o",color=colors[m],label=rf"$am_0$ = {m}")

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$w_0 / a$")

    plt.legend()
    
    plt.savefig(args.output_filename)
    plt.close()


if __name__ == "__main__":
    main()
