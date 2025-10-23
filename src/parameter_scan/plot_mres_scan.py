import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---------------------------
# Enable LaTeX for plotting
# ---------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})

# ---------------------------
# Parse arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("input", nargs="+", help="List of m_res.txt files")
parser.add_argument("--output_filename", required=True, help="Output plot filename")

# Mutually exclusive flags for scan type
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--a5", action="store_true", help="Scan over a5")
group.add_argument("--alpha", action="store_true", help="Scan over alpha")
group.add_argument("--M5", action="store_true", help="Scan over M5")
group.add_argument("--mpv", action="store_true", help="Scan over mpv")  # <-- new flag

args = parser.parse_args()

# ---------------------------
# Prepare lists
# ---------------------------
x_vals = []
values = []
errors = []
other_params = []

# Updated regex to include mpv
pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/mesons/m_res\.txt"
)

# ---------------------------
# Read data from each file
# ---------------------------
for filepath in args.input:
    m = pattern.search(filepath)
    if m is None:
        raise ValueError(f"Cannot parse parameters from path: {filepath}")

    p = m.groupdict()
    beta = float(p["beta"])
    mass = float(p["mass"])
    Ls = int(p["Ls"])

    # Load m_res.txt: columns = [index, value, error]
    data = np.loadtxt(filepath)
    mid_idx = len(data) // 2
    val, err = data[mid_idx, 1], data[mid_idx, 2]

    # Select x variable depending on scan type
    if args.a5:
        x_var = float(p["a5"])
        title = rf"$L_s={Ls},\,\alpha={p['alpha']},\,am_5={p['M5']}$"
        xlabel = r"$a_5/a$"
    elif args.alpha:
        x_var = float(p["alpha"])
        title = rf"$L_s={Ls},\,a_5/a={p['a5']},\,am_5={p['M5']}$"
        xlabel = r"$\alpha$"
    elif args.M5:
        x_var = float(p["M5"])
        title = rf"$L_s={Ls},\,\alpha={p['alpha']},\,a_5/a={p['a5']}$"
        xlabel = r"$am_5$"
    elif args.mpv:  # <-- new scan
        x_var = float(p["mpv"])
        title = rf"$L_s={Ls},\,\alpha={p['alpha']},\,a_5/a={p['a5']},\,am_5={p['M5']}$"
        xlabel = r"$m_{\mathrm{pv}}$"

    x_vals.append(x_var)
    values.append(val)
    errors.append(err)
    other_params.append((beta, mass))

# ---------------------------
# Sort by x variable (x-axis)
# ---------------------------
x_vals, values, errors, other_params = zip(*sorted(zip(x_vals, values, errors, other_params)))

beta, mass = other_params[0]
label = rf"$\beta={beta}$, $am_0={mass}$"

# ---------------------------
# Plot
# ---------------------------
plt.errorbar(
    x_vals, values, yerr=errors,
    fmt="o--", capsize=4, mfc='none', linewidth=0.5, label=label
)

plt.xlabel(xlabel)
plt.ylabel(r"$am_{\mathrm{res}}$")
plt.yscale("log")
plt.ylim(0.001, 0.3)
plt.title(title)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(args.output_filename, bbox_inches="tight")

print(f"Saved combined plot: {args.output_filename}")
