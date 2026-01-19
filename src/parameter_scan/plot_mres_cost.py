#!/usr/bin/env python3
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt

plt.style.use("tableau-colorblind10")

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Merged plot: m_res vs CG and time.")

parser.add_argument("--mres", nargs="+", required=True)
parser.add_argument("--hmc",  nargs="+", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--costs", required=True)

parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)

args = parser.parse_args()
show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)

# ------------------------------------------------------------
# Path parser
# ------------------------------------------------------------
pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
)

def parse_params(path):
    m = pattern.search(path)
    if m is None:
        raise ValueError(f"Cannot parse metadata from path: {path}")
    out = m.groupdict()
    return {k:(int(v) if k in ["Nt","Ns","Ls"] else float(v)) for k,v in out.items()}

# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def load_mres_fit(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        return float(arr[0]), float(arr[1])
    return float(arr[0,0]), float(arr[0,1])

def load_hmc_cost(path):
    arr = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    if arr.ndim != 0:
        arr = arr[0]
    return (
        float(arr["bcs"]), float(arr["bcs_err"]),
        float(arr["t_traj"]), float(arr["t_traj_err"])
    )

# ------------------------------------------------------------
# Merge MRES and HMC entries
# ------------------------------------------------------------
entries = []
hmc_lookup = {}

# Load all HMC files
for fp in args.hmc:
    p = parse_params(fp)
    hmc_lookup[tuple(sorted(p.items()))] = load_hmc_cost(fp)

# Match MRES with HMC
for fp in args.mres:
    p = parse_params(fp)
    key = tuple(sorted(p.items()))
    if key not in hmc_lookup:
        raise ValueError(f"No matching HMC file for: {fp}")

    mres, mres_err = load_mres_fit(fp)
    bcs, bcs_err, ttraj, ttraj_err = hmc_lookup[key]

    entries.append({
        **p,
        "mres": mres,
        "mres_err": mres_err,
        "bcs": bcs,
        "bcs_err": bcs_err,
        "t_traj": ttraj,
        "t_traj_err": ttraj_err
    })

if len(entries) == 0:
    print("WARNING: no entries found.")
    exit(0)

beta_value = entries[0]["beta"]
mass_value = entries[0]["mass"]

# ------------------------------------------------------------
# Style helpers
# ------------------------------------------------------------
LS_MARKERS = ["o","s","D","^","v","P","X","*"]
def marker_from_Ls(Ls):
    return LS_MARKERS[hash(Ls) % len(LS_MARKERS)]

def color_from_alpha(alpha):
    return "black" if np.isclose(alpha, 1.0) else "C1"

# ------------------------------------------------------------
# Label placement (directional)
# ------------------------------------------------------------
def place_label(ax, x, y, text, family, subplot):
    """
    family: "shamir" or "mobius"
    subplot: "cg" or "t"
    """

    dx = 0.03 * (x if x != 0 else 1)
    dy_up   = y * 1.06
    dy_down = y * 0.94

    # ----- placement rules -----
    if subplot == "cg":
        if family == "shamir":
            xt, yt = x - dx, dy_down      # bottom-left
        else:
            xt, yt = x + dx, dy_up        # top-right
    else:  # subplot == "t"
        if family == "shamir":
            xt, yt = x + dx, dy_up        # top-right
        else:
            xt, yt = x - dx, dy_down      # bottom-left

    ax.text(
        xt, yt, text, fontsize=7,
        ha="left" if xt > x else "right",
        va="bottom" if yt > y else "top"
    )

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------
fig, axs = plt.subplots(
    1, 2, figsize=(7,2.5),
    sharey=True, layout="constrained"
)

ax_cg, ax_t = axs

# Lists for connecting lines
sh_cg_x, sh_cg_y = [], []
mo_cg_x, mo_cg_y = [], []
sh_t_x,  sh_t_y  = [], []
mo_t_x,  mo_t_y  = [], []

# ------------------------------------------------------------
# Main plotting loop
# ------------------------------------------------------------
for e in entries:

    y, dy = e["mres"], e["mres_err"]
    Ls    = e["Ls"]
    alpha = e["alpha"]
    is_shamir = np.isclose(alpha, 1.0)
    family = "shamir" if is_shamir else "mobius"

    marker = marker_from_Ls(Ls)
    color  = color_from_alpha(alpha)

    # Two-line label for Möbius
    if is_shamir:
        label_text = rf"$L_s={Ls}$"
    else:
        label_text = rf"$L_s={Ls}$" + "\n" + rf"$\alpha={alpha}$"

    # ---------- CG subplot ----------
    x, dx = e["bcs"], e["bcs_err"]

    if is_shamir:
        sh_cg_x.append(x); sh_cg_y.append(y)
    else:
        mo_cg_x.append(x); mo_cg_y.append(y)

    ax_cg.errorbar(x, y, xerr=dx, yerr=dy,
                   fmt=marker, color=color, mec=color)

    place_label(ax_cg, x, y, label_text, family, "cg")
    ax_cg.set_xlim(52000, 148000)
    ax_t.set_ylim(0.000006, 0.01)

    # ---------- TIME subplot ----------
    x, dx = e["t_traj"], e["t_traj_err"]

    if is_shamir:
        sh_t_x.append(x); sh_t_y.append(y)
    else:
        mo_t_x.append(x); mo_t_y.append(y)

    ax_t.errorbar(x, y, xerr=dx, yerr=dy,
                  fmt=marker, color=color, mec=color)

    place_label(ax_t, x, y, label_text, family, "t")
    ax_t.set_xlim(50, 1500)
    ax_t.set_ylim(0.000006, 0.01)

# ------------------------------------------------------------
# Connect lines
# ------------------------------------------------------------
def connect(ax, xs, ys, style, color):
    if len(xs) > 1:
        xs_sorted, ys_sorted = zip(*sorted(zip(xs, ys)))
        ax.plot(xs_sorted, ys_sorted, style, color=color)

connect(ax_cg, sh_cg_x, sh_cg_y, "--", "black")
connect(ax_cg, mo_cg_x, mo_cg_y, ":",  "C1")

connect(ax_t, sh_t_x, sh_t_y, "--", "black")
connect(ax_t, mo_t_x, mo_t_y, ":",  "C1")

# ------------------------------------------------------------
# Axis labels, legend, title
# ------------------------------------------------------------
ax_cg.set_xlabel(r"CG applications")
ax_t.set_xlabel(r"Time [s]")

ax_cg.set_ylabel(r"$a m_{\rm res}$")
ax_cg.set_yscale("log")

fig.suptitle(rf"$\beta={beta_value}$, $am_0={mass_value}$")

if show_legend:
    handles = [
        plt.Line2D([], [], linestyle="--", color="black", label="Shamir, $\\alpha=1.0$"),
        plt.Line2D([], [], linestyle=":",  color="C1",    label="Möbius"),
    ]
    ax_cg.legend(handles=handles, loc="upper right")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
plt.savefig(args.costs, dpi=300)
plt.close()

print(f"Saved merged plot → {args.costs}")
