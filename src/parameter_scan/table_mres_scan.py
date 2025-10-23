import os
import argparse
import numpy as np
import re

def read_mres_file(filename):
    """Read m_res.txt and return value and error (middle row)."""
    data = np.loadtxt(filename)
    mid_idx = len(data) // 2
    value = data[mid_idx, 1]
    error = data[mid_idx, 2]
    return value, error

def format_value_error(value, error):
    """Format value and error as value(error), e.g., 0.03234(45) with 2 digits in error."""
    if error == 0:
        return f"{value:.5f}(00)"
    digits = -int(np.floor(np.log10(error))) + 1  # two significant digits
    val_str = f"{value:.{digits}f}"
    err_str = f"{int(round(error * 10 ** digits)):02d}"
    return f"{val_str}({err_str})"

def parse_params_from_path(filepath):
    """
    Extract scan parameters (Nt, Ns, Ls, B, M0, mpv, alpha, a5, M5)
    from m_res.txt path using regex.
    """
    pattern = re.compile(
        r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
        r"B(?P<B>[0-9\.]+)/M(?P<M0>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"   # <-- added mpv here
        r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/mesons/m_res\.txt"
    )
    m = pattern.search(filepath)
    if not m:
        raise ValueError(f"Cannot parse parameters from {filepath}")
    # convert to float or int depending on presence of '.'
    return {k: float(v) if '.' in v else int(v) for k, v in m.groupdict().items()}

def read_autocorr_file(folder):
    """Read autocorr.txt in the folder and return tau, tau_error, spacing, n_traj."""
    autocorr_path = os.path.join(folder, "autocorr.txt")
    if not os.path.exists(autocorr_path):
        return 0.0, 0.0, 0, 0
    with open(autocorr_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith("#"):
            continue
        parts = line.strip().split()
        tau = float(parts[0])
        tau_err = float(parts[1])
        spacing = int(parts[2])
        n_traj = int(parts[3])
        return tau, tau_err, spacing, n_traj
    return 0.0, 0.0, 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="List of m_res.txt files")
    parser.add_argument("--output_filename", required=True, help="Output LaTeX table filename")
    parser.add_argument("--alpha", action="store_true", help="Scan over alpha")
    parser.add_argument("--a5", action="store_true", help="Scan over a5")
    parser.add_argument("--M5", action="store_true", help="Scan over M5")
    parser.add_argument("--mpv", action="store_true", help="Scan over mpv")  # <-- new flag
    args = parser.parse_args()

    # Determine which parameter we’re scanning
    if args.alpha:
        scan_name = "alpha"
    elif args.a5:
        scan_name = "a5"
    elif args.M5:
        scan_name = "M5"
    elif args.mpv:
        scan_name = "mpv"  # <-- new option
    else:
        scan_name = "alpha"

    table_rows = []
    for filepath in sorted(args.input):
        value, error = read_mres_file(filepath)
        value_str = format_value_error(value, error)
        params = parse_params_from_path(filepath)
        scan_val = params.get(scan_name, 0.0)

        input_dir = os.path.dirname(filepath)
        tau, tau_err, spacing, n_traj = read_autocorr_file(input_dir)
        tau_str = format_value_error(tau, tau_err)

        table_rows.append((scan_val, value_str, tau_str, spacing, n_traj))

    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
    with open(args.output_filename, "w") as f:
        f.write("\\begin{tabular}{ccccc}\n")
        f.write(f"{scan_name} & mres & autocorrelation & traj spacing & n traj \\\\\n")
        f.write("\\hline\n")
        for row in table_rows:
            f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n")
        f.write("\\end{tabular}\n")

    print(f"LaTeX table saved to {args.output_filename}")

if __name__ == "__main__":
    main()
