import h5py
import numpy as np
import argparse
import glob
import os
import re

# -----------------------------
# File readers
# -----------------------------
def read_mres_file(filename):
    """Read the real part of PJ5q dataset from mres HDF5 file (nested under wardIdentity)."""
    with h5py.File(filename, "r") as f:
        data = f["wardIdentity/PJ5q"][:]
        return data["re"]

def read_ptll_file(filename, n_elems=16):
    """Read the real part of the corr dataset for meson_1, ignoring attributes."""
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        if len(data) == n_elems:
            return data["re"]
    raise ValueError(f"corr dataset does not have {n_elems} elements in {filename}")

# -----------------------------
# Bootstrap ratio
# -----------------------------
def bootstrap_ratio(data1, data2, n_boot=1000):
    n_samples = data1.shape[0]
    ratios_boot = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n_samples, n_samples)
        mean1 = np.mean(data1[idx], axis=0)
        mean2 = np.mean(data2[idx], axis=0)
        ratios_boot.append(mean1 / mean2)
    ratios_boot = np.array(ratios_boot)
    ratio_mean = np.mean(data1, axis=0) / np.mean(data2, axis=0)
    ratio_err = np.std(ratios_boot, axis=0)
    return ratio_mean, ratio_err

# -----------------------------
# Madras & Sokal autocorrelation
# -----------------------------
def integrated_autocorrelation_time(x, c=5.0, M_max=50):
    """
    Compute the integrated autocorrelation time using the Madras-Sokal automatic windowing.
    
    Parameters:
        x : 1D array of measurements
        c : constant to define M (usually 4–10)
        M_max : maximum lag to consider for window search
    
    Returns:
        tau_int : integrated autocorrelation time
        tau_int_err : statistical error of tau_int
        M : selected window
    """
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return 0.5, 0.0, 0

    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.5, 0.0, 0

    # Biased autocorrelation function
    acf = np.correlate(x, x, mode='full')[n-1:] / (var * np.arange(n, 0, -1))

    # Candidate M values
    candidate_M = range(1, min(M_max, n-1)+1)

    # Compute tau_int(M) for each candidate M
    tau_int_M = np.array([0.5 + np.sum(acf[1:M+1]) for M in candidate_M])

    # Find smallest M satisfying M >= c * tau_int(M)
    valid = [M for M, tau in zip(candidate_M, tau_int_M) if M >= c * tau]
    if not valid:
        M_selected = candidate_M[-1]  # fallback to max candidate
    else:
        M_selected = min(valid)

    # Final tau_int and error
    tau_int = 0.5 + np.sum(acf[1:M_selected+1])
    tau_int_err = np.sqrt(2.0 * (M_selected + 1) / n) * tau_int

    return tau_int, tau_int_err


# -----------------------------
# Trajectory number extractor
# -----------------------------
def extract_trajectory_numbers(file_list, pattern=r".*\.([0-9]+)\.h5"):
    """Extract integer trajectory numbers from filenames."""
    traj_numbers = []
    for f in file_list:
        m = re.match(pattern, os.path.basename(f))
        if m:
            traj_numbers.append(int(m.group(1)))
        else:
            raise ValueError(f"Cannot extract trajectory number from filename: {f}")
    return sorted(traj_numbers)

# -----------------------------
# Main script
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input mesons directory")
    parser.add_argument("--output_file", required=True, help="Output ratio file")
    args = parser.parse_args()

    # -----------------------------
    # Read HDF5 files
    # -----------------------------
    mres_files = sorted(glob.glob(os.path.join(args.input_dir, "mres.*.h5")))
    if not mres_files:
        print("No mres HDF5 files found.")
        return
    print(f"Found {len(mres_files)} mres files. Reading...")
    mres_data = np.array([read_mres_file(f) for f in mres_files])

    ptll_files = sorted(glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5")))
    if not ptll_files:
        print("No pt_ll HDF5 files found.")
        return
    print(f"Found {len(ptll_files)} pt_ll files. Reading...")
    n_times = mres_data.shape[1]
    ptll_data = np.array([read_ptll_file(f, n_elems=n_times) for f in ptll_files])

    # -----------------------------
    # Bootstrap ratio
    # -----------------------------
    min_len = min(len(mres_data), len(ptll_data))
    ratio_mean, ratio_err = bootstrap_ratio(mres_data[:min_len], ptll_data[:min_len], n_boot=1000)

    # Compute ratio per trajectory (mean over t)
    ratio_per_traj = np.mean(mres_data[:min_len] / ptll_data[:min_len], axis=1)

    # -----------------------------
    # Autocorrelation time using Madras & Sokal
    # -----------------------------
    tau_int, tau_int_err = integrated_autocorrelation_time(ratio_per_traj)
    n_traj = min_len

    # -----------------------------
    # Trajectory spacing (integer)
    # -----------------------------
    traj_numbers = extract_trajectory_numbers(mres_files[:min_len])
    if len(traj_numbers) > 1:
        traj_spacing = int(np.round(np.mean(np.diff(traj_numbers))))
    else:
        traj_spacing = 0

    # -----------------------------
    # Save ratio.txt
    # -----------------------------
    with open(args.output_file, "w") as f:
        for idx, (r, e) in enumerate(zip(ratio_mean, ratio_err)):
            f.write(f"{idx}\t{r}\t{e}\n")
    print(f"Ratio with bootstrap error saved to {args.output_file}")

    # -----------------------------
    # Save autocorrelation info
    # -----------------------------
    autocorr_path = os.path.join(os.path.dirname(args.output_file), "autocorr.txt")
    with open(autocorr_path, "w") as f:
        f.write("#autocorrelation time\t#tau error\t#traj spacing\t#number of traj\n")
        f.write(f"{tau_int:.6f}\t{tau_int_err:.6f}\t{traj_spacing}\t{n_traj}\n")
    print(f"Autocorrelation info saved to {autocorr_path}")


if __name__ == "__main__":
    main()
