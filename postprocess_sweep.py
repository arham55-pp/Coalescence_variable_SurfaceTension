"""
# Calculate x_0 global maxima for coalescence simulations.
# 
# Usage:
#     python postprocess_sweep.py [sweep_results_directory] [options]
#     python postprocess_sweep.py sweep_beta_Pe_results  # default
#     python postprocess_sweep.py sweep_beta_Pe_results --beta 0.1 0.2 --pe 10 100 1000
#     python postprocess_sweep.py --beta-range 0.05 0.2 --pe-range 1 1000
# 
# Options:
#     --beta VALS               Select specific beta values (space-separated)
#     --pe VALS                 Select specific Pe values (space-separated)
#     --beta-range MIN MAX      Select beta values in range [MIN, MAX]
#     --pe-range MIN MAX        Select Pe values in range [MIN, MAX]
# 
# Output:
#     <sweep_results_directory>_x0_global_maxima.csv
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

from postprocess_functions import find_neck


def extract_params_from_folder(folder_name):
    """Extract beta and Pe from folder names like 'beta_0.1_Pe_10'."""
    match = re.search(r"beta_([\d.]+)_Pe_([\d.]+)", folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def filter_folders_by_parameters(folders, beta_values=None, pe_values=None, 
                                  beta_min=None, beta_max=None, 
                                  pe_min=None, pe_max=None):
    """
    Filter folders based on parameter criteria.
    
    Args:
        folders: list of folder names
        beta_values: list of specific beta values to include
        pe_values: list of specific Pe values to include
        beta_min: minimum beta value (inclusive)
        beta_max: maximum beta value (inclusive)
        pe_min: minimum Pe value (inclusive)
        pe_max: maximum Pe value (inclusive)
    
    Returns:
        filtered list of folders
    """
    filtered = []
    for folder in folders:
        beta, pe = extract_params_from_folder(folder)
        if beta is None or pe is None:
            continue
        
        # Check specific values
        if beta_values is not None and beta not in beta_values:
            continue
        if pe_values is not None and pe not in pe_values:
            continue
        
        # Check ranges
        if beta_min is not None and beta < beta_min:
            continue
        if beta_max is not None and beta > beta_max:
            continue
        if pe_min is not None and pe < pe_min:
            continue
        if pe_max is not None and pe > pe_max:
            continue
        
        filtered.append(folder)
    
    return filtered


def load_time_series(base_folder):
    """Load t, h_0(t), and x_0(t) from one simulation folder."""
    output_dir = os.path.join(base_folder, "domain")

    if not os.path.exists(output_dir):
        return None, None, None

    files = sorted([f for f in os.listdir(output_dir) if f.endswith(".txt")])
    if len(files) == 0:
        return None, None, None

    time_data = []
    h0_data = []
    x0_data = []

    for filename in files:
        try:
            with open(os.path.join(output_dir, filename)) as file:
                header = file.readline()
                time = float(header.split("@time=")[-1])
                data = np.loadtxt(file)
                x = data[:, 0]
                h = data[:, 1]

                x0, h0 = find_neck(x, h)
                time_data.append(time)
                h0_data.append(h0)
                x0_data.append(x0)
        except Exception as error:
            print(f"Warning: skipped {filename} in {base_folder}: {error}")

    if len(time_data) == 0:
        return None, None, None

    time_data = np.array(time_data)
    h0_data = np.array(h0_data)
    x0_data = np.array(x0_data)

    valid_mask = time_data > 0
    time_data = time_data[valid_mask]
    h0_data = h0_data[valid_mask]
    x0_data = x0_data[valid_mask]

    valid_x0_mask = np.abs(x0_data) > 1e-7
    time_data = time_data[valid_x0_mask]
    h0_data = h0_data[valid_x0_mask]
    x0_data = x0_data[valid_x0_mask]

    if len(time_data) == 0:
        return None, None, None

    sort_idx = np.argsort(time_data)
    return time_data[sort_idx], h0_data[sort_idx], x0_data[sort_idx]


def calculate_x0_global_maximum(time_data, x0_data):
    """Calculate the global maximum of x_0(t)."""
    if time_data is None or len(time_data) == 0:
        return None, None, 0

    max_idx = np.argmax(x0_data)
    return time_data[max_idx], x0_data[max_idx], len(time_data)


# ============================================================================
# COMMENTED OUT: Log-log fitting and plotting functionality
# ============================================================================

"""
def linear_fit_r2(x, y):
    Fit y = intercept + slope*x and return slope, intercept, and R^2.
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = intercept + slope * x
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2
'''
'''
def collapse_repeated_values(time_data, quantity_data):
    
    Replace repeated quantity values by one point at the mean time.

    This is useful for x_0 because the neck location can stay fixed on the same
    grid point for many timesteps, which creates horizontal stacks in log-log
    space and biases the fit.
    
    grouped = {}
    for time, quantity in zip(time_data, quantity_data):
        grouped.setdefault(quantity, []).append(time)

    collapsed_time = []
    collapsed_quantity = []
    for quantity, times in grouped.items():
        collapsed_time.append(np.mean(times))
        collapsed_quantity.append(quantity)

    collapsed_time = np.array(collapsed_time)
    collapsed_quantity = np.array(collapsed_quantity)
    sort_idx = np.argsort(collapsed_time)
    return collapsed_time[sort_idx], collapsed_quantity[sort_idx]


def trim_to_initial_growth_branch(time_data, quantity_data):
    Trim non-growing early points before fitting the x_0 growth branch.
    if len(quantity_data) < MIN_FIT_POINTS:
        return time_data, quantity_data

    diffs = np.diff(quantity_data)
    increasing = diffs > 0
    best_start = 0
    best_length = 0
    current_start = None

    for idx, is_increasing in enumerate(increasing):
        if is_increasing and current_start is None:
            current_start = idx
        elif not is_increasing and current_start is not None:
            length = idx - current_start + 1
            if length > best_length:
                best_start = current_start
                best_length = length
            current_start = None

    if current_start is not None:
        length = len(quantity_data) - current_start
        if length > best_length:
            best_start = current_start
            best_length = length

    if best_length >= MIN_FIT_POINTS:
        return time_data[best_start:], quantity_data[best_start:]

    return time_data, quantity_data


def fit_initial_linear_region(
        log_time, log_quantity, threshold=R2_THRESHOLD,
        min_points=MIN_FIT_POINTS, min_log_span=MIN_LOG_TIME_SPAN):
    
    Fit from the start and stop when the initial straight segment degrades.

    R^2 alone can stay above 0.95 even after the curve bends, so we also stop
    when R^2 drops noticeably from its best early value.
    
    if len(log_time) < min_points:
        return None

    last_valid_fit = None
    first_fit = None
    best_r2 = None

    for end_idx in range(min_points, len(log_time) + 1):
        x_fit = log_time[:end_idx]
        y_fit = log_quantity[:end_idx]
        slope, intercept, r2 = linear_fit_r2(x_fit, y_fit)
        y_line = intercept + slope * x_fit
        residuals = y_fit - y_line
        curvature = np.max(np.abs(residuals))
        fit = {
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "fit_points": end_idx,
            "start_idx": 0,
            "end_idx": end_idx,
            "log_time_start": x_fit[0],
            "log_time_end": x_fit[-1],
            "curvature": curvature,
        }

        if first_fit is None:
            first_fit = fit

        if best_r2 is None or r2 > best_r2:
            best_r2 = r2

        if r2 < threshold:
            return last_valid_fit if last_valid_fit is not None else first_fit

        if last_valid_fit is not None and best_r2 - r2 > R2_DROP_TOLERANCE:
            return last_valid_fit

        if x_fit[-1] - x_fit[0] >= min_log_span:
            last_valid_fit = fit

    return last_valid_fit if last_valid_fit is not None else first_fit


def prepare_loglog_fit(time_data, quantity_data, collapse_repeats=False, trim_growth_branch=False):
    Prepare positive log-log data and fit it with the R^2 threshold rule.
    valid_mask = (time_data > 0) & (quantity_data > 0)
    time_valid = time_data[valid_mask]
    quantity_valid = quantity_data[valid_mask]

    if collapse_repeats:
        time_valid, quantity_valid = collapse_repeated_values(time_valid, quantity_valid)

    if trim_growth_branch:
        time_valid, quantity_valid = trim_to_initial_growth_branch(time_valid, quantity_valid)

    if len(time_valid) < MIN_FIT_POINTS:
        return None, None, None

    log_time = np.log(time_valid)
    log_quantity = np.log(quantity_valid)
    fit = fit_initial_linear_region(log_time, log_quantity)
    return log_time, log_quantity, fit


def plot_fit_on_axis(ax, log_time, log_quantity, fit, color, label):
    Plot log-log data and the fitted segment on an axis.
    if log_time is None or log_quantity is None:
        ax.text(0.5, 0.5, "No valid log-log data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=plt_settings["LegendFont"])
        return

    ax.plot(log_time, log_quantity, "o", color=color, markersize=3, alpha=0.28)

    if fit is None:
        ax.text(0.5, 0.5, "No valid fit",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=plt_settings["LegendFont"])
        return

    fit_slice = slice(fit["start_idx"], fit["end_idx"])
    ax.plot(log_time[fit_slice], log_quantity[fit_slice], "o", color=color, markersize=4, alpha=0.85)
    fit_x = np.linspace(fit["log_time_start"], fit["log_time_end"], 100)
    fit_y = fit["intercept"] + fit["slope"] * fit_x
    ax.plot(fit_x, fit_y, color=color, linewidth=2.5, label=label)


def save_case_fit_plot(plot_dir, folder, beta, Pe, log_time_h0, log_h0, h0_fit, log_time_x0, log_abs_x0, x0_fit):
    Save one diagnostic Plot 3b-style figure for a single parameter case.
    fig, (ax_h0, ax_x0) = plt.subplots(1, 2, figsize=(16, 7))

    h0_label = "h_0 fit"
    if h0_fit is not None:
        h0_label = f"h_0 fit: m={h0_fit['slope']:.4f}, R^2={h0_fit['r2']:.4f}, n={h0_fit['fit_points']}"
    x0_label = "|x_0| fit"
    if x0_fit is not None:
        x0_label = f"|x_0| fit: m={x0_fit['slope']:.4f}, R^2={x0_fit['r2']:.4f}, n={x0_fit['fit_points']}"

    plot_fit_on_axis(ax_h0, log_time_h0, log_h0, h0_fit, "tab:blue", h0_label)
    plot_fit_on_axis(ax_x0, log_time_x0, log_abs_x0, x0_fit, "tab:green", x0_label)

    style_axis(ax_h0, xlabel="log(Time t)", ylabel="log(Neck height h_0)",
               title=f"h_0 log-log fit: beta={beta}, Pe={Pe}")
    style_axis(ax_x0, xlabel="log(Time t)", ylabel="log(Neck position |x_0|)",
               title=f"|x_0| log-log fit: beta={beta}, Pe={Pe}")

    if h0_fit is not None:
        ax_h0.legend(fontsize=plt_settings["LegendFont"], frameon=True, loc="best")
    if x0_fit is not None:
        ax_x0.legend(fontsize=plt_settings["LegendFont"], frameon=True, loc="best")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{folder}_loglog_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw_timeseries(ax_h_lin, ax_x_lin, ax_h_log, ax_x_log, time_data, h0_data, x0_data, color, label):
    Add one parameter case to the combined linear and log-log time-series plot.
    ax_h_lin.plot(time_data, h0_data, color=color, linewidth=2, alpha=0.8, label=label)
    ax_x_lin.plot(time_data, x0_data, color=color, linewidth=2, alpha=0.8, label=label)

    valid_h_mask = (time_data > 0) & (h0_data > 0)
    if np.any(valid_h_mask):
        ax_h_log.plot(
            np.log(time_data[valid_h_mask]),
            np.log(h0_data[valid_h_mask]),
            color=color,
            linewidth=2,
            alpha=0.8,
            label=label,
        )

    abs_x0 = np.abs(x0_data)
    valid_x_mask = (time_data > 0) & (abs_x0 > 0)
    if np.any(valid_x_mask):
        ax_x_log.plot(
            np.log(time_data[valid_x_mask]),
            np.log(abs_x0[valid_x_mask]),
            color=color,
            linewidth=2,
            alpha=0.8,
            label=label,
        )
"""


def main():
    parser = argparse.ArgumentParser(
        description="Calculate x_0 global maxima for coalescence simulations."
    )
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        default="sweep_beta_Pe_results",
        help="Directory containing sweep results (default: sweep_beta_Pe_results)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs="+",
        help="Select specific beta values (space-separated)"
    )
    parser.add_argument(
        "--pe",
        type=float,
        nargs="+",
        help="Select specific Pe values (space-separated)"
    )
    parser.add_argument(
        "--beta-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Select beta values in range [MIN, MAX]"
    )
    parser.add_argument(
        "--pe-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Select Pe values in range [MIN, MAX]"
    )
    
    args = parser.parse_args()
    sweep_dir = args.sweep_dir

    if not os.path.exists(sweep_dir):
        print(f"Error: directory not found: {sweep_dir}")
        sys.exit(1)

    folders = sorted([
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d)) and d.startswith("beta_")
    ])

    # Apply parameter filters if specified
    if args.beta or args.pe or args.beta_range or args.pe_range:
        beta_min, beta_max = None, None
        pe_min, pe_max = None, None
        
        if args.beta_range:
            beta_min, beta_max = args.beta_range
        if args.pe_range:
            pe_min, pe_max = args.pe_range
        
        folders = filter_folders_by_parameters(
            folders,
            beta_values=args.beta,
            pe_values=args.pe,
            beta_min=beta_min,
            beta_max=beta_max,
            pe_min=pe_min,
            pe_max=pe_max
        )
        
        print(f"After filtering: {len(folders)} parameter folders selected")
    else:
        print(f"Found {len(folders)} parameter folders")

    maxima_file = f"{sweep_dir}_x0_global_maxima.csv"
    # slopes_file = f"{sweep_dir}_loglog_slopes.csv"
    # plot_file = f"{sweep_dir}_loglog_fits.png"
    # all_timeseries_plot_file = f"{sweep_dir}_all_timeseries_combined.png"
    # case_plot_dir = f"{sweep_dir}_loglog_fit_plots"
    # os.makedirs(case_plot_dir, exist_ok=True)

    # fig, (ax_h0, ax_x0) = plt.subplots(1, 2, figsize=(16, 7))
    # fig_all, axes_all = plt.subplots(2, 2, figsize=(18, 14))
    # ax_h_lin, ax_x_lin = axes_all[0]
    # ax_h_log, ax_x_log = axes_all[1]
    # colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(folders))))

    with open(maxima_file, "w", newline="") as maxima_csv:
        maxima_writer = csv.DictWriter(
            maxima_csv,
            fieldnames=["folder", "beta", "Pe", "x0_max_time", "x0_max_value", "n_points"],
        )
        maxima_writer.writeheader()
        
        # slopes_writer = csv.DictWriter(
        #     slopes_csv,
        #     fieldnames=[
        #         "folder", "beta", "Pe",
        #         "h0_slope", "h0_intercept", "h0_r2", "h0_fit_points", "h0_log_time_start", "h0_log_time_end", "h0_curvature",
        #         "x0_slope", "x0_intercept", "x0_r2", "x0_fit_points", "x0_log_time_start", "x0_log_time_end", "x0_curvature",
        #     ],
        # )
        # slopes_writer.writeheader()

        valid_results = 0
        # colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(folders))))
        for idx, folder in enumerate(folders):
            beta, Pe = extract_params_from_folder(folder)
            if beta is None:
                print(f"Skipped folder with unrecognized parameter format: {folder}")
                continue

            base_path = os.path.join(sweep_dir, folder)
            time_data, h0_data, x0_data = load_time_series(base_path)

            if time_data is None:
                print(f"Failed to load beta={beta}, Pe={Pe}")
                continue

            x0_max_time, x0_max_value, n_points = calculate_x0_global_maximum(time_data, x0_data)
            maxima_writer.writerow({
                "folder": folder,
                "beta": beta,
                "Pe": Pe,
                "x0_max_time": x0_max_time,
                "x0_max_value": x0_max_value,
                "n_points": n_points,
            })
            maxima_csv.flush()

            # log_time_h0, log_h0, h0_fit = prepare_loglog_fit(time_data, h0_data)
            # log_time_x0, log_abs_x0, x0_fit = prepare_loglog_fit(
            #     time_data, np.abs(x0_data), collapse_repeats=True, trim_growth_branch=True
            # )

            # slopes_writer.writerow({
            #     "folder": folder,
            #     "beta": beta,
            #     "Pe": Pe,
            #     "h0_slope": None if h0_fit is None else h0_fit["slope"],
            #     "h0_intercept": None if h0_fit is None else h0_fit["intercept"],
            #     "h0_r2": None if h0_fit is None else h0_fit["r2"],
            #     "h0_fit_points": None if h0_fit is None else h0_fit["fit_points"],
            #     "h0_log_time_start": None if h0_fit is None else h0_fit["log_time_start"],
            #     "h0_log_time_end": None if h0_fit is None else h0_fit["log_time_end"],
            #     "h0_curvature": None if h0_fit is None else h0_fit["curvature"],
            #     "x0_slope": None if x0_fit is None else x0_fit["slope"],
            #     "x0_intercept": None if x0_fit is None else x0_fit["intercept"],
            #     "x0_r2": None if x0_fit is None else x0_fit["r2"],
            #     "x0_fit_points": None if x0_fit is None else x0_fit["fit_points"],
            #     "x0_log_time_start": None if x0_fit is None else x0_fit["log_time_start"],
            #     "x0_log_time_end": None if x0_fit is None else x0_fit["log_time_end"],
            #     "x0_curvature": None if x0_fit is None else x0_fit["curvature"],
            # })
            # slopes_csv.flush()

            # color = colors[idx]
            # label = f"beta={beta}, Pe={Pe}"

            # plot_raw_timeseries(
            #     ax_h_lin, ax_x_lin, ax_h_log, ax_x_log,
            #     time_data, h0_data, x0_data, color, label
            # )

            # if h0_fit is not None:
            #     plot_fit_on_axis(
            #         ax_h0, log_time_h0, log_h0, h0_fit, color,
            #         f"{label}, m={h0_fit['slope']:.3f}"
            #     )

            # if x0_fit is not None:
            #     plot_fit_on_axis(
            #         ax_x0, log_time_x0, log_abs_x0, x0_fit, color,
            #         f"{label}, m={x0_fit['slope']:.3f}"
            #     )

            # save_case_fit_plot(
            #     case_plot_dir, folder, beta, Pe,
            #     log_time_h0, log_h0, h0_fit,
            #     log_time_x0, log_abs_x0, x0_fit,
            # )

            valid_results += 1

            # h0_slope_text = "nan" if h0_fit is None else f"{h0_fit['slope']:.6f}"
            # x0_slope_text = "nan" if x0_fit is None else f"{x0_fit['slope']:.6f}"
            print(
                f"beta={beta}, Pe={Pe}: max x_0={x0_max_value:.6f} at t={x0_max_time:.6f}"
            )

    if valid_results == 0:
        print("No valid maxima found")
        sys.exit(1)

    # style_axis(ax_h0, xlabel="log(Time t)", ylabel="log(Neck height h_0)",
    #            title=f"h_0 log-log fits, R^2 >= {R2_THRESHOLD}")
    # style_axis(ax_x0, xlabel="log(Time t)", ylabel="log(Neck position |x_0|)",
    #            title=f"|x_0| log-log fits, R^2 >= {R2_THRESHOLD}")

    # ax_h0.legend(fontsize=max(8, plt_settings["LegendFont"] - 8), frameon=True, loc="best")
    # ax_x0.legend(fontsize=max(8, plt_settings["LegendFont"] - 8), frameon=True, loc="best")

    # plt.tight_layout()
    # plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    # plt.close(fig)

    # style_axis(ax_h_lin, xlabel="Time t", ylabel="Neck height h_0",
    #            title="All Parameters: h_0(t)")
    # style_axis(ax_x_lin, xlabel="Time t", ylabel="Neck position x_0",
    #            title="All Parameters: x_0(t)")
    # style_axis(ax_h_log, xlabel="log(Time t)", ylabel="log(Neck height h_0)",
    #            title="All Parameters: log(h_0) vs log(t)")
    # style_axis(ax_x_log, xlabel="log(Time t)", ylabel="log(Neck position |x_0|)",
    #            title="All Parameters: log(|x_0|) vs log(t)")

    # handles, labels = ax_h_lin.get_legend_handles_labels()
    # if handles:
    #     fig_all.legend(
    #         handles, labels, loc="upper center", ncol=4,
    #         fontsize=max(8, plt_settings["LegendFont"] - 8), frameon=True
    #     )
    # fig_all.tight_layout(rect=[0, 0, 1, 0.94])
    # fig_all.savefig(all_timeseries_plot_file, dpi=150, bbox_inches="tight")
    # plt.close(fig_all)

    print(f"\nSaved x_0 global maxima to: {maxima_file}")
    # print(f"Saved log-log slopes to: {slopes_file}")
    # print(f"Saved log-log fit plot to: {plot_file}")
    # print(f"Saved combined all-parameter time-series plot to: {all_timeseries_plot_file}")
    # print(f"Saved per-case log-log fit plots to: {case_plot_dir}")


if __name__ == "__main__":
    main()
