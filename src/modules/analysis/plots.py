from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib import colormaps
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from itertools import cycle
from modules.analysis.ranking import get_learners, get_phcm
import matplotlib.ticker as ticker
import modules.common.utils as util
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

mpl.use('Agg')


def plot_scatter_cc(inv_res:dict, ds_md:dict, archs:list, cc_m:str, outfile:str) -> None:
    """
    #TODO: FILL
    """

    archs = get_learners(archs)
    
    y_unit = cc_m.split("_")[-1].lower()
    y_title  = " ".join(cc_m.split("_")[:-1]).title()

    fig_width = 10
    fig_height = 6
    plt.figure(figsize=(fig_width, fig_height))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of cells (rows × features)")
    plt.ylabel(y_title + f" ({y_unit})")
    plt.title(y_title + " vs. Problem Size")

    marker_cycle = cycle(['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>'])
    linestyle_cycle = cycle(['-', '--', '-.', ':'])

    for arch in archs:
        mk = next(marker_cycle)
        ls = next(linestyle_cycle)

        x_vals = []
        y_vals = []
        for ds_name, ds_dict in inv_res.items():
            if "fit" in cc_m:
                n_inst = round(ds_md[ds_name]["n_rows"] * 4/5) #TODO: Pass down from main
            else:
                n_inst = round(ds_md[ds_name]["n_rows"] * 1/5)
            n_features = ds_md[ds_name]["n_columns"] - 1
            n_cells = n_inst * n_features

            x_run = []
            y_run = []
            for run in ds_dict[arch]:
                x_run.append(n_cells)
                y_run.append(run[cc_m])
            x_vals.append(np.mean(x_run))
            y_vals.append(np.mean(y_run))

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        # Plot transparent base points
        scatter = plt.scatter(x_vals, y_vals, alpha=0.3, label=None)

        # Overlay less transparent points in same color
        plt.scatter(x_vals, y_vals, alpha=0.3, color=scatter.get_facecolor()[0], label=None,  marker=mk)

        # Fit polynomial in log-log space
        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)
        degree = min(2, len(log_x) - 1)  # degree 2 unless very few points
        coeffs = np.polyfit(log_x, log_y, deg=degree)
        poly = np.poly1d(coeffs)
        x_smooth = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 200)
        y_smooth = 10 ** poly(np.log10(x_smooth))

        # Plot polynomial curve
        plt.plot(x_smooth, y_smooth, color=scatter.get_facecolor()[0],alpha=1, linewidth=2, label=arch, linestyle=ls)

    # Move legend to the right of the plot
    plt.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Make room on right for legend

    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_box_cu(res: dict, archs: list, cu_m: str, output_path: str) -> None:
    """
    Plots a two-dimensional grid of box-plots where each row is a learner
    and each column is a post-hoc calibration method.  Each cell shows the
    distribution (across all datasets and runs) of the measure `cu_m`.

    Exports a PNG figure to: output_path + cu_m + ".png"

    Args:
        res (dict): Nested results { arch_key: { ds_name: [run_dicts,…] , … }, … }
        archs (list): List of arch keys like "lrn.phc" or "lrn"
        cu_m (str):   The measure name to plot (must exist in each run dict)
        output_path (str):  Base folder for output
    """
    lrns = get_learners(archs)   
    phcms = get_phcm(archs)      
    phcms.insert(0, "none")
    n_rows = len(lrns)
    n_cols = len(phcms)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4*n_cols, 3*n_rows),
        squeeze=False,
        sharey=True
    )
    pct_formatter = FuncFormatter(lambda y, _: f"{y:.0f}%")
    
    for i, lrn in enumerate(lrns):
        for j, phc in enumerate(phcms):
            arch_key = f"{lrn}.{phc}" if phc and phc != "none" else lrn

            values = []
            ds_dict = res[arch_key]
            for runs in ds_dict.values():
                for run in runs:
                    values.append(run[cu_m])

            ax = axes[i][j]
            ax.boxplot(values, vert=True)
            ax.set_xticks([])
            ax.yaxis.set_major_formatter(pct_formatter)
            
            ax.set_title(phc if phc and phc != "none" else "none")
            if j == 0:
                ax.set_ylabel(lrn)

    fig.suptitle(f"Distribution of {cu_m}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_file = output_path + f"{cu_m}.png"
    fig.savefig(out_file)
    plt.close(fig)


def plot_rankings(data: dict, outfile: str, title:str, x_label:str, top_n: int = 5, bottom_n: int = 5) -> None:
    """
    Export a horizontal bar chart of learner rankings and highlight top and bottom performers.

    Args:
        data (dict): Mapping from learner names to ranking metric (lower is better).
        outfile (str): File path to save the figure
        title (str)
        x_label (str)
        top_n (int, optional): Number of top performers to highlight in orange. Defaults to 5.
        bottom_n (int, optional): Number of bottom performers to highlight in green. Defaults to 5.
    """
    # Prepare and sort data
    names = list(data.keys())
    values = np.array([data[n] for n in names], dtype=float)
    order = np.argsort(values)
    sorted_names = [names[i] for i in order]
    sorted_values = values[order]

    # Colors: default, top, bottom
    default_color = '#1f77b4'
    top_color = '#ff7f0e'
    bottom_color = '#2ca02c'
    colors = [default_color] * len(sorted_values)
    for i in range(min(top_n, len(colors))):
        colors[i] = top_color
    for i in range(1, min(bottom_n, len(colors)) + 1):
        colors[-i] = bottom_color

    # Dynamic figure size
    n = len(sorted_names)
    height = max(6, 0.3 * n)
    fig, ax = plt.subplots(figsize=(10, height))

    # Plot bars
    bars = ax.barh(sorted_names, sorted_values, color=colors)
    ax.invert_yaxis()  # best at top

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_title(title)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.01 * sorted_values.max()),
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            va='center', fontsize=8
        )

    # Legend
    legend_elements = [
        Patch(facecolor=top_color, label=f'Top {top_n}'),
        Patch(facecolor=default_color, label='Middle performers'),
        Patch(facecolor=bottom_color, label=f'Bottom {bottom_n}')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Layout adjustments
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dual_rankings(data: dict, archs: list, outfile: str, meas_1: str, meas_2: str, monkey_fix = False) -> None:
    # Extract unique base models and PHCMs
    base_models = sorted({arch.split('.')[0] for arch in archs})
    phcms = sorted({'none' if '.' not in arch else arch.split('.', 1)[1] for arch in archs})
    
    # Create mappings for visual attributes
    base_colors = colormaps['tab20'].resampled(len(base_models))
    base_color_map = {model: base_colors(i) for i, model in enumerate(base_models)}
    
    all_markers = [m for m in MarkerStyle.markers if m not in [' ', '', None, 'None']]
    phcm_marker_map = {phcm: all_markers[i % len(all_markers)] for i, phcm in enumerate(phcms)}
    
    # Setup figure with constrained layout
    fig, ax = plt.subplots(figsize=(14, 10), layout='constrained')
    
    # Plot each architecture
    for arch in archs:
        # Parse base model and PHCM
        parts = arch.split('.', 1)
        base = parts[0]
        phcm = 'none' if len(parts) == 1 else parts[1]
        
        # Get data values
        x_val = data.get(meas_1, {}).get(arch)
        y_val = data.get(meas_2, {}).get(arch)
        
        if x_val is None or y_val is None:
            continue
            
        # Convert numpy types to float
        x_val = float(x_val) if isinstance(x_val, np.generic) else x_val
        y_val = float(y_val) if isinstance(y_val, np.generic) else y_val
        
        # Create composite label
        label = f"{base}\n({phcm})" if phcm != 'none' else base
        
        ax.scatter(
            x_val, y_val,
            color=base_color_map[base],
            marker=phcm_marker_map[phcm],
            s=1000 if monkey_fix else 120,
            edgecolor='white',
            linewidth=1.2,
            alpha=0.9,
            label=label
        )
    
    # Configure axes
    ax.set_xlabel(f"Rank: {meas_1.replace('_', ' ').title()}", fontsize=12)
    ax.set_ylabel(f"Rank: {meas_2.replace('_', ' ').title()}", fontsize=12)
    if monkey_fix:
        title_str = f"Expected performance rank across models, folds, and datasets:\nΔ {meas_1.replace('_', ' ').title()} vs Δ {meas_2.replace('_', ' ').title()}"
    else:
        title_str = f"Expected performance rank across folds and datasets:\n{meas_1.replace('_', ' ').title()} vs {meas_2.replace('_', ' ').title()}"
    ax.set_title(title_str,
                 fontsize=14, pad=15)
    
    # Use integer ticks for ranks
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=len(archs)+5,steps=[1], integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=len(archs)+5,steps=[1], integer=True))
    
    # Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Create dual legend system
    legend_elements = []
    
    # Base model legend (color-coded)
    for model in base_models:
        legend_elements.append(Line2D(
            [0], [0], 
            marker='s', 
            color='w', 
            markerfacecolor=base_color_map[model],
            markersize=10,
            label=model
        ))
    if not monkey_fix:
        # PHCM legend (shape-coded)
        for phcm in phcms:
            legend_elements.append(Line2D(
                [0], [0], 
                marker=phcm_marker_map[phcm], 
                color='w', 
                markerfacecolor='gray',
                markersize=10,
                label=phcm
            ))
        
    # Position combined legend outside
    ax.legend(
        handles=legend_elements,
        title="PHCMs"if monkey_fix else f"Base Models (color)\nPHCMs (shape)",
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.9
    )
    
    # Save high-quality image
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ranking_hist(data:dict, arch:str) -> None:
    """
    Plots a horizontal bar chart of ranking probabilities for a given architecture.
    
    Args:
        data (dict): dict mapping architecture names to 1D numpy arrays of probabilities 
        architecture (str): str key of the architecture's name
    """
    probs = data[arch]
    ranks = np.arange(1, len(probs) + 1)  # 1st place = 1, 2nd place = 2, ...

    fig, ax = plt.subplots()
    ax.barh(ranks, probs)
    
    # Invert y-axis so rank 1 appears at the top
    ax.invert_yaxis()
    
    # Label ticks
    ax.set_yticks(ranks)
    ax.set_yticklabels([f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th" for i in ranks])
    
    ax.set_xlabel("Probability")
    ax.set_ylabel("Rank Position")
    ax.set_title(f"Ranking Distribution for {arch}")
    plt.tight_layout()
    plt.savefig('ranking_lr.png', dpi=150, bbox_inches='tight')


def find_bin_edges(data: dict, d_type: str, center: float, max_iter=1000):
    """
    Finds bin edges center around 0.
    No bins can have more then 20% of observations, unless: 
    If it's a core bin, decrease step until it's low enough.
    If it's an edge bin increase interval. 
    """
    # Initialize parameters
    if d_type == "marginal":
        interval = 1.0
        step = 0.1
        int_inc = 1.0
        stp_inc = 0.02
    else:  # "relative"
        interval = 20.0
        step = 1.0
        int_inc = 2.0
        stp_inc = 0.1

    assert -interval <= center <= interval
    min_step = stp_inc  # Minimum step size

    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        done = True

        # Generate core edges (left: [-interval, center]; right: [center, interval])
        left_edges = np.arange(-interval, center, step)
        right_edges = np.arange(center, interval + step, step)  # +step to include endpoint
        core_edges = np.unique(np.concatenate([left_edges, [center], right_edges]))
        edges = np.concatenate([[-np.inf], core_edges, [np.inf]])

        # Check all datasets
        for values in data.values():
            if len(values) == 0:
                continue
            counts, _ = np.histogram(values, bins=edges)
            prob_vec = counts / len(values) 

            # Edge bin violation (first or last bin)
            if prob_vec[0] >= 0.2 or prob_vec[-1] >= 0.2:
                interval += int_inc
                done = False
                break  # Restart with new interval

            # Core bin violation (bins 1 to -2)
            if np.any(prob_vec[1:-1] >= 0.2):
                if step > min_step + 1e-10:  # Avoid float precision issues
                    step -= stp_inc
                    step = max(step, min_step)  # Enforce minimum step
                    done = False
                    break
                else:
                    # Cannot reduce step further; force terminate
                    done = False
                    break

        if done:
            return edges  # All violations resolved

    # Max iterations reached; return best-effort edges
    return edges


def plot_centered_stacked_hist(data: dict, group: str, meas: str, d_type: str, 
                               more_is_better: bool, path: str, center: int = 0) -> None:
    """
    Plots stacked histograms centered around zero with dynamic bin edges.
    One row per PHCM method, showing distribution of deltas with color-coded improvements/degradations.
    
    Args:
        data (dict): {phc_method: [delta_value1, delta_value2, ...], ...}
        group (str): Aggregation level/grouping name
        meas (str): Performance measure name
        d_type (str): 'marginal' or 'relative'
        more_is_better (bool): Whether higher values indicate better performance
        path (str): Output directory path
        center (int, optional): Center reference point. Defaults to 0.
    """
    edges = find_bin_edges(data, d_type, center)
    core_edges = edges[1:-1]  
    first_edge = core_edges[0]
    last_edge = core_edges[-1]
    
    out_file = os.path.join(path, f"{group}_{meas}.png")
    
    total_range = last_edge - first_edge
    plot_left_bound = first_edge - 0.1 * total_range
    plot_right_bound = last_edge + 0.1 * total_range
    plot_edges = np.concatenate([[plot_left_bound], core_edges, [plot_right_bound]])
    
    n_phc = len(data)
    fig, axes = plt.subplots(
        n_phc, 1, 
        figsize=(12, max(2.5 * n_phc, 4)), 
        sharex=True, 
        squeeze=False
    )
    axes = axes.flatten()
    
    for i, (phc, values) in enumerate(data.items()):
        ax = axes[i]
        vals = np.array(values)
        
        counts, _ = np.histogram(vals, bins=edges)
        fracs = counts / counts.sum()
        
        colors = []
        for j in range(len(edges)-1):
            left = edges[j]
            right = edges[j+1]
            
            # Underflow bin (handles all values < first_edge)
            if j == 0:
                colors.append('tab:red' if more_is_better else 'tab:blue')
            
            # Overflow bin (handles all values > last_edge)
            elif j == len(edges)-2:
                colors.append('tab:blue' if more_is_better else 'tab:red')
            
            # Core bins (finite ranges)
            else:
                if right <= center:
                    colors.append('tab:red' if more_is_better else 'tab:blue')
                elif left >= center:
                    colors.append('tab:blue' if more_is_better else 'tab:red')
                # Should never happen since 0 is bin edge
                else:
                    colors.append('gray')

        # Plot bars
        ax.bar(
            plot_edges[:-1], fracs, 
            width=np.diff(plot_edges), 
            align='edge', 
            color=colors, 
            edgecolor='k', 
            linewidth=0.5
        )
        
        # Configure plot aesthetics
        ax.set_ylabel(phc, rotation=0, ha='right', va='center')
        ax.set_ylim(0, np.max(fracs) * 1.15)
        ax.axvline(center, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(plot_left_bound, plot_right_bound)
        
        # Calculate statistics
        expected_value = np.mean(vals)
        if more_is_better:
            frac_improved = np.sum(vals > center) / len(vals)
            frac_degraded = np.sum(vals < center) / len(vals)
        else:
            frac_improved = np.sum(vals < center) / len(vals)
            frac_degraded = np.sum(vals > center) / len(vals)
        
        # Annotation box
        suffix = '%' if d_type == 'relative' else ''
        stats_text = (
            f"Ê[Δ]: {expected_value:.3f}{suffix}\n"
            f"N: {len(vals)}\n"
            f"P̂(Δ{f">{center}{suffix}" if more_is_better else f"<{center}{suffix}"}): {frac_improved:.1%}\n"
            f"P̂(Δ{f"<{center}{suffix}" if more_is_better else f">{center}{suffix}"}): {frac_degraded:.1%}"
        )
        
        ax.text(
            0.98, 0.95, stats_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Configure bottom plot with special annotations
    xlabel = 'Relative Change (%)' if d_type == 'relative' else 'Marginal Change'
    axes[-1].set_xlabel(xlabel)

    # Create special ticks for underflow/overflow
    underflow_pos = (plot_left_bound + first_edge) / 2
    overflow_pos = (last_edge + plot_right_bound) / 2
    special_ticks = [underflow_pos, first_edge, center, last_edge, overflow_pos]

    # Get auto-generated ticks for entire x-axis range
    auto_locator = ticker.MaxNLocator(nbins=15) 
    auto_ticks = auto_locator.tick_values(plot_left_bound, plot_right_bound)

    # Calculate typical tick spacing for proximity threshold
    tick_spacing = np.min(np.diff(auto_ticks)) if len(auto_ticks) > 1 else 1.0
    proximity_threshold = tick_spacing * 0.4  # 40% of typical spacing

    # Filter and combine ticks
    core_ticks = [t for t in auto_ticks 
                if first_edge <= t <= last_edge  # Only core region
                and not any(abs(t - s) < proximity_threshold  # Avoid special ticks
                    for s in [first_edge, center, last_edge])]

    all_ticks = np.unique(np.concatenate([special_ticks, core_ticks]))

    # Generate labels with special handling
    xticklabels = []
    for t in all_ticks:
        if np.isclose(t, underflow_pos, atol=1e-5):
            label = f"<{format_value(first_edge)}{suffix}"
        elif np.isclose(t, overflow_pos, atol=1e-5):
            label = f">={format_value(last_edge)}{suffix}"
        else:
            # Clean number formatting
            label = f"{format_value(t)}{suffix}"
        xticklabels.append(label)

    # Apply to plot
    axes[-1].set_xticks(all_ticks)
    axes[-1].set_xticklabels(xticklabels, rotation=45, ha='right')

    # Set title and save
    plt.suptitle(f"Δ {meas} distributions | Agg: {group}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def format_value(value):
    """Consistent number formatting (strip trailing zeros)"""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip('0').rstrip('.')  

def plot_changes(d_inv_res: dict, measures_d: dict, archs: list, dir_path: str, d_type: str) -> None:
    lrns = get_learners(archs)
    phcms = get_phcm(archs)

    match d_type:
        case "marginal":
            d_dir = "/delta/marg/dist/"
        case "relative":
            d_dir = "/delta/rel/dist/"

    learner_dirs = {}
    for l in lrns:
        learner_dirs[l] = util.create_pwd_dir(dir_path + d_dir + l)
    phcms_dir = util.create_pwd_dir(dir_path + d_dir + "phcms")
    
    for meas, more_is_better in measures_d.items():
        all_values = {}
        for lrn in lrns:
            lrn_values = {}
            for ds_dict in d_inv_res.values():
                for phc in phcms:
                    vals = [float(v[meas]) for v in ds_dict[f"{lrn}.{phc}"]]
                    lrn_values.setdefault(phc, []).extend(vals)
                    all_values.setdefault(phc, []).extend(vals)

            path = learner_dirs[lrn]
            plot_centered_stacked_hist(lrn_values, lrn, meas, d_type, more_is_better, path)
        
        
        plot_centered_stacked_hist(all_values, "all", meas, d_type, more_is_better, phcms_dir)