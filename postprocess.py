"""
Postprocessing for surfactant coalescence simulations.

Generates 5 publication-quality PDF figures from simulation output,
analyzing the evolution of film height, surfactant concentration,
and coalescence dynamics.

Usage:
    python postprocess.py <output_directory>
    python postprocess.py coalescence  # default

Output Files (saved to <directory>_plots/):
    1. height_surfactant_evolution.pdf
       - Top panel: height profiles h(x) at selected times
       - Bottom panel: surfactant profiles Gamma(x) at same times
       - Time indicated by colorbar (viridis colormap)

    2. spacetime_height.pdf
       - Contour plot of h(x,t) showing coalescence progression
       - x-axis: spatial position, y-axis: time
       - Useful for visualizing bridge growth and neck motion

    3. time_series_analysis.pdf
       - 2x2 grid of key quantities vs time:
         * Neck height h_0(t): minimum film thickness
         * Neck position x_0(t): location of minimum
         * Maximum height h_max(t)
         * Maximum surfactant |Gamma|_max(t)

    4. bridge_region_zoom.pdf
       - Detailed view of bridge region (|x| < 0.5) at 5 stages
       - Shows h, p, Gamma profiles to analyze Marangoni effects
       - Stages: Initial, Early, Middle, Late, Final

    5. neck_trajectory.pdf
       - Phase portrait: x_0 vs h_0 parametrized by time
       - Arrows indicate direction of evolution
       - Colorbar shows time progression

Key Metrics Extracted:
    - h_0(t): Neck height (from find_neck using peak detection)
    - x_0(t): Horizontal position of neck minimum
    - Gamma_max(t): Maximum surfactant concentration
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering
matplotlib.rcParams['text.latex.preamble'] = r''  # Empty LaTeX preamble
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'  # Use DejaVu font for math
matplotlib.rcParams['mathtext.default'] = 'regular'  # Use regular text for math
matplotlib.rcParams['figure.constrained_layout.use'] = False  # Disable constrained layout
matplotlib.rcParams['pdf.use14corefonts'] = True
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.gridspec import GridSpec
from postprocess_functions import plt_settings, style_axis, find_neck

'''
def calculate_x0_global_maximum(time, x0):
    """
    Find the global maximum of the neck location x_0(t).

    Returns:
    --------
    tuple
        (global_max_time, global_max_value)
    """
    time = np.asarray(time)
    x0 = np.asarray(x0)

    if len(x0) == 0:
        return None, None

    global_max_idx = np.argmax(x0)
    return time[global_max_idx], x0[global_max_idx]

'''
# Get base folder from command line, default to "coalescence"
base_folder = sys.argv[1] if len(sys.argv) > 1 else "coalescence"
output_dir = f"{base_folder}/domain"
plot_dir = f"{base_folder}_plots"

files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')])

# Create folder for plots
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Initialize lists for time series data
time_data = []
h0_data = []       # neck height
x0_data = []       # neck position
h_max_data = []
c_max_data = []

# Read first and last files to understand the evolution
with open(os.path.join(output_dir, files[0])) as f:
    header = f.readline()
    data_first = np.loadtxt(f)
    
with open(os.path.join(output_dir, files[-1])) as f:
    header = f.readline()
    data_last = np.loadtxt(f)

# Plot 1: Height profile evolution at selected times
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

# Select files to plot (every 40th file)
selected_indices = range(0, len(files), 40)
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

time_values = []
for idx, file_idx in enumerate(selected_indices):
    with open(os.path.join(output_dir, files[file_idx])) as f:
        header = f.readline()
        time = float(header.split('@time=')[-1])
        time_values.append(time)
        data = np.loadtxt(f)
        x = data[:, 0]
        h = data[:, 1]
        c = data[:, 3]/data[:,1]

        ax1.plot(x, h, color=colors[idx])
        ax2.plot(x, c, color=colors[idx])

style_axis(ax1, ylabel='Height h', title='Evolution of Droplet Coalescence with Surfactants')
style_axis(ax2, xlabel='Position x', ylabel='Volume Fraction Field')

# Add colorbar for time
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=time_values[0], vmax=time_values[-1]))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=[ax1, ax2], location='right', shrink=0.6, pad=0.08)
cbar.set_label('Time t', fontsize=plt_settings['ColorbarFont'], labelpad=10)
cbar.ax.tick_params(labelsize=plt_settings['AxesFont'])

plt.savefig(f'{plot_dir}/height_c_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
'''
# Plot 2: Spacetime diagram of height
print("Creating spacetime diagram...")
times = []
height_matrix = []

# First pass: determine common x grid from first file
with open(os.path.join(output_dir, files[0])) as file:
    file.readline()
    data = np.loadtxt(file)
    x_common = np.linspace(data[:, 0].min(), data[:, 0].max(), 500)

for i, f in enumerate(files[::2]):  # Use every other file for speed
    with open(os.path.join(output_dir, f)) as file:
        header = file.readline()
        time = float(header.split('@time=')[-1])
        times.append(time)
        data = np.loadtxt(file)
        x_data = data[:, 0]
        h_data = data[:, 1]
        # Interpolate onto common grid
        h_interp = np.interp(x_common, x_data, h_data)
        height_matrix.append(h_interp)

height_matrix = np.array(height_matrix)
x = x_common

fig2, ax = plt.subplots(figsize=(12, 8))
im = ax.pcolormesh(x, times, height_matrix, shading='auto', cmap='viridis')
style_axis(ax, xlabel='Position x', ylabel='Time t', title='Spacetime Evolution of Film Height')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Height h', fontsize=plt_settings['ColorbarFont'], labelpad=10)
cbar.ax.tick_params(labelsize=plt_settings['AxesFont'])
plt.savefig(f'{plot_dir}/spacetime_height.png', dpi=150, bbox_inches='tight')
plt.close()
'''
# Plot 3: Time series analysis

#INPUT DATA
print("Analyzing time series data...")
for f in files:
    with open(os.path.join(output_dir, f)) as file:
        header = file.readline()
        time = float(header.split('@time=')[-1])
        data = np.loadtxt(file)
        x = data[:, 0]
        h = data[:, 1]
        c = data[:, 3]/data[:,1]
        
        time_data.append(time)
        x0, h0 = find_neck(x, h)
        x0_data.append(x0)
        h0_data.append(h0)
        h_max_data.append(np.max(h))
        c_max_data.append(np.max(np.abs(c)))



# Convert to numpy arrays for easier manipulation
time_data = np.array(time_data)
h0_data = np.array(h0_data)
x0_data = np.array(x0_data)
'''
# Detect discontinuities in h0_data and x0_data
coalescence_times = []

# Calculate differences between consecutive points
dh0 = np.abs(np.diff(h0_data))
dx0 = np.abs(np.diff(x0_data))

# Find discontinuities using a threshold (e.g., 2 standard deviations above mean)
threshold_h0 = np.mean(dh0) + 2 * np.std(dh0)
threshold_x0 = np.mean(dx0) + 2 * np.std(dx0)

# Find indices where discontinuities occur
h0_discontinuity_indices = np.where(dh0 > threshold_h0)[0]
x0_discontinuity_indices = np.where(dx0 > threshold_x0)[0]

# Combine and sort discontinuity indices
all_discontinuity_indices = np.unique(np.concatenate([h0_discontinuity_indices, x0_discontinuity_indices]))

# Skip the first 20% of data to avoid false positives at the beginning
skip_points = max(1, int(0.2 * len(time_data)))
all_discontinuity_indices = all_discontinuity_indices[all_discontinuity_indices >= skip_points]

if len(all_discontinuity_indices) > 0:
    # Store only the first discontinuity time after the skip region
    first_discontinuity_idx = all_discontinuity_indices[0]
    coalescence_times.append(time_data[first_discontinuity_idx + 1])

coalescence_times = np.array(coalescence_times)'''

# Filter time_data to only include positive times, then compute log
valid_time_mask = time_data > 0
time_data_valid = time_data[valid_time_mask]
h0_data_valid = h0_data[valid_time_mask]
x0_data_valid = x0_data[valid_time_mask]

# Remove oscillating points where neck position is near zero
# These points are typically noisy and don't represent true dynamics
# Using a very small threshold to only catch extreme oscillations
h0_threshold = 1e-5  # Minimum magnitude of neck height to include
valid_h0_mask = np.abs(h0_data_valid) > h0_threshold
time_data_valid = time_data_valid[valid_h0_mask]
h0_data_valid = h0_data_valid[valid_h0_mask]
x0_data_valid = x0_data_valid[valid_h0_mask]

log_time_data = np.log(time_data_valid)
log_h0_data = np.log(h0_data_valid)
log_x0_data = np.log(np.abs(x0_data_valid))

x0_global_max_time, x0_global_max_value = calculate_x0_global_maximum(time_data_valid, x0_data_valid)

if x0_global_max_time is not None:
    print(f"\nGlobal maximum of neck position x_0: t = {x0_global_max_time:.6f}, x_0 = {x0_global_max_value:.6f}")


fig3 = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig3)

ax1 = fig3.add_subplot(gs[0, 0])
ax1.scatter(time_data_valid, h0_data_valid, c='b', s=50, zorder=3)
style_axis(ax1, xlabel='Time t', ylabel='Neck height h_0',
           title='Neck Height Evolution')

ax2 = fig3.add_subplot(gs[0, 1])
ax2.scatter(time_data_valid, x0_data_valid, c='g', s=50, zorder=3)
style_axis(ax2, xlabel='Time t', ylabel='Neck position x_0',
           title='Neck Position Evolution')
           
plt.tight_layout()
plt.savefig(f'{plot_dir}/time_series_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3b: Log-log time series analysis with t^alpha scaling
print("Creating log-log time series plot")

# Parameters for straight line: y = 10^c * x^m
m1 = 1 # slope
c1 = -5.4 # intercept
m2=1
c2=-4.8
fig3b = plt.figure(figsize=(14, 10))
gs3b = GridSpec(2, 2, figure=fig3b)

ax1 = fig3b.add_subplot(gs3b[0, 0])
ax1.plot(log_time_data, log_h0_data, 'bo', markersize=6, linestyle='none', label='Neck height h_0')
style_axis(ax1, xlabel='Time t', ylabel='Neck height h_0',
           title=f'Neck Height Evolution (Log-Log Scale)')
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')
ax1.grid(True, which='both', alpha=0.3)

# Add straight line with slope m1 and intercept c1 to ax1
x_line = np.linspace(log_time_data.min(), log_time_data.max(), 100)
y_line = c1 + m1 *x_line
ax1.plot(x_line, y_line, 'k-', linewidth=2.5)
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')

ax2 = fig3b.add_subplot(gs3b[0, 1])
ax2.plot(log_time_data, log_x0_data, 'go', markersize=6, linestyle='none', label='Neck position |x_0|')
style_axis(ax2, xlabel='Time t', ylabel='Neck position',
           title=f'Neck Position Evolution (Log-Log Scale)')
ax2.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')
ax2.grid(True, which='both', alpha=0.3)

# Add straight line with slope m2 and intercept c2 to ax2
y_line_ax2 = c2 + m2*x_line
ax2.plot(x_line, y_line_ax2, 'k-', linewidth=3)
ax2.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')


plt.tight_layout()
plt.savefig(f'{plot_dir}/time_series_analysis_loglog.png', dpi=150, bbox_inches='tight')
plt.close()


\
# Plot 4: Zoom on the bridge region

print("Creating bridge region plot...")
fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot at different stages of coalescence
stages = [0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]
colors = ['blue', 'green', 'orange', 'red', 'purple']
labels = ['Initial', 'Early', 'Middle', 'Late', 'Final']

for idx, (stage, color, label) in enumerate(zip(stages, colors, labels)):
    with open(os.path.join(output_dir, files[stage])) as f:
        header = f.readline()
        time = float(header.split('@time=')[-1])
        data = np.loadtxt(f)
        x = data[:, 0]
        h = data[:, 1]
        p = data[:, 2]
        c = data[:, 3]/data[:, 1]

        # Bridge is at x=0 where the two drops meet
        x_bridge = 0.0

        # Zoom window and sort by x-coordinate
        mask = np.abs(x - x_bridge) < 0.5
        sort_idx = np.argsort(x[mask])
        x_sorted = x[mask][sort_idx]
        h_sorted = h[mask][sort_idx]
        p_sorted = p[mask][sort_idx]
        c_sorted = c[mask][sort_idx]

        ax1.plot(x_sorted, h_sorted, color=color, linewidth=3, label=f'{label} (t={time:.1f})')
        ax2.plot(x_sorted, p_sorted, color=color, linewidth=3)
        ax3.plot(x_sorted, c_sorted, color=color, linewidth=3)

style_axis(ax1, ylabel='Height h', title='Bridge Region Evolution During Coalescence')
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False)

style_axis(ax2, ylabel='Pressure p')

style_axis(ax3, xlabel='Position x', ylabel='Volume fraction field')

plt.tight_layout()
plt.savefig(f'{plot_dir}/bridge_region_zoom.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Neck trajectory (x0 vs h0)
print("Creating neck trajectory plot...")
fig5, ax = plt.subplots(figsize=(10, 10))

# Use color gradient for time
colors = plt.cm.plasma(np.linspace(0, 1, len(time_data_valid)))

ax.scatter(x0_data_valid, h0_data_valid, c=colors, alpha=0.7, s=80, edgecolors='w', linewidth=0.5, zorder=3)
ax.plot(x0_data_valid, h0_data_valid, 'k-', alpha=0.3, linewidth=1, zorder=2)

# Add arrows to show direction
arrow_indices = np.linspace(0, len(x0_data_valid)-2, 10, dtype=int)
for idx in arrow_indices:
    ax.annotate('', xy=(x0_data_valid[idx+1], h0_data_valid[idx+1]),
                xytext=(x0_data_valid[idx], h0_data_valid[idx]),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, lw=2))

style_axis(ax, xlabel='Neck position x_0', ylabel='Neck height h_0',
           title='Neck Trajectory During Coalescence')

# Add colorbar for time
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=time_data_valid[0], vmax=time_data_valid[-1]))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Time t', fontsize=plt_settings['ColorbarFont'], labelpad=10)
cbar.ax.tick_params(labelsize=plt_settings['AxesFont'])

plt.savefig(f'{plot_dir}/neck_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"All plots have been saved to the '{plot_dir}' folder!")
print(f"Total number of timesteps analyzed: {len(files)}")
print(f"Time range (filtered): {time_data_valid[0]:.2f} to {time_data_valid[-1]:.2f}")
print(f"Final neck height: {h0_data_valid[-1]:.6f}")
print(f"Final neck position: {x0_data_valid[-1]:.6f}")
#print(f"\nLog-log plot generated with α={ALPHA} (edit ALPHA at the top to change the t^alpha exponent)")
'''

# ============================================================================
# Function to extract slopes for parameter sweep
# ============================================================================
def extract_slopes(base_folder, beta=None, Pe=None):
    """
    Extract slopes from neck height and neck position data.
    
    Parameters:
    -----------
    base_folder : str
        Base folder containing the simulation output
    beta : float, optional
        Surfactant elasticity parameter
    Pe : float, optional
        Peclet number
    
    Returns:
    --------
    tuple : (beta, Pe, m_h0, m_x0)
        Array of [beta, Pe, slope_from_h0, slope_from_x0]
        Returns None if computation fails
    """
    try:
        output_dir = f"{base_folder}/domain"
        if not os.path.exists(output_dir):
            print(f"Warning: {output_dir} does not exist")
            return None
        
        files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')])
        if len(files) == 0:
            print(f"Warning: No data files found in {output_dir}")
            return None
        
        # Collect time series data
        time_data_sweep = []
        h0_data_sweep = []
        x0_data_sweep = []
        
        for f in files:
            with open(os.path.join(output_dir, f)) as file:
                header = file.readline()
                time = float(header.split('@time=')[-1])
                data = np.loadtxt(file)
                x = data[:, 0]
                h = data[:, 1]
                
                time_data_sweep.append(time)
                x0, h0 = find_neck(x, h)
                x0_data_sweep.append(x0)
                h0_data_sweep.append(h0)
        
        time_data_sweep = np.array(time_data_sweep)
        h0_data_sweep = np.array(h0_data_sweep)
        x0_data_sweep = np.array(x0_data_sweep)
        
        # Filter time_data to only include positive times
        valid_time_mask = time_data_sweep > 0
        time_data_sweep = time_data_sweep[valid_time_mask]
        h0_data_sweep = h0_data_sweep[valid_time_mask]
        x0_data_sweep = x0_data_sweep[valid_time_mask]
        
        # Remove oscillating points where neck position is near zero BEFORE discontinuity detection
        x0_threshold = 1e-3  # Minimum magnitude of neck position to include
        valid_x0_mask = np.abs(x0_data_sweep) > x0_threshold
        time_data_sweep = time_data_sweep[valid_x0_mask]
        h0_data_sweep = h0_data_sweep[valid_x0_mask]
        x0_data_sweep = x0_data_sweep[valid_x0_mask]
        
        # Detect discontinuities
        dh0 = np.abs(np.diff(h0_data_sweep))
        dx0 = np.abs(np.diff(x0_data_sweep))
        
        threshold_h0 = np.mean(dh0) + 2 * np.std(dh0)
        threshold_x0 = np.mean(dx0) + 2 * np.std(dx0)
        
        h0_discontinuity_indices = np.where(dh0 > threshold_h0)[0]
        x0_discontinuity_indices = np.where(dx0 > threshold_x0)[0]
        
        all_discontinuity_indices = np.unique(np.concatenate([h0_discontinuity_indices, x0_discontinuity_indices]))
        
        # Skip the first 20% of data
        skip_points = max(1, int(0.2 * len(time_data_sweep)))
        all_discontinuity_indices = all_discontinuity_indices[all_discontinuity_indices >= skip_points]
        
        # Use discontinuity if detected, otherwise use full dataset
        if len(all_discontinuity_indices) > 0:
            coalescence_time = time_data_sweep[all_discontinuity_indices[0] + 1]
            log_coalescence_time = np.log(coalescence_time)
            
            log_time_data = np.log(time_data_sweep)
            filter_mask = log_time_data < log_coalescence_time
            time_filtered = time_data_sweep[filter_mask]
            h0_filtered = h0_data_sweep[filter_mask]
            x0_filtered = x0_data_sweep[filter_mask]
            log_time_filtered = log_time_data[filter_mask]
        else:
            # No discontinuity detected, use full dataset
            time_filtered = time_data_sweep
            h0_filtered = h0_data_sweep
            x0_filtered = x0_data_sweep
            log_time_filtered = np.log(time_data_sweep)
        
        if len(time_filtered) < 2:
            print(f"Warning: Not enough data points for beta={beta}, Pe={Pe}")
            return None
        
        # Ensure positive values and fit
        h0_filtered = np.maximum(h0_filtered, 1e-10)
        x0_filtered = np.maximum(np.abs(x0_filtered), 1e-10)
        
        log_h0_filtered = np.log(h0_filtered)
        log_x0_filtered = np.log(x0_filtered)
        
        # Fit lines
        coeffs_h0 = np.polyfit(log_time_filtered, log_h0_filtered, 1)
        m_h0 = coeffs_h0[0]
        
        coeffs_x0 = np.polyfit(log_time_filtered, log_x0_filtered, 1)
        m_x0 = coeffs_x0[0]
        
        return np.array([beta, Pe, m_h0, m_x0])
    
    except Exception as e:
        print(f"Error processing beta={beta}, Pe={Pe}: {str(e)}")
        return None'''
