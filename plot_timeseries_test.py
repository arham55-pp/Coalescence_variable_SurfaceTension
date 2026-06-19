"""
Test plot for combined time series data - using only a subset of parameters
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, '.')
from postprocess_functions import plt_settings, style_axis, find_neck

sweep_dir = "sweep_beta_Pe_results"

# Test with just a few parameters
test_params = [
    ("beta_0.1_Pe_1", 0.1, 1),
    ("beta_0.1_Pe_10", 0.1, 10),
    ("beta_0.5_Pe_1", 0.5, 1),
    ("beta_0.5_Pe_10", 0.5, 10),
]

print("Testing with subset of parameters:")
print(test_params)

fig = plt.figure(figsize=(14, 10))
gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

ax_h_lin = fig.add_subplot(gs[0, 0])
ax_x_lin = fig.add_subplot(gs[0, 1])
ax_h_log = fig.add_subplot(gs[1, 0])
ax_x_log = fig.add_subplot(gs[1, 1])

colors = ['red', 'blue', 'green', 'orange']

for idx, (folder_name, beta, Pe) in enumerate(test_params):
    domain_dir = os.path.join(sweep_dir, folder_name, 'domain')
    
    print(f"\nLoading {folder_name}...")
    
    if not os.path.exists(domain_dir):
        print(f"  Skipped: directory not found")
        continue
    
    try:
        files = sorted([f for f in os.listdir(domain_dir) if f.endswith('.txt')])[:50]  # Limit to first 50 files
        
        time_data = []
        h0_data = []
        x0_data = []
        
        for f in files:
            with open(os.path.join(domain_dir, f)) as file:
                header = file.readline()
                time = float(header.split('@time=')[-1])
                data = np.loadtxt(file)
                x = data[:, 0]
                h = data[:, 1]
                
                time_data.append(time)
                x0, h0 = find_neck(x, h)
                x0_data.append(x0)
                h0_data.append(h0)
        
        time_data = np.array(time_data)
        h0_data = np.array(h0_data)
        x0_data = np.array(x0_data)
        
        # Filter
        valid_time_mask = time_data > 0
        time_data = time_data[valid_time_mask]
        h0_data = h0_data[valid_time_mask]
        x0_data = x0_data[valid_time_mask]
        
        valid_x0_mask = np.abs(x0_data) > 1e-3
        time_data = time_data[valid_x0_mask]
        h0_data = h0_data[valid_x0_mask]
        x0_data = x0_data[valid_x0_mask]
        
        print(f"  Loaded {len(time_data)} timesteps")
        
        label = f'β={beta}, Pe={Pe}'
        color = colors[idx]
        
        # Linear
        ax_h_lin.scatter(time_data, h0_data, c=color, s=20, alpha=0.6, label=label)
        ax_x_lin.scatter(time_data, x0_data, c=color, s=20, alpha=0.6, label=label)
        
        # Log-log
        log_time = np.log(time_data)
        log_h0 = np.log(np.maximum(h0_data, 1e-10))
        log_x0 = np.log(np.maximum(np.abs(x0_data), 1e-10))
        
        ax_h_log.scatter(log_time, log_h0, c=color, s=20, alpha=0.6, label=label)
        ax_x_log.scatter(log_time, log_x0, c=color, s=20, alpha=0.6, label=label)
        
    except Exception as e:
        print(f"  Error: {e}")

# Style
style_axis(ax_h_lin, xlabel='Time t', ylabel='h₀', title='Neck Height (Linear)')
ax_h_lin.legend(fontsize=9)
ax_h_lin.grid(True, alpha=0.3)

style_axis(ax_x_lin, xlabel='Time t', ylabel='x₀', title='Neck Position (Linear)')
ax_x_lin.legend(fontsize=9)
ax_x_lin.grid(True, alpha=0.3)

style_axis(ax_h_log, xlabel='log(t)', ylabel='log(h₀)', title='Neck Height (Log-Log)')
ax_h_log.legend(fontsize=9)
ax_h_log.grid(True, which='both', alpha=0.3)

style_axis(ax_x_log, xlabel='log(t)', ylabel='log(|x₀|)', title='Neck Position (Log-Log)')
ax_x_log.legend(fontsize=9)
ax_x_log.grid(True, which='both', alpha=0.3)

plot_dir = "sweep_comparison_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

outfile = f'{plot_dir}/combined_timeseries_test.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {outfile}")
plt.close()

print("Done!")
