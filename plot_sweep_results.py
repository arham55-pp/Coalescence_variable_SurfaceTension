"""
Plot sweep results for beta and Pe combinations on the same graphs.
Reads the log files from each parameter combination and creates comparison plots.

Configuration:
- Automatically detects which parameter combinations have been simulated
- Set BETA_SUBSET and PE_SUBSET below to choose which parameters to plot
- Leave empty or None to use all available parameters
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import matplotlib.pyplot as plt
import numpy as np
import os
from postprocess_functions import plt_settings, style_axis, find_neck

# Base directory for sweep results
sweep_dir = "sweep_beta_Pe_results"

# =====================================================================
# CONFIGURATION: Specify which parameters to plot
# =====================================================================
# Set to None or empty list to use ALL available parameters
# Set to specific values to use a subset (e.g., [0, 0.1] or [1, 10, 100])
BETA_SUBSET = None   # All available: [0, 0.1, 0.5, 0.9]
PE_SUBSET = None     # All available: [1, 10, 100, 1000, 10000]

# Detect available parameters
available_beta = set()
available_Pe = set()

for item in os.listdir(sweep_dir):
    if os.path.isdir(os.path.join(sweep_dir, item)) and item.startswith('beta_'):
        parts = item.split('_')
        if len(parts) >= 4:
            try:
                beta = float(parts[1])
                Pe = int(parts[3])
                # Check if this folder has domain data
                domain_dir = os.path.join(sweep_dir, item, 'domain')
                if os.path.exists(domain_dir):
                    available_beta.add(beta)
                    available_Pe.add(Pe)
            except:
                pass

# Use subset if specified, otherwise use all available
beta_values = sorted(list(BETA_SUBSET if BETA_SUBSET else available_beta))
Pe_values = sorted(list(PE_SUBSET if PE_SUBSET else available_Pe))

print(f"\nAvailable parameters:")
print(f"  Beta values: {sorted(list(available_beta))}")
print(f"  Pe values: {sorted(list(available_Pe))}")
print(f"\nPlotting with:")
print(f"  Beta subset: {beta_values}")
print(f"  Pe subset: {Pe_values}\n")

# Storage for results
slopes_h0 = {}  # slopes_h0[(beta, Pe)] = m_h0
slopes_x0 = {}  # slopes_x0[(beta, Pe)] = m_x0
all_results = []  # List of [beta, Pe, m_h0, m_x0]

print("Reading sweep results...")
print("-" * 60)

for beta in beta_values:
    for Pe in Pe_values:
        folder_name = f"beta_{beta}_Pe_{Pe}"
        folder_path = os.path.join(sweep_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"WARNING: {folder_name} not found")
            continue
        
        # Check for domain subfolder
        domain_dir = os.path.join(folder_path, "domain")
        if not os.path.exists(domain_dir):
            print(f"WARNING: {domain_dir} does not exist")
            continue
        
        try:
            files = sorted([f for f in os.listdir(domain_dir) if f.endswith('.txt')])
            
            if len(files) == 0:
                print(f"WARNING: No data files in {folder_name}")
                continue
            
            # Collect time series data
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
            
            # Filter valid times
            valid_time_mask = time_data > 0
            time_data = time_data[valid_time_mask]
            h0_data = h0_data[valid_time_mask]
            x0_data = x0_data[valid_time_mask]
            
            # Remove oscillating points
            x0_threshold = 1e-3
            valid_x0_mask = np.abs(x0_data) > x0_threshold
            time_data = time_data[valid_x0_mask]
            h0_data = h0_data[valid_x0_mask]
            x0_data = x0_data[valid_x0_mask]
            
            if len(time_data) < 2:
                print(f"WARNING: Not enough data for beta={beta}, Pe={Pe}")
                continue
            
            # Fit log-log
            log_time_data = np.log(time_data)
            log_h0_data = np.log(np.maximum(h0_data, 1e-10))
            log_x0_data = np.log(np.maximum(np.abs(x0_data), 1e-10))
            
            coeffs_h0 = np.polyfit(log_time_data, log_h0_data, 1)
            m_h0 = coeffs_h0[0]
            
            coeffs_x0 = np.polyfit(log_time_data, log_x0_data, 1)
            m_x0 = coeffs_x0[0]
            
            slopes_h0[(beta, Pe)] = m_h0
            slopes_x0[(beta, Pe)] = m_x0
            all_results.append([beta, Pe, m_h0, m_x0])
            
            print(f"beta={beta:3.1f}, Pe={Pe:5d}: m_h0={m_h0:7.4f}, m_x0={m_x0:7.4f}")
            
        except Exception as e:
            print(f"ERROR processing beta={beta}, Pe={Pe}: {str(e)}")

print("-" * 60)
print(f"\nSuccessfully processed {len(all_results)} parameter combinations\n")

if len(all_results) == 0:
    print("No results to plot!")
    exit()

# Create output directory
plot_dir = "sweep_comparison_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Convert to numpy array
all_results = np.array(all_results)
betas = np.unique(all_results[:, 0])
Pes = np.unique(all_results[:, 1])

# ============================================================================
# Plot 1: Slopes vs Pe for each beta (line plot)
# ============================================================================
print("Creating slope vs Pe plots...")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.Set1(np.linspace(0, 1, len(betas)))

for idx, beta in enumerate(betas):
    mask = all_results[:, 0] == beta
    beta_results = all_results[mask]
    # Sort by Pe
    beta_results = beta_results[np.argsort(beta_results[:, 1])]
    
    ax1.plot(beta_results[:, 1], beta_results[:, 2], 'o-', 
             color=colors[idx], linewidth=2.5, markersize=8, label=f'β={beta}')
    ax2.plot(beta_results[:, 1], beta_results[:, 3], 'o-', 
             color=colors[idx], linewidth=2.5, markersize=8, label=f'β={beta}')

style_axis(ax1, xlabel='Peclet number Pe', ylabel='Slope m_h0',
           title='Neck Height Scaling Exponent vs Pe')
ax1.set_xscale('log')
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')
ax1.grid(True, which='both', alpha=0.3)

style_axis(ax2, xlabel='Peclet number Pe', ylabel='Slope m_x0',
           title='Neck Position Scaling Exponent vs Pe')
ax2.set_xscale('log')
ax2.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/slopes_vs_Pe.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 2: Slopes vs Beta for each Pe (line plot)
# ============================================================================
print("Creating slope vs Beta plots...")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.tab10(np.linspace(0, 1, len(Pes)))

for idx, Pe in enumerate(Pes):
    mask = all_results[:, 1] == Pe
    Pe_results = all_results[mask]
    # Sort by beta
    Pe_results = Pe_results[np.argsort(Pe_results[:, 0])]
    
    ax1.plot(Pe_results[:, 0], Pe_results[:, 2], 'o-', 
             color=colors[idx], linewidth=2.5, markersize=8, label=f'Pe={Pe}')
    ax2.plot(Pe_results[:, 0], Pe_results[:, 3], 'o-', 
             color=colors[idx], linewidth=2.5, markersize=8, label=f'Pe={Pe}')

style_axis(ax1, xlabel='Surfactant Elasticity β', ylabel='Slope m_h0',
           title='Neck Height Scaling Exponent vs β')
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')
ax1.grid(True, alpha=0.3)

style_axis(ax2, xlabel='Surfactant Elasticity β', ylabel='Slope m_x0',
           title='Neck Position Scaling Exponent vs β')
ax2.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/slopes_vs_beta.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 3: Heatmaps of slopes (m_h0 and m_x0)
# ============================================================================
print("Creating heatmap plots...")

# Create 2D arrays for heatmaps
m_h0_matrix = np.zeros((len(betas), len(Pes)))
m_x0_matrix = np.zeros((len(betas), len(Pes)))

for result in all_results:
    beta_idx = np.where(betas == result[0])[0][0]
    Pe_idx = np.where(Pes == result[1])[0][0]
    m_h0_matrix[beta_idx, Pe_idx] = result[2]
    m_x0_matrix[beta_idx, Pe_idx] = result[3]

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot m_h0 heatmap
im1 = ax1.imshow(m_h0_matrix, cmap='RdBu_r', aspect='auto', origin='lower')
ax1.set_xticks(range(len(Pes)))
ax1.set_xticklabels([f'{Pe}' for Pe in Pes], fontsize=plt_settings['AxesFont'])
ax1.set_yticks(range(len(betas)))
ax1.set_yticklabels([f'{beta}' for beta in betas], fontsize=plt_settings['AxesFont'])
ax1.set_xlabel('Peclet number Pe', fontsize=plt_settings['LabelsFont'])
ax1.set_ylabel('Surfactant Elasticity β', fontsize=plt_settings['LabelsFont'])
ax1.set_title('Neck Height Scaling Exponent m_h0', fontsize=plt_settings['TitleFont'], fontweight='bold')

# Add values to heatmap
for i in range(len(betas)):
    for j in range(len(Pes)):
        text = ax1.text(j, i, f'{m_h0_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=9)

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('m_h0', fontsize=plt_settings['ColorbarFont'])

# Plot m_x0 heatmap
im2 = ax2.imshow(m_x0_matrix, cmap='RdBu_r', aspect='auto', origin='lower')
ax2.set_xticks(range(len(Pes)))
ax2.set_xticklabels([f'{Pe}' for Pe in Pes], fontsize=plt_settings['AxesFont'])
ax2.set_yticks(range(len(betas)))
ax2.set_yticklabels([f'{beta}' for beta in betas], fontsize=plt_settings['AxesFont'])
ax2.set_xlabel('Peclet number Pe', fontsize=plt_settings['LabelsFont'])
ax2.set_ylabel('Surfactant Elasticity β', fontsize=plt_settings['LabelsFont'])
ax2.set_title('Neck Position Scaling Exponent m_x0', fontsize=plt_settings['TitleFont'], fontweight='bold')

# Add values to heatmap
for i in range(len(betas)):
    for j in range(len(Pes)):
        text = ax2.text(j, i, f'{m_x0_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=9)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('m_x0', fontsize=plt_settings['ColorbarFont'])

plt.tight_layout()
plt.savefig(f'{plot_dir}/slopes_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Plot 4: 3D surface plot
# ============================================================================
print("Creating 3D surface plots...")

from mpl_toolkits.mplot3d import Axes3D

fig4 = plt.figure(figsize=(16, 6))

# Create meshgrid
Pe_mesh, beta_mesh = np.meshgrid(Pes, betas)

ax1 = fig4.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(Pe_mesh, beta_mesh, m_h0_matrix, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Peclet number Pe', fontsize=plt_settings['LabelsFont'])
ax1.set_ylabel('Surfactant Elasticity β', fontsize=plt_settings['LabelsFont'])
ax1.set_zlabel('m_h0', fontsize=plt_settings['LabelsFont'])
ax1.set_title('Neck Height Scaling Exponent', fontsize=plt_settings['TitleFont'], fontweight='bold')
fig4.colorbar(surf1, ax=ax1, label='m_h0', shrink=0.5)

ax2 = fig4.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(Pe_mesh, beta_mesh, m_x0_matrix, cmap='plasma', alpha=0.8)
ax2.set_xlabel('Peclet number Pe', fontsize=plt_settings['LabelsFont'])
ax2.set_ylabel('Surfactant Elasticity β', fontsize=plt_settings['LabelsFont'])
ax2.set_zlabel('m_x0', fontsize=plt_settings['LabelsFont'])
ax2.set_title('Neck Position Scaling Exponent', fontsize=plt_settings['TitleFont'], fontweight='bold')
fig4.colorbar(surf2, ax=ax2, label='m_x0', shrink=0.5)

plt.tight_layout()
plt.savefig(f'{plot_dir}/slopes_3d_surface.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print(f"All comparison plots saved to '{plot_dir}' folder!")
print("=" * 60)
print("\nGenerated plots:")
print("  1. slopes_vs_Pe.png - Scaling exponents vs Peclet number")
print("  2. slopes_vs_beta.png - Scaling exponents vs surfactant elasticity")
print("  3. slopes_heatmap.png - 2D heatmaps of both exponents")
print("  4. slopes_3d_surface.png - 3D surface plots of both exponents")
