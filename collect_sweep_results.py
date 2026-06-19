"""
Collect results from parameter sweep (beta and Pe) into a structured array.
Computes slopes from neck height and neck position for each parameter combination.

Usage:
    python collect_sweep_results.py <output_directory> <beta> <Pe>
    
Returns:
    Prints: beta Pe slope_h0 slope_x0
    
Example:
    python collect_sweep_results.py sweep_beta_Pe_results/beta_0_Pe_1 0 1
    # Output: 0 1 -0.5 -0.25
"""
import sys
import numpy as np
from postprocess import extract_slopes

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python collect_sweep_results.py <base_folder> <beta> <Pe>")
        sys.exit(1)
    
    base_folder = sys.argv[1]
    beta = float(sys.argv[2])
    Pe = float(sys.argv[3])
    
    result = extract_slopes(base_folder, beta=beta, Pe=Pe)
    
    if result is not None:
        # Print in format: beta Pe slope_h0 slope_x0
        print(f"{result[0]} {result[1]} {result[2]:.6f} {result[3]:.6f}")
    else:
        print(f"Failed to extract slopes for beta={beta}, Pe={Pe}")
        sys.exit(1)
