"""Estimate HH ion-channel conductances from synthetic noisy voltage observations.

Run `python Vm_generator.py` first to generate Generated_Vm.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from Vm_generator import (
    TRUE_G_NA, TRUE_G_K, TRUE_G_L, simulate_hh, t_span,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "Generated_Vm.csv"
PARAMETERS_FILE = PROJECT_DIR / "recovered_g_params.txt"


def load_observations():
    data = pd.read_csv(DATA_FILE)
    times = data["Time_step"].to_numpy(dtype=float)
    voltages = data["Noisy_Vm"].to_numpy(dtype=float)
    if (len(times) < 2 or not np.all(np.diff(times) > 0)
            or times[0] < t_span[0] or times[-1] > t_span[1]
            or not np.all(np.isfinite(voltages))):
        raise ValueError("Invalid voltage observations or time coordinates")
    return times, voltages


def make_objective(times, observed_voltage):
    """Return an objective that computes voltage mean squared error (mV^2)."""
    def objective(conductances):
        try:
            predicted = simulate_hh(conductances, times=times).y[0]
            return float(np.mean((observed_voltage - predicted)**2))
        except (RuntimeError, ValueError, OverflowError, FloatingPointError):
            return 1e12
    return objective


def main():
    times, observed_voltage = load_observations()
    objective = make_objective(times, observed_voltage)

    # Conductance bounds (mS/cm^2): Na+, K+, leak.
    bounds = [(50.0, 150.0), (10.0, 50.0), (0.0, 1.0)]
    print("Starting differential-evolution optimisation...")
    result = differential_evolution(
        objective, bounds, strategy="best1bin", maxiter=30,
        popsize=10, seed=42, disp=True, polish=True,
    )
    if not np.isfinite(result.fun) or result.fun >= 1e12:
        raise RuntimeError("Optimisation failed to find a valid simulation")

    print("\nOptimisation finished")
    print(f"Status: {result.message}")
    print(f"Optimiser convergence flag: {result.success}")
    print(f"Mean squared error: {result.fun:.6f} mV^2")
    print(f"Ground truth: Na={TRUE_G_NA}, K={TRUE_G_K}, L={TRUE_G_L}")
    print(f"Fitted:       Na={result.x[0]:.6f}, K={result.x[1]:.6f}, L={result.x[2]:.6f}")

    with PARAMETERS_FILE.open("w", encoding="utf-8") as f:
        f.write(f"gNa_bar_recovered = {result.x[0]:.12g}\n")
        f.write(f"gK_bar_recovered = {result.x[1]:.12g}\n")
        f.write(f"gL_bar_recovered = {result.x[2]:.12g}\n")
    print(f"Saved fitted conductances to {PARAMETERS_FILE.name}")


if __name__ == "__main__":
    main()
