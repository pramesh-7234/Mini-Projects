"""Compare noisy synthetic HH observations with the recovered and true voltage traces.

Run Vm_generator.py then Optimizer.py before running this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Vm_generator import simulate_hh

PROJECT_DIR = Path(__file__).resolve().parent


def load_conductances(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", maxsplit=1)
            values[name.strip()] = float(value.strip())
    keys = ["gNa_bar_recovered", "gK_bar_recovered", "gL_bar_recovered"]
    return np.array([values[key] for key in keys], dtype=float)


def main():
    fitted = load_conductances(PROJECT_DIR / "recovered_g_params.txt")
    data = pd.read_csv(PROJECT_DIR / "Generated_Vm.csv")
    times = data["Time_step"].to_numpy(dtype=float)
    observed = data["Noisy_Vm"].to_numpy(dtype=float)
    truth = data["True_Vm"].to_numpy(dtype=float)

    predicted = simulate_hh(fitted, times=times).y[0]
    mse = float(np.mean((observed - predicted)**2))

    plt.figure(figsize=(11, 6))
    plt.plot(times, observed, color="0.45", alpha=0.35, label="Synthetic noisy observations")
    plt.plot(times, truth, color="tab:blue", linewidth=1.5, linestyle="--", label="True voltage")
    plt.plot(times, predicted, color="tab:red", linewidth=1.5, label="Fitted model")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title(
        "Hodgkin–Huxley conductance fit\n"
        f"Na={fitted[0]:.2f}, K={fitted[1]:.2f}, L={fitted[2]:.3f} mS/cm² | "
        f"MSE={mse:.3f} mV²"
    )
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    path = PROJECT_DIR / "HH_parameter_fit.png"
    plt.savefig(path, dpi=200)
    print(f"Saved comparison plot to {path.name}; MSE={mse:.6f} mV²")
    plt.show()


if __name__ == "__main__":
    main()
