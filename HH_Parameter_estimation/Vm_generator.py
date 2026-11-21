"""Generate a reproducible synthetic voltage recording from the Hodgkin–Huxley model.

Run `python Vm_generator.py` to create Generated_Vm.csv in this directory.
Units: time (ms), voltage (mV), current density (uA/cm^2),
conductance density (mS/cm^2), capacitance density (uF/cm^2).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "Generated_Vm.csv"

C_m = 1.0
E_Na = 50.0
E_K = -77.0
E_L = -54.387

# Ground-truth maximum ion-channel conductances, in mS/cm^2.
TRUE_G_NA = 120.0
TRUE_G_K = 36.0
TRUE_G_L = 0.3

# Same protocol and solver settings are used for data generation and fitting.
t_span = (0.0, 80.0)
t_eval = np.linspace(*t_span, 1600)
y0 = [-65.0, 0.317, 0.05, 0.6]  # V, n, m, h
NOISE_STD_MV = 1.0
RANDOM_SEED = 42


def vtrap(x, scale):
    """Return x / (1 - exp(-x/scale)), including its limit at x = 0."""
    if abs(x) < 1e-8:
        return float(scale)
    return x / (-np.expm1(-x / scale))


def alpha_n(V):
    return 0.01 * vtrap(V + 55.0, 10.0)


def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def alpha_m(V):
    return 0.1 * vtrap(V + 40.0, 10.0)


def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def I_inj(t):
    """Apply 10 uA/cm^2 from 10 ms through 65 ms."""
    return 10.0 if 10.0 <= t <= 65.0 else 0.0


def hodgkin_huxley(t, y, g_Na, g_K, g_L):
    """Four coupled HH equations for (V, n, m, h)."""
    V, n, m, h = y
    dn_dt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dm_dt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh_dt = alpha_h(V) * (1 - h) - beta_h(V) * h

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)
    dV_dt = (I_inj(t) - I_Na - I_K - I_L) / C_m
    return [dV_dt, dn_dt, dm_dt, dh_dt]


def simulate_hh(conductances, times=None):
    """Integrate the HH model with a stiff BDF method; fail on solver errors."""
    if times is None:
        times = t_eval
    sol = solve_ivp(
        hodgkin_huxley, t_span, y0, args=tuple(conductances),
        t_eval=times, method="BDF", rtol=1e-6, atol=1e-8,
        max_step=0.5,  # resolve the on/off current pulse
    )
    if not sol.success or len(sol.t) != len(times):
        raise RuntimeError(f"Hodgkin–Huxley integration failed: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Hodgkin–Huxley integration returned non-finite values")
    return sol


def main():
    sol = simulate_hh((TRUE_G_NA, TRUE_G_K, TRUE_G_L))
    V_true = sol.y[0]
    rng = np.random.default_rng(RANDOM_SEED)
    V_noisy = V_true + rng.normal(0.0, NOISE_STD_MV, V_true.shape)

    pd.DataFrame({
        "Time_step": sol.t, "True_Vm": V_true, "Noisy_Vm": V_noisy
    }).to_csv(DATA_FILE, index=False)
    print(f"Saved {len(sol.t)} synthetic voltage samples to {DATA_FILE.name}")
    print(f"Noise: Gaussian(0, {NOISE_STD_MV} mV), seed={RANDOM_SEED}")

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, V_noisy, color="0.3", alpha=0.45, label="Synthetic noisy observations")
    plt.plot(sol.t, V_true, color="tab:blue", linewidth=1.6, label="Ground truth")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Synthetic Hodgkin–Huxley recording")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "HH_synthetic_data.png", dpi=170)
    plt.show()


if __name__ == "__main__":
    main()
