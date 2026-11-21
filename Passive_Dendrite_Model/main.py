"""Passive dendritic cable equation: Method of Lines and Green-function check.

The dependent variable V is voltage *relative to resting potential* (mV).
The forcing I is an effective voltage-equivalent input (mV), not a
biophysical current density in microamperes per square centimetre.

Run from this directory with `python main.py`; use `python main.py --help`
for parameter controls. Figures and small datasets are written to disk.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.special import erf


@dataclass(frozen=True)
class CableParameters:
    length_um: float = 1000.0
    n_points: int = 201
    tau_ms: float = 10.0
    lambda_um: float = 200.0
    pulse_start_ms: float = 10.0
    pulse_end_ms: float = 40.0
    input_mV: float = 100.0
    duration_ms: float = 100.0
    time_samples: int = 501
    solver: str = "BDF"

    def __post_init__(self) -> None:
        if self.length_um <= 0 or self.tau_ms <= 0 or self.lambda_um <= 0:
            raise ValueError("Length, tau and lambda must be strictly positive.")
        if self.n_points < 5 or self.n_points % 2 != 1:
            raise ValueError("n_points must be an odd integer >= 5 (to centre the input).")
        if self.time_samples < 4:
            raise ValueError("time_samples must be >= 4.")
        if not (0 < self.pulse_start_ms < self.pulse_end_ms < self.duration_ms):
            raise ValueError("Require 0 < pulse_start < pulse_end < duration.")
        if self.solver not in {"BDF", "RK45"}:
            raise ValueError("solver must be BDF or RK45.")

    @property
    def dx_um(self) -> float:
        return self.length_um / (self.n_points - 1)

    @property
    def diffusivity_um2_per_ms(self) -> float:
        return self.lambda_um ** 2 / self.tau_ms


@dataclass
class CableSolution:
    params: CableParameters
    x_um: np.ndarray
    t_ms: np.ndarray
    voltage_mV: np.ndarray  # Shape (space, time)


def build_operator(params: CableParameters):
    """Sparse semidiscrete linear operator with V=0 at cable endpoints."""
    n = params.n_points
    diagonals = [np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1)]
    lap = diags(diagonals, offsets=[-1, 0, 1], format="lil")
    lap[0, :] = 0.0
    lap[-1, :] = 0.0
    op = (params.diffusivity_um2_per_ms / params.dx_um ** 2) * lap.tocsr()
    op = op.tolil()
    op.setdiag(op.diagonal() - 1.0 / params.tau_ms)
    # Voltage deviation is clamped at zero at the endpoints (Dirichlet BC).
    op[0, :] = 0.0
    op[-1, :] = 0.0
    return op.tocsr()


def simulate(params: CableParameters = CableParameters()) -> CableSolution:
    """Integrate piecewise across current-pulse changes to avoid step errors."""
    x = np.linspace(0.0, params.length_um, params.n_points)
    t_requested = np.linspace(0.0, params.duration_ms, params.time_samples)
    times = np.unique(np.r_[t_requested, params.pulse_start_ms, params.pulse_end_ms])

    op = build_operator(params)
    source = np.zeros(params.n_points)
    source[params.n_points // 2] = params.input_mV / params.tau_ms
    y = np.zeros(params.n_points)
    t_chunks, v_chunks = [], []
    segments = [
        (0.0, params.pulse_start_ms, False),
        (params.pulse_start_ms, params.pulse_end_ms, True),
        (params.pulse_end_ms, params.duration_ms, False),
    ]

    for i, (t0, t1, on) in enumerate(segments):
        t_eval = times[(times >= t0 - 1e-10) & (times <= t1 + 1e-10)]
        if t_eval.size < 2 or abs(t_eval[0] - t0) > 1e-8 or abs(t_eval[-1] - t1) > 1e-8:
            raise RuntimeError("Missing pulse boundary in simulation time grid.")
        forcing = source if on else 0.0

        kwargs = {"method": params.solver, "t_eval": t_eval, "rtol": 1e-7, "atol": 1e-9}
        if params.solver == "BDF":
            kwargs["jac"] = op
        result = solve_ivp(lambda t, v: op @ v + forcing, (t0, t1), y, **kwargs)
        if not result.success:
            raise RuntimeError(f"Numerical solution failed: {result.message}")
        y = result.y[:, -1].copy()
        start = 0 if i == 0 else 1  # Discard only duplicated boundary samples.
        t_chunks.append(result.t[start:])
        v_chunks.append(result.y[:, start:])

    t = np.concatenate(t_chunks)
    voltage = np.hstack(v_chunks)
    if voltage.shape != (params.n_points, t.size):
        raise RuntimeError("Unexpected space/time array shape.")
    if not np.all(np.isfinite(voltage)):
        raise RuntimeError("Non-finite voltage in computed solution.")
    if not np.allclose(voltage[[0, -1], :], 0.0, atol=1e-10):
        raise RuntimeError("Endpoint Dirichlet conditions were not preserved.")
    return CableSolution(params, x, t, voltage)


def infinite_cable_reference(
    x_um: np.ndarray, t_ms: np.ndarray, params: CableParameters,
    quadrature_order: int = 80,
) -> np.ndarray:
    """Infinite-line Green function for an input uniform over one cell.

    For tau V_t = lambda^2 V_xx - V + I, the infinite-line fundamental
    solution is exp(-u/tau) * exp(-(x-y)^2/(4Du)) / sqrt(4*pi*D*u).
    The y integral over the stimulated cell is evaluated with error
    functions, and the remaining time integral with Gauss-Legendre nodes.
    This is an infinite-domain reference, not an exact finite-cable solution.
    """
    x = np.atleast_1d(np.asarray(x_um, dtype=float))[:, None]
    t = np.atleast_1d(np.asarray(t_ms, dtype=float))
    output = np.zeros((x.shape[0], t.size))
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    diffusion = params.diffusivity_um2_per_ms
    half_cell = params.dx_um / 2.0
    centre = params.length_um / 2.0
    left, right = centre - half_cell, centre + half_cell

    for j, time in enumerate(t):
        if time <= params.pulse_start_ms:
            continue
        lower = max(0.0, time - params.pulse_end_ms)
        upper = time - params.pulse_start_ms
        if upper <= lower:
            continue
        half_width = (upper - lower) / 2.0
        u = (lower + upper) / 2.0 + half_width * nodes
        scale = np.sqrt(4.0 * diffusion * u)[None, :]
        cell_kernel = 0.5 * (erf((x - left) / scale) - erf((x - right) / scale))
        time_kernel = np.exp(-u / params.tau_ms)[None, :]
        output[:, j] = (params.input_mV / params.tau_ms) * half_width * (
            (cell_kernel * time_kernel) @ weights
        )
    return output


def calculate_validation(solution: CableSolution):
    """Compare infinite- and finite-domain solutions near the injection site."""
    p = solution.params
    centre = p.n_points // 2
    analytic_center = infinite_cable_reference(
        solution.x_um[[centre]], solution.t_ms, p
    )[0]
    profile_index = int(np.argmin(np.abs(solution.t_ms - p.pulse_end_ms)))
    analytic_profile = infinite_cable_reference(
        solution.x_um, solution.t_ms[[profile_index]], p
    )[:, 0]
    interior_mask = np.abs(solution.x_um - p.length_um / 2.0) <= p.lambda_um
    error_center = solution.voltage_mV[centre] - analytic_center
    error_profile = solution.voltage_mV[:, profile_index] - analytic_profile
    metrics = {
        "solver": p.solver,
        "boundary_conditions": "V(0,t)=V(L,t)=0 (Dirichlet)",
        "reference_domain": "infinite cable with a one-cell-wide, uniform pulse",
        "centre_temporal_rmse_mV": float(np.sqrt(np.mean(error_center ** 2))),
        "centre_temporal_max_abs_error_mV": float(np.max(np.abs(error_center))),
        "central_1lambda_profile_rmse_mV": float(
            np.sqrt(np.mean(error_profile[interior_mask] ** 2))
        ),
        "peak_voltage_mV": float(np.max(solution.voltage_mV)),
        "profile_time_ms": float(solution.t_ms[profile_index]),
        "parameters": asdict(p),
    }
    return metrics, analytic_center, analytic_profile


def save_outputs(solution: CableSolution, output_dir: Path, show: bool = False) -> dict:
    """Save three figures, numerical arrays, and Green-function comparison metrics."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    p = solution.params
    centre = p.n_points // 2
    metrics, analytic_center, analytic_profile = calculate_validation(solution)
    pulse = (p.pulse_start_ms, p.pulse_end_ms)
    idx = int(np.argmin(np.abs(solution.t_ms - p.pulse_end_ms)))

    # Use pcolormesh to respect spatial and temporal data coordinates.
    fig, ax = plt.subplots(figsize=(9.3, 5.3))
    image = ax.pcolormesh(solution.t_ms, solution.x_um,
                          solution.voltage_mV, cmap="inferno", shading="auto")
    fig.colorbar(image, ax=ax, label="Voltage deviation from rest (mV)")
    ax.set(xlabel="Time (ms)", ylabel="Position along cable (µm)",
           title="Passive cable: spatial and temporal voltage dynamics")
    ax.axvline(pulse[0], linestyle="--", alpha=0.5, color="white")
    ax.axvline(pulse[1], linestyle="--", alpha=0.5, color="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "voltage_heatmap.png", dpi=160)
    if not show: plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    a1.plot(solution.t_ms, solution.voltage_mV[centre], color="C0", lw=1.8)
    a1.axvspan(*pulse, alpha=0.15, color="C1", label="Input on")
    a1.set(xlabel="Time (ms)", ylabel="Voltage deviation (mV)",
           title=f"Temporal response at x = {p.length_um/2:g} µm")
    a1.legend(); a1.grid(alpha=0.25)
    a2.plot(solution.x_um, solution.voltage_mV[:, idx], color="C1", lw=1.8)
    a2.axvline(p.length_um/2 - p.lambda_um, color="gray", ls="--", alpha=0.7)
    a2.axvline(p.length_um/2 + p.lambda_um, color="gray", ls="--", alpha=0.7,
               label="±λ from centre (reference)")
    a2.set(xlabel="Position (µm)", ylabel="Voltage deviation (mV)",
           title=f"Spatial response at t = {solution.t_ms[idx]:g} ms")
    a2.legend(); a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_spatial_profiles.png", dpi=160)
    if not show: plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    a1.plot(solution.t_ms, solution.voltage_mV[centre], color="C0", label="Finite difference (Dirichlet)")
    a1.plot(solution.t_ms, analytic_center, "--", color="C3", label="Infinite-cable Green reference")
    a1.axvspan(*pulse, alpha=0.1, color="C1")
    a1.set(xlabel="Time (ms)", ylabel="Voltage deviation (mV)",
           title="Centre: numerical vs. Green-function reference")
    a1.grid(alpha=0.2); a1.legend(fontsize=8)
    a2.plot(solution.x_um, solution.voltage_mV[:, idx], color="C0", label="Finite difference (Dirichlet)")
    a2.plot(solution.x_um, analytic_profile, "--", color="C3", label="Infinite-cable Green reference")
    a2.axvspan(p.length_um/2 - p.lambda_um, p.length_um/2 + p.lambda_um,
               color="gray", alpha=0.08, label="Central comparison region")
    a2.set(xlabel="Position (µm)", ylabel="Voltage deviation (mV)",
           title=f"Spatial profile at t = {solution.t_ms[idx]:g} ms")
    a2.grid(alpha=0.2); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "green_function_comparison.png", dpi=160)
    if not show: plt.close(fig)

    np.savez_compressed(
        data_dir / "simulation_results.npz",
        x_um=solution.x_um, t_ms=solution.t_ms,
        voltage_mV=solution.voltage_mV,
        reference_center_mV=analytic_center,
        reference_profile_at_pulse_end_mV=analytic_profile,
    )
    (data_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if show:
        plt.show()
        plt.close("all")
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=1000.0, help="Cable length (µm)")
    parser.add_argument("--points", type=int, default=201, help="Odd number of spatial grid points")
    parser.add_argument("--tau", type=float, default=10.0, help="Membrane time constant (ms)")
    parser.add_argument("--lambda-um", type=float, default=200.0, help="Cable space constant (µm)")
    parser.add_argument("--amplitude", type=float, default=100.0, help="Effective injected voltage term (mV)")
    parser.add_argument("--pulse-start", type=float, default=10.0, help="Pulse begins (ms)")
    parser.add_argument("--pulse-end", type=float, default=40.0, help="Pulse ends (ms)")
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation ends (ms)")
    parser.add_argument("--time-samples", type=int, default=501, help="Approximate number of time samples")
    parser.add_argument("--method", choices=["BDF", "RK45"], default="BDF")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--show", action="store_true", help="Also display figures interactively")
    args = parser.parse_args()

    params = CableParameters(
        length_um=args.length, n_points=args.points, tau_ms=args.tau,
        lambda_um=args.lambda_um, input_mV=args.amplitude,
        pulse_start_ms=args.pulse_start, pulse_end_ms=args.pulse_end,
        duration_ms=args.duration, time_samples=args.time_samples, solver=args.method,
    )
    solution = simulate(params)
    metrics = save_outputs(solution, args.output_dir, show=args.show)
    print(f"Generated {solution.voltage_mV.shape[0]} spatial × {solution.voltage_mV.shape[1]} temporal values")
    print(f"Peak voltage deviation: {metrics['peak_voltage_mV']:.3f} mV")
    print(f"Centre Green-reference RMSE: {metrics['centre_temporal_rmse_mV']:.5f} mV")
    print(f"Central ±λ spatial RMSE: {metrics['central_1lambda_profile_rmse_mV']:.5f} mV")
    print(f"Saved figures and data to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    cli()
