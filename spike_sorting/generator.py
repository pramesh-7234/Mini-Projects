"""Generate a synthetic extracellular-like recording from three spike templates.

This is a toy signal-generation model intended for spike-detection experiments.
It does not simulate extracellular biophysics or perform spike sorting.

Run:
    python generator.py
    python generator.py --no-show
    python generator.py --duration 10 --seed 123 --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FS = 20_000                 # samples per second (Hz)
DEFAULT_DURATION_S = 30.0          # seconds
DEFAULT_WINDOW_MS = 3.0           # template duration, milliseconds
DEFAULT_FIRING_RATE_HZ = 15.0     # mean rate for each synthetic unit
DEFAULT_NOISE_STD = 0.4           # arbitrary voltage units
DEFAULT_SEED = 42                 # random seed for reproducibility


def make_templates(fs: int = DEFAULT_FS, window_ms: float = DEFAULT_WINDOW_MS) -> list[np.ndarray]:
    """Return three normalised, amplitude-scaled difference-of-Gaussian waveforms."""
    n_points = int(round(fs * window_ms / 1_000))
    if n_points < 3:
        raise ValueError("Template needs at least three samples; increase fs or window_ms.")

    t = np.linspace(-1.0, 1.0, n_points)

    template_1 = -np.exp(-(t**2) / 0.1) + 0.5 * np.exp(-((t - 0.2) ** 2) / 0.2)
    template_2 = -0.8 * np.exp(-(t**2) / 0.2) + 0.3 * np.exp(-((t - 0.3) ** 2) / 0.4)
    template_3 = 0.6 * np.exp(-((t + 0.2) ** 2) / 0.1) - np.exp(-((t - 0.2) ** 2) / 0.1)

    amplitudes = (1.5, 0.8, 1.2)
    templates = []
    for waveform, amplitude in zip((template_1, template_2, template_3), amplitudes):
        templates.append(amplitude * waveform / np.max(np.abs(waveform)))
    return templates


def generate_recording(
    fs: int = DEFAULT_FS,
    duration_s: float = DEFAULT_DURATION_S,
    window_ms: float = DEFAULT_WINDOW_MS,
    firing_rate_hz: float = DEFAULT_FIRING_RATE_HZ,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Generate waveform samples, onset annotations, and templates.

    For each unit, the spike count is Poisson-distributed. Conditional on that
    count, onset samples are selected uniformly without replacement. This
    approximates a homogeneous Poisson spike train on a discrete sampling grid.
    Overlapping templates from different units sum linearly.
    """
    if fs <= 0 or duration_s <= 0 or window_ms <= 0:
        raise ValueError("Sample rate, duration, and template width must be positive.")
    if firing_rate_hz < 0 or noise_std < 0:
        raise ValueError("Firing rate and noise standard deviation cannot be negative.")

    rng = np.random.default_rng(seed)
    templates = make_templates(fs, window_ms)
    n_samples = int(round(fs * duration_s))
    template_len = len(templates[0])
    valid_start_count = n_samples - template_len + 1
    if valid_start_count <= 0:
        raise ValueError("Recording duration must exceed the spike-template duration.")

    clean_signal = np.zeros(n_samples, dtype=float)
    events = []
    for unit_id, template in enumerate(templates, start=1):
        spike_count = int(rng.poisson(firing_rate_hz * duration_s))
        if spike_count > valid_start_count:
            raise ValueError("Too many spikes for distinct onset samples. Lower firing rate.")
        start_samples = np.sort(rng.choice(valid_start_count, size=spike_count, replace=False))
        for start_sample in start_samples:
            start_sample = int(start_sample)
            clean_signal[start_sample : start_sample + template_len] += template
            events.append((unit_id, start_sample, start_sample / fs))

    noise = rng.normal(0.0, noise_std, size=n_samples)
    samples = pd.DataFrame(
        {
            "time": np.arange(n_samples, dtype=float) / fs,  # seconds
            "v_gen": clean_signal,                           # arbitrary units
            "v_noisy": clean_signal + noise,                 # arbitrary units
        }
    )
    annotations = pd.DataFrame(events, columns=["unit_id", "start_sample", "start_time_s"])
    annotations = annotations.sort_values(["start_sample", "unit_id"]).reset_index(drop=True)
    return samples, annotations, templates


def create_figures(
    samples: pd.DataFrame,
    annotations: pd.DataFrame,
    templates: list[np.ndarray],
    fs: int,
    output_dir: Path,
    preview_ms: float = 200.0,
) -> list[plt.Figure]:
    """Save waveform, recording-preview, and ground-truth raster figures."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    # Template shapes; their amplitudes are in arbitrary units.
    template_time_ms = np.arange(len(templates[0])) / fs * 1_000
    fig, ax = plt.subplots(figsize=(9, 4))
    for unit_id, template in enumerate(templates, start=1):
        ax.plot(template_time_ms, template, label=f"Unit {unit_id}")
    ax.set(xlabel="Time since template onset (ms)", ylabel="Amplitude (a.u.)",
           title="Synthetic spike templates")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "spike_templates.png", dpi=160)
    figures.append(fig)

    # Plot only a short window so individual spikes remain visible.
    n_preview = max(1, min(len(samples), int(round(fs * preview_ms / 1_000))))
    preview = samples.iloc[:n_preview]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(preview["time"].to_numpy() * 1_000, preview["v_noisy"],
            linewidth=0.8, label="Noisy signal")
    ax.plot(preview["time"].to_numpy() * 1_000, preview["v_gen"],
            linewidth=1.2, label="Noise-free signal")
    ax.set(xlabel="Time (ms)", ylabel="Amplitude (a.u.)",
           title=f"Synthetic recording — first {preview_ms:g} ms")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "signal_preview.png", dpi=160)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    for unit_id in (1, 2, 3):
        times = annotations.loc[annotations["unit_id"] == unit_id, "start_time_s"].to_numpy()
        ax.vlines(times, unit_id - 0.35, unit_id + 0.35, linewidth=0.6)
    ax.set(xlabel="Time (s)", ylabel="Synthetic unit", title="Ground-truth spike-onset raster",
           yticks=[1, 2, 3], ylim=(0.5, 3.5))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "ground_truth_raster.png", dpi=160)
    figures.append(fig)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_FS,
                        help="Samples per second (default: 20000)")
    parser.add_argument("--rate", type=float, default=DEFAULT_FIRING_RATE_HZ,
                        help="Mean firing rate per unit in Hz (default: 15)")
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD,
                        help="Gaussian noise standard deviation (default: 0.4)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Reproducible random seed (default: 42)")
    parser.add_argument("--preview-ms", type=float, default=200.0,
                        help="Length of plotted recording preview in ms")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory for generated CSVs and figures")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots without opening interactive windows")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, annotations, templates = generate_recording(
        fs=args.sample_rate,
        duration_s=args.duration,
        firing_rate_hz=args.rate,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    samples.to_csv(args.output_dir / "synthetic_spikes.csv", index=False)
    annotations.to_csv(args.output_dir / "ground_truth_spikes.csv", index=False)
    figures = create_figures(samples, annotations, templates, args.sample_rate,
                             args.output_dir, preview_ms=args.preview_ms)

    print(f"Saved {len(samples):,} samples ({args.duration:g} s at {args.sample_rate:,} Hz)")
    for unit_id in (1, 2, 3):
        count = int((annotations["unit_id"] == unit_id).sum())
        print(f"Unit {unit_id}: {count} spikes; realised rate {count / args.duration:.2f} Hz")
    print(f"Output directory: {args.output_dir.resolve()}")
    if not args.no_show:
        plt.show()
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
