# Mini-Projects — Numerical Modelling and Scientific Computing

This repository contains short, self-contained computational exercises undertaken as part of my independent study of mathematical modelling, numerical methods and scientific computing. The projects connect mathematical equations with numerical implementations in Python, with particular applications to computational neuroscience.

Each project has its own folder, source code, explanatory Jupyter notebook and README. The focus is on understanding the mathematical model, implementing an appropriate numerical method, and checking or visualising the resulting behaviour. This is an ongoing collection and will be expanded as I explore further topics.

## Projects

| Project | Main question | Methods and tools |
| --- | --- | --- |
| [Hodgkin–Huxley Parameter Estimation](HH_Parameter_estimation/) | Can ion-channel conductances be estimated from a noisy synthetic membrane-voltage trace? | Coupled nonlinear ODEs, SciPy BDF integration, mean squared error, bounded differential evolution |
| [Synthetic Spike Generation](Synthetic_Spike_Generation/) | How can simplified extracellular-like recordings be generated with known spike times and added noise? | Waveform templates, random event generation, Gaussian noise, NumPy, pandas and Matplotlib |
| [Passive Dendrite Model](Passive_Dendrite_Model/) | How does a local voltage disturbance spread and decay along an idealised passive dendrite? | Cable PDE, finite differences, sparse matrices, Method of Lines, BDF/RK45, Green-function comparison |

### Hodgkin–Huxley parameter estimation

The [HH_param_estimation](HH_param_estimation/) project treats conductance recovery as an **inverse problem**. Synthetic voltage recordings are generated from the Hodgkin–Huxley equations, corrupted with Gaussian noise, and used to estimate the maximum sodium, potassium and leak conductances by minimising a voltage-fitting error. It includes forward simulation, optimisation and a comparison of the recovered and reference voltage traces.

[View the project README](HH_param_estimation/README.md) · [View the fit](HH_param_estimation/HH_parameter_fit.png)

### Synthetic spike generation

The [Synthetic_Spike_Generation](Synthetic_Spike_Generation/) project constructs a sampled, extracellular-like signal by combining three distinct spike-waveform templates, randomly placed events and Gaussian noise. The generator saves both the recording and known spike-onset annotations for possible future detection or classification exercises.

**Scope:** This project generates synthetic signals; it does not yet implement spike sorting.

[View the project README](Synthetic_Spike_Generation/README.md) · [View a signal preview](Synthetic_Spike_Generation/figures/signal_preview.png)

### Passive dendrite: the cable equation

The [Passive_Dendrite_Model](Passive_Dendrite_Model/) project numerically solves a one-dimensional passive cable equation. A finite-difference discretisation of the spatial derivative produces a sparse system of ODEs, which is integrated using SciPy. The results are visualised as voltage profiles and heatmaps, and compared with a Green-function reference for an infinite cable.

**Scope:** The numerical cable has fixed-voltage endpoints, while the analytical reference represents an infinite cable; agreement is assessed with that distinction in mind.

[View the project README](Passive_Dendrite_Model/README.md) · [View the numerical/analytical comparison](Passive_Dendrite_Model/figures/green_function_comparison.png)

## Repository structure

```text
Mini-Projects/
├── HH_param_estimation/
│   ├── README.md
│   ├── Vm_generator.py
│   ├── Optimizer.py
│   ├── Plotter.py
│   └── HH_Parameter_Estimation.ipynb
├── Synthetic_Spike_Generation/
│   ├── README.md
│   ├── generator.py
│   └── Synthetic_Spike_Generation.ipynb
├── Passive_Dendrite_Model/
│   ├── README.md
│   ├── main.py
│   ├── Passive_Dendrite_Model.ipynb
│   └── tests/
└── README.md
```

Each project folder also includes its own `requirements.txt` and, where applicable, example datasets, figures and other generated results. The tree above shows the principal files rather than every output.

## Requirements and setup

The projects use **Python**, primarily with:

- **NumPy:** numerical arrays and mathematical operations
- **SciPy:** ODE solvers, sparse linear algebra, special functions and optimisation
- **Matplotlib:** time-series plots, heatmaps and comparison figures
- **pandas:** tabular datasets and CSV files
- **Jupyter Notebook / JupyterLab:** annotated, interactive computational exercises

**Not every project requires every package.** Use the `requirements.txt` in the individual project folder rather than installing all scientific packages into one existing environment. In particular, **JAX is not required for any of the three projects currently in this repository**; it is used in other dynamical-systems work, not here.

For a project, create a fresh virtual environment and install its dependencies. For example, from the `Mini-Projects` repository root:

```bash
cd HH_param_estimation
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of the `source` command. Repeat with a separate environment for another project if its dependencies differ.

## Running the projects

Run the commands below **from the corresponding project directory** after installing that project's dependencies.

**Hodgkin–Huxley parameter estimation:**

```bash
python Vm_generator.py
python Optimizer.py
python Plotter.py
```

Run these in order: the generator creates the voltage dataset, the optimiser writes the recovered conductances, and the plotter compares the traces. Optimisation is computationally more expensive than plotting or data generation.

**Synthetic spike generation:**

```bash
python generator.py --no-show
```

This generates the recording, spike-onset annotations and example figures. Omit `--no-show` to display the figures interactively.

**Passive dendrite model:**

```bash
python main.py
python -m unittest discover -s tests -v
```

The script regenerates numerical results and plots; the tests check aspects of its numerical implementation. Use `python main.py --help` for adjustable simulation settings.

The notebooks can be opened through Jupyter from their respective project directories. Full instructions, parameter definitions and project-specific limitations are given in each folder's README.

## Notes on reproducibility and scope

- The example data in the Hodgkin–Huxley and spike-generation exercises are **synthetic**, not experimental neuronal recordings.
- Random seeds are fixed in the provided default examples so generated observations can be reproduced.
- Saved plots make the numerical results accessible on GitHub; binary NumPy files, where present, can be loaded using Python or regenerated from source.
- The implementations are educational numerical experiments, not validated biomedical tools. Their modelling assumptions and limitations are documented in the project-level READMEs.

## Learning goals

The collection is intended to strengthen practical understanding of **differential equations, PDE discretisation, numerical integration, stochastic data generation, inverse problems, optimisation and reproducible scientific programming**. Additional projects will be added as this study progresses.
