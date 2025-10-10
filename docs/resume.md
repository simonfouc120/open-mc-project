# Reactor Study — Project Summary

Location: `/Users/simonfoucambert/Documents/projects/open-mc-project/src/studies/reactor_study`

This document summarizes the purpose, structure, experiments, and results of the reactor_study folder. Replace placeholder filenames and images with actual items from the folder.

## 1. Purpose
- Investigate reactor behavior (design, kinetics, control or simulation) for the Open MC Project.
- Run experiments, produce plots and derive conclusions to inform next development steps.

## 2. High-level goals
- Implement reactor models and numerical solvers.
- Compare model variants and parameter sets.
- Visualize transient and steady-state behavior.
- Document reproducible experiments.

## 3. Expected folder structure (adjust to actual contents)
- src/
    - models/                 — reactor model implementations (e.g., `cstr.py`, `pfr.py`)
    - solvers/                — numerical solvers or wrappers (e.g., `rk4.py`, `scipy_wrapper.py`)
    - experiments/            — experiment scripts that run parameter sweeps (`run_experiment.py`)
    - analysis/               — plotting and metrics (`plot_results.py`, `metrics.py`)
    - data/                   — raw and processed results (`.csv`, `.npz`)
    - README.md               — short instructions
- figures/                  — generated figures referenced below
- notebooks/                — exploratory analysis (optional)

## 4. Key files to document (examples)
- models/reactor.py — core reactor class with state, parameters, and step function.
- solvers/integrator.py — time integration utilities and tolerances.
- experiments/sweep_temperature.py — runs a parametric sweep over temperature.
- analysis/plot_transient.py — produces transient response plots.
- tests/test_models.py — unit tests for mass/energy balances.

## 5. How to run (example)
1. Create a venv and install requirements:
     ```
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```
2. Run an experiment:
     ```
     python src/studies/reactor_study/experiments/run_experiment.py --config configs/exp1.yaml
     ```
3. Generate figures:
     ```
     python src/studies/reactor_study/analysis/plot_results.py --input data/exp1_results.npz --out figures/exp1.png
     ```

## 6. Figures (placeholders — replace with real outputs)
- Architecture / data flow
    ![Architecture diagram](figures/architecture.png)
    Caption: High-level modules and data flow between models, solvers, experiments and analysis.

- Reactor schematic
    ![Reactor schematic](figures/reactor_schematic.png)
    Caption: Diagram of the reactor(s) studied (CSTR/PFR), inlets/outlets and control points.

- Sample transient response
    ![Transient response](figures/transient_response.png)
    Caption: State variables (temperature, concentration) vs time for selected parameter set.

- Parameter sweep results
    ![Sweep results](figures/param_sweep.png)
    Caption: Final steady-state values as a function of the swept parameter (e.g., feed rate, temperature).

If images are not yet generated, run experiments and save plots to `figures/` with the filenames above.

## 7. Typical explanations to include for each figure
- What was varied (parameter, initial condition).
- Simulation settings (solver, tolerances, time horizon).
- Observed behaviour (stability, oscillations, bifurcation).
- Quantitative metrics (settling time, overshoot, steady-state error).

Example caption with explanation:
- Transient response for T_feed = 350 K, solver = RK4, dt = 0.01 s. The system shows a damped oscillation settling to a steady state at t ≈ 120 s. Increasing feed temperature shifts steady-state concentration upward by ~12%.

## 8. Results & conclusions (template)
- Key findings (short bullets).
    - Finding 1: e.g., "Model A is stable under nominal conditions; Model B shows limit cycles."
    - Finding 2: e.g., "Feed temperature strongly affects conversion; critical threshold at 330 K."
- Uncertainties and sensitivity directions.
- Recommended next steps (refinements, validation experiments, parameter estimation).

## 9. Notes for documentation completion
- Populate actual file list by running: `ls -R src/studies/reactor_study`.
- Capture exact commands used to generate each figure.
- Embed small code snippets or equations for the core model to make the doc self-contained.
- Add links to notebooks or tests that reproduce results.

## 10. Quick checklist before finalizing resume doc
- [ ] Replace placeholder images with generated PNG/SVG files.
- [ ] Fill in actual filenames and commands.
- [ ] Add one paragraph per figure explaining method and interpretation.
- [ ] Add provenance (date, author, runtime environment).

End of summary.