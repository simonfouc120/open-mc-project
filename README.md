# open-mc-project

**open-mc-project** is a modular Python project for simulating nuclear reactor physics using [OpenMC](https://openmc.org/). It provides tools for modeling reactor geometries, defining materials, running criticality calculations, and analyzing results with custom utilities for pre- and post-processing. The project is organized for extensibility and reproducibility, supporting studies such as fission rate analysis and gamma camera simulations.


## Project Structure



---

## Key Components

- **Material and Geometry Definitions**
  - [`parameters/parameters_materials.py`](parameters/parameters_materials.py): Contains material definitions such as `FUEL_MATERIAL`, `HELIUM_MATERIAL`, etc.
  - [`src/models/model_hexagon_lattice_fuel.py`](src/models/model_hexagon_lattice_fuel.py): Defines the hexagonal lattice geometry and main cell.

- **Cross Section Data**
  - [`parameters/parameters_paths.py`](parameters/parameters_paths.py): Sets the path to the cross section XML file.
  - [`lib/cross_sections/cross_sections.xml`](lib/cross_sections/cross_sections.xml): Main cross section data for OpenMC.

- **Simulation Scripts**
  - [`src/studies/lattice_fuel/core.py`](src/studies/lattice_fuel/core.py): Main entry point for lattice fuel simulations, including geometry export, plotting, and criticality calculations.
  - [`src/studies/lattice_fuel/radiopro_mini_core/radiopro_core.py`](src/studies/lattice_fuel/radiopro_mini_core/radiopro_core.py): Specialized script for radioprotection studies.

- **Utilities**
  - Pre-processing: Geometry creation, mesh tally setup, and result cleanup.
  - Post-processing: Loading and analyzing tally results.

---

## Getting Started

### Prerequisites

- Python 3.8+
- [OpenMC](https://docs.openmc.org/en/stable/)
- NumPy, Matplotlib, Pillow