# Scroll Wave Simulation in 3D Cardiac Tissue

This repository contains Python code for simulating **scroll waves** in a 2D cardiac tissue model using monodomain model combines with equations (FitzHugh–Nagumo type) and finite element methods.

## Files

### `scroll_data.py`
- Contains the **precomputed matrices** `M_matrix` and `K_matrix` used for spatial discretization of the 2D domain.
- These matrices represent mass and stiffness for the finite element method.
- **Purpose:** Provides the spatial framework for the simulation.

### `scroll_simulation.py`
- Defines the **`SprialSimulation` class**, which:
  - Initializes simulation parameters (conductivity, capacitance, surface-to-volume ratio).
  - Computes stable time steps.
  - Sets up the reaction-diffusion system.
  - Updates the state variables using Forward Euler integration.
- **Purpose:** Encapsulates the simulation logic and finite element computations.

### `scroll_wave_main.py`
- The **main script** that runs the simulation and generates the animation.
- Key steps:
  1. Imports `scroll_data` and `scroll_simulation`.
  2. Initializes the simulation object with parameters.
  3. Integrates the system over time.
  4. Stores snapshots of the transmembrane potential for visualization.
  5. Creates and optionally saves an animated MP4 using Matplotlib.
- **Purpose:** Entry point for running the spiral wave simulation and visualizing results.

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scipy` (if used in `scroll_simulation.py`)
- FFmpeg (optional, for saving animations as MP4)

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Yoseph-Tereda/cardiac-electrophysilogy.git
cd cardiac-electrophysilogy


## Documentation & Thesis
- For a complete and detailed description of the mathematical model, numerical methods,     finite element formulation, parameter choices, and validation results, see my master’s thesis:

[**Fast Simulations of Models of Cardiac Electrophysiology – Master Thesis (PDF)**](https://harvest.usask.ca/server/api/core/bitstreams/ffd11c9f-73ac-4afc-a722-25aa67f042ec/content)

The thesis includes:

- Full derivation of the monodomain model 
- Finite element spatial discretization ( mass and stiffness matrices)  
- Stability analysis and time-step estimation  
- Details on the Forward Euler integration scheme  
- Examples of simulation results and analysis of spiral‑wave behavior  
