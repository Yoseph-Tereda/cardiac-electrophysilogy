# Spiral Wave Simulation in 2D Cardiac Tissue

This repository contains Python code for simulating **spiral waves** in a 2D cardiac tissue model using reaction-diffusion equations (FitzHugh–Nagumo type) and finite element methods.

## Files

### `spiral_data.py`
- Contains the **precomputed matrices** `M_matrix` and `K_matrix` used for spatial discretization of the 2D domain.
- These matrices represent mass and stiffness for the finite element method.
- **Purpose:** Provides the spatial framework for the simulation.

### `spiral_simulation.py`
- Defines the **`spiralsimulation` class**, which:
  - Initializes simulation parameters (conductivity, capacitance, surface-to-volume ratio).
  - Computes stable time steps.
  - Sets up the reaction-diffusion system.
  - Updates the state variables using Forward Euler integration.
- **Purpose:** Encapsulates the simulation logic and finite element computations.

### `spiral_wave_main.py`
- The **main script** that runs the simulation and generates the animation.
- Key steps:
  1. Imports `spiral_data` and `spiral_simulation`.
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
  - `scipy` (if used in `spiral_simulation.py`)
- FFmpeg (optional, for saving animations as MP4)

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Yoseph-Tereda/spiral-wave-simulation.git
cd spiral-wave-simulation
