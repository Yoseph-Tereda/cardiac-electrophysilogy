Cardiac Electrophysiology Simulations  
Spiral Waves in 2D and Scroll Waves in 3D Using the Monodomain Model Combined FitzHugh–Nagumo model

This repository contains Python implementations for simulating spiral waves (2D) and scroll waves (3D) in cardiac tissue using the monodomain model coupled with a FitzHugh–Nagumo–type ionic model. The simulations employ finite element spatial discretization and Forward Euler time integration.

The project is divided into two main modules:
- spiral_wave/ – 2D spiral-wave simulation  
- scroll_wave/ – 3D scroll-wave simulation  

Both modules follow the same numerical framework but differ in spatial dimensionality and visualization.

--------------------------------------------------------------------------------

1. Repository Structure

cardiac-electrophysiology/
│
├── spiral_wave/        # 2D spiral wave simulation
│   ├── spiral_data.py
│   ├── spiral_simulation.py
│   ├── spiral_wave_main.py
│   └── README.md
│
├── scroll_wave/        # 3D scroll wave simulation
│   ├── scroll_data.py
│   ├── scroll_simulation.py
│   ├── scroll_wave_main.py
│   └── README.md
│
└── README.md           # Combined README

--------------------------------------------------------------------------------

2. Overview of the Mathematical & Numerical Model

Both the 2D and 3D simulations solve the monodomain reaction-diffusion equation with a simple FitzHugh–Nagumo–type ionic model:


Spatial discretization uses finite element methods, where the stiffness (K_matrix) and mass (M_matrix) matrices are precomputed and stored in the *_data.py files.

Time integration: Forward Euler with stable timestep estimation.

--------------------------------------------------------------------------------

3. Spiral Wave Simulation (2D)

Location: spiral_wave/

Files:
- spiral_data.py – Precomputed FE mass and stiffness matrices  
- spiral_simulation.py – SpiralSimulation class  
- spiral_wave_main.py – Runs the 2D simulation and generates animations

--------------------------------------------------------------------------------

4. Scroll Wave Simulation (3D)

Location: scroll_wave/

Files:
- scroll_data.py – Precomputed FE mass and stiffness matrices (3D)
- scroll_simulation.py – ScrollSimulation class
- scroll_wave_main.py – Runs the 3D simulation and generates volume visualizations

--------------------------------------------------------------------------------

5. Requirements

- Python 3.x  
- numpy  
- matplotlib  
- scipy  
- FFmpeg (optional for MP4 export)

Install:

pip install numpy matplotlib scipy

--------------------------------------------------------------------------------

6. How to Run

Clone repo:

git clone https://github.com/Yoseph-Tereda/cardiac-electrophysilogy.git
cd cardiac-electrophysilogy

Run 2D:

cd spiral_wave
python spiral_wave_main.py

Run 3D:

cd scroll_wave
python scroll_wave_main.py

--------------------------------------------------------------------------------

7. Documentation & Thesis

Full derivations and explanations are in the master’s thesis:

Fast Simulations of Models of Cardiac Electrophysiology – Master Thesis (PDF)
https://harvest.usask.ca/server/api/core/bitstreams/ffd11c9f-73ac-4afc-a722-25aa67f042ec/content

Includes:
- Monodomain derivation  
- FEM discretization  
- Euler scheme  
- Stability analysis  
- Spiral & scroll-wave examples  



