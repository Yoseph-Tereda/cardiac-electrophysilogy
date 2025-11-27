"""
3D Cardiac Scroll-Wave Simulation with Isosurface Visualization
Domain corrected to x, y, z ∈ [0, 2.5]

This script simulates 3D cardiac scroll waves using a simple reaction-diffusion
model and visualizes the evolving transmembrane potential as isosurfaces.

Dependencies:
- numpy
- scipy
- matplotlib
- scikit-image
- PythonProject.scroll_simulation (custom module)
- scroll_data (M_matrix, K_matrix)
"""

from scipy.sparse import kron, csc_matrix
from scipy.sparse.linalg import factorized
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import numpy as np
from scroll_simulation import ScrollSimulation
from scroll_data import M_matrix, K_matrix

# =========================================================
# MAIN SCRIPT
# =========================================================
if __name__ == '__main__':

    # -----------------------------
    # MODEL PARAMETERS
    # -----------------------------
    sigma_i = 0.1       # Intracellular conductivity
    cm = 1              # Membrane capacitance
    chi = 1000          # Surface-to-volume ratio
    Nxx = 51            # Grid points in x-direction
    Nyy = 51            # Grid points in y-direction
    Nzz = 51            # Grid points in z-direction

    # Correct physical domain size
    Lx = 2.5
    Ly = 2.5
    Lz = 2.5

    t_span = [0, 1000]  # Simulation time interval
    b = 1               # Placeholder parameter for model

    # Create ScrollSimulation object
    scw = ScrollSimulation(M_matrix, K_matrix, sigma_i, cm, chi,
                           Nxx, Nyy, Nzz, t_span, b)

    # -----------------------------
    # TIME STEP AND INITIAL CONDITIONS
    # -----------------------------
    dt = scw.step_size_FE()  # Stable Forward Euler step size
    v_current = scw.v_intial()
    s_current = scw.s_intial()

    # -----------------------------
    # SIMULATION PARAMETERS
    # -----------------------------
    num_intervals = scw.number_mesh_time()
    frame_skip = 100
    iso_level = 0.5
    thickness_levels = [iso_level - 0.03, iso_level, iso_level + 0.03]
    video_flag = True

    # Frame storage
    v_frames = []
    frame_numbers = []

    # -----------------------------
    # 3D FINITE ELEMENT KRONECKER MATRICES
    # -----------------------------
    BBB = kron(M_matrix, kron(M_matrix, M_matrix), format="csc")
    AAA = (
        kron(M_matrix, kron(M_matrix, K_matrix), format="csc") +
        kron(M_matrix, kron(K_matrix, M_matrix), format="csc") +
        kron(K_matrix, kron(M_matrix, M_matrix), format="csc")
    )

    # -----------------------------
    # SOLVER FACTORIZATION
    # -----------------------------
    solve_B = factorized(BBB)

    # -----------------------------
    # TIME INTEGRATION LOOP (Forward Euler)
    # -----------------------------
    for n in range(num_intervals - 1):

        # --- Reaction ---
        dv = v_current * (1 - v_current) * (v_current - 0.1) - s_current
        ds = 0.01 * (0.5 * v_current - s_current)

        v_next = v_current + dt * dv
        s_next = s_current + dt * ds

        # --- Diffusion ---
        diffusion_rhs = -1e-4 * (AAA @ v_next)
        diffusion = solve_B(diffusion_rhs)
        v_next += dt * diffusion

        # --- Save frames ---
        if n % frame_skip == 0:
            print(f"Saving frame at step {n}/{num_intervals-1}")
            v_frames.append(v_next.copy())
            frame_numbers.append(n)

        v_current = v_next
        s_current = s_next

    # -----------------------------
    # GRID FOR PLOTTING
    # -----------------------------
    x = np.linspace(0, Lx, Nxx)
    y = np.linspace(0, Ly, Nyy)
    z = np.linspace(0, Lz, Nzz)

    # Spacing for marching cubes
    dx = Lx / (Nxx - 1)
    dy = Ly / (Nyy - 1)
    dz = Lz / (Nzz - 1)
    spacing = (dx, dy, dz)

    # -----------------------------
    # FIGURE SETUP
    # -----------------------------
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_facecolor("white")
    ax.view_init(30, 45)

    # -----------------------------
    # VIDEO WRITER DETECTION
    # -----------------------------
    print("\n=== Checking available video writers ===")
    available_writers = animation.writers.list()
    print("Available writers:", available_writers)

    if "ffmpeg" in available_writers:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=10, metadata=dict(artist="Python"), bitrate=1800)
    elif "avconv" in available_writers:
        Writer = animation.writers["avconv"]
        writer = Writer(fps=10, metadata=dict(artist="Python"), bitrate=1800)
    else:
        Writer = animation.writers["pillow"]
        writer = Writer(fps=10)

    print("=======================================================\n")

    # -----------------------------
    # ANIMATION UPDATE FUNCTION
    # -----------------------------
    def update_frame(i):
        ax.cla()
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_zlim(0, Lz)
        ax.set_facecolor("white")
        ax.view_init(30, 45)

        v3D = v_frames[i].reshape((Nxx, Nyy, Nzz))

        for lvl in thickness_levels:
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    v3D, level=lvl, spacing=spacing
                )
            except Exception:
                continue

            # Normalize values
            norm_vals = (values - values.min()) / (values.max() - values.min())
            facecolors = plt.cm.turbo(norm_vals)
            facecolors[:, -1] = 0.85

            mesh = Poly3DCollection(
                verts[faces],
                facecolor=facecolors,
                edgecolor="gray",
                linewidth=0.1,
                alpha=0.85
            )
            ax.add_collection3d(mesh)

        #ax.set_title(f"Time = {frame_numbers[i] * dt:.2f} s", color="black")

    # -----------------------------
    # RUN ANIMATION
    # -----------------------------
    ani = animation.FuncAnimation(fig, update_frame, frames=len(v_frames))

    if video_flag:
        ani.save("scroll_simulation.mp4", writer=writer)

    plt.close(fig)
