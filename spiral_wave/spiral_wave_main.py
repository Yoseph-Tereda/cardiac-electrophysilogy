import matplotlib.pyplot as plt
import matplotlib.animation as animation
from spiral_simulation import  SpiralSimulation  # Import the simulation class
from spiral_data import M_matrix, K_matrix  # Import matrices for spatial discretization


if __name__ == '__main__':

    # =========================================================
    #                  SIMULATION PARAMETERS
    # =========================================================
    sigma_i = 0.1   # Intracellular conductivity
    cm = 1          # Membrane capacitance
    chi = 1000      # Surface-to-volume ratio
    Nxx = 51        # Number of grid points in x
    Nyy = 51        # Number of grid points in y
    t_span = [0, 1000]  # Time interval
    b = 1           # Forward Euler coefficient (from general formula)

    # Initialize spiral simulation object
    scw = SpiralSimulation(M_matrix, K_matrix, sigma_i, cm, chi, Nxx, Nyy, t_span, b)

    dt = scw.step_size_FE()          # Compute stable time step using eigenvalues
    ZZ = scw.constant_matrix()       # Compute constant matrix for diffusion term
    s_current = scw.s_intial()       # Initialize state variable (recovery variable)
    v_current = scw.v_intial()       # Initialize transmembrane potential


    # =========================================================
    #                  ANIMATION PARAMETERS
    # =========================================================
    frame_skip = 10   # Save every 10th step for visualization
    video_flag = True # Flag to save the animation as MP4

    # =========================================================
    #        STORAGE FOR VISUALIZATION FRAMES
    # =========================================================
    v_frames = []     # Store v snapshots for animation
    frame_numbers = []


    # =========================================================
    #                  TIME INTEGRATION LOOP
    # =========================================================
    for n in range(0, scw.number_mesh_time() - 1):

        # Simple reaction kinetics (FitzHugh–Nagumo type)
        dv = v_current * (1 - v_current) * (v_current - 0.1) - s_current
        ds = 0.01 * (0.5 * v_current - s_current)

        # Forward Euler update for reaction terms
        v_next = v_current + dt * dv
        s_next = s_current + dt * ds

        # Add diffusion effect using precomputed constant matrix
        v_next = v_next + dt * (ZZ @ v_next)

        # Save frame for visualization
        if n % frame_skip == 0:
            print(f"Saving frame at step {n}/{scw.number_mesh_time()-1}")
            v_frames.append(v_next.copy())
            frame_numbers.append(n)

        # Update current state
        v_current = v_next
        s_current = s_next


    # =========================================================
    #                  FIGURE SETUP FOR ANIMATION
    # =========================================================
    fig, ax = plt.subplots()
    im = ax.imshow(v_frames[0].reshape(Nxx, Nxx), cmap="jet", origin="lower", vmin=0, vmax=1)
    ax.set_title("Time = 0.00 s")
    plt.colorbar(im)


    # =========================================================
    #        ANIMATION UPDATE FUNCTION
    # =========================================================
    def update_frame(i):
        im.set_data(v_frames[i].reshape(Nxx, Nxx))
        ax.set_title(f"Time = {frame_numbers[i] * dt:.2f} s")
        return [im]


    # =========================================================
    #                     RUN ANIMATION
    # =========================================================
    ani = animation.FuncAnimation(fig, update_frame, frames=len(v_frames), blit=True)

    if video_flag:
        ani.save("spiral_simulation_2D.mp4", writer="ffmpeg", fps=10)
        print("MP4 saved → spiral_simulation_2D.mp4")

    plt.show()
