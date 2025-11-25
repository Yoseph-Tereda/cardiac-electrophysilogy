"""
spiral_simulation.py

This module defines a class `spiralsimulation` to simulate spiral wave dynamics
in a 2D excitable medium, based on a discretized reaction-diffusion system.
It provides functions to set up initial conditions, compute the constant matrix
for time integration, and determine suitable step sizes for a forward Euler solver.
"""

import numpy as np

class SpiralSimulation:
    """
    Class to simulate spiral wave propagation in a 2D excitable medium.

    Attributes:
        M_matrix (ndarray): Mass matrix from spatial discretization.
        K_matrix (ndarray): Stiffness matrix from spatial discretization.
        sigma_i (float): Conductivity parameter of the medium.
        cm (float): Membrane capacitance.
        chi (float): Surface-to-volume ratio.
        Nxx, Nyy (int): Number of grid points in x and y directions.
        t_span (tuple): Time interval for simulation, e.g., (t0, tf).
        b (float): Scaling or other parameter (purpose context-specific).
    """

    def __init__(self, M_matrix, K_matrix, sigma_i, cm, chi, Nxx, Nyy, t_span, b):
        """
        Initialize the spiral simulation with system parameters.

        Args:
            M_matrix (ndarray): Mass matrix.
            K_matrix (ndarray): Stiffness matrix.
            sigma_i (float): Conductivity of the tissue.
            cm (float): Membrane capacitance.
            chi (float): Surface-to-volume ratio.
            Nxx (int): Number of mesh points in x-direction.
            Nyy (int): Number of mesh points in y-direction.
            t_span (tuple): Time interval for simulation.
            b : float
                Time integration weight coefficient.
                For Forward Euler method, b = 1. This corresponds to using only the current derivative
                in the general linear multistep or Runge-Kutta formula:
                u_{n+1} = u_n + b * dt * f(u_n)
                where f(u_n) is the derivative of the state at time step n.
            """

        self.sigma_i = sigma_i
        self.cm = cm
        self.chi = chi
        self.Nxx = Nxx
        self.Nyy = Nyy
        self.t_span = t_span
        self.b = b
        self.M_matrix = M_matrix
        self.K_matrix = K_matrix

    def constant_matrix(self):
        """
        Compute the constant matrix used in time integration of the system.

        The constant matrix is defined as:
        A = inv(M ⊗ M) * (-σ_i / (cm * χ)) * (K ⊗ M + M ⊗ K)

        Returns:
            ndarray: The constant matrix for the time-stepping scheme.
        """
        # Compute the Kronecker products of the stiffness and mass matrices
        kron_term = np.kron(self.K_matrix, self.M_matrix) + np.kron(self.M_matrix, self.K_matrix)

        # Scale by conductivity and membrane properties
        scaled_term = -(self.sigma_i / (self.cm * self.chi)) * kron_term

        # Multiply by inverse of Kronecker product of mass matrices
        constant_matrix = np.linalg.inv(np.kron(self.M_matrix, self.M_matrix)).dot(scaled_term)
        return constant_matrix

    def v_intial(self):
        """
        Generate the initial condition for transmembrane potential.

        The initial potential is zero everywhere except in the top-left 25x25 block,
        which is set to 1 to initiate the spiral wave.

        Returns:
            ndarray: Flattened 1D array of initial transmembrane potentials.
        """
        # Initialize potential matrix
        v_initial = np.zeros((self.Nxx, self.Nyy))

        # Set the initial stimulus region
        v_initial[0:25, 0:25] = 1

        # Flatten 2D matrix to 1D for solver
        v_initial = v_initial.flatten()
        return v_initial

    def s_intial(self):
        """
        Generate the initial condition for the vector of state variables of the cell model.

        The state variable is zero everywhere except in specific regions set to 0.1.

        Returns:
            ndarray: Flattened 1D array of initial state variables.
        """
        # Initialize state variable matrix
        s_initial = np.zeros((self.Nxx, self.Nyy))

        # Set initial values in specific blocks
        s_initial[0:25, 26:51] = 0.1
        s_initial[26:51, 26:51] = 0.1

        # Flatten 2D matrix to 1D for solver
        s_initial = s_initial.flatten()
        return s_initial

    def step_size_FE(self):
        """
        Calculate a stable time step size for forward Euler integration.

        The step size is computed from the eigenvalues of the constant matrix as:
        dt_FE = -2 / min(eigenvalue(A))

        Returns:
            float: Recommended forward Euler step size.
        """
        # Compute eigenvalues of the constant matrix
        egv = np.linalg.eigvalsh(self.constant_matrix())

        # Use the smallest eigenvalue for stability criterion
        min_eigen_value = min(egv)

        # Compute the step size based on stability
        step_size_FE = -2 / min_eigen_value
        return step_size_FE

    def number_mesh_time(self):
        """
        Calculate the total number of time points (mesh points) for the simulation interval.

        Uses the recommended step size from forward Euler.

        Returns:
            int: Number of time points for the simulation.
        """
        # Compute total time points by dividing interval by step size
        total_mesh_point = round((self.t_span[1] - self.t_span[0]) / self.step_size_FE())
        return total_mesh_point
