"""

This class provides:
- Initialization of transmembrane potential and state variable fields.
- Construction of a matrix-free linear operator for the diffusion term.
- Estimation of a stable Forward Euler time step using spectral radius.
- Calculation of the number of time steps for the simulation.

"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu


class ScrollSimulation():
    """
    A class that constructs and analyzes a 3-dimensional cardiac scroll-wave
    simulation using finite-element mass and stiffness matrices.

    Parameters
    ----------
    M_matrix : ndarray or sparse matrix
        Finite-element mass matrix.
    K_matrix : ndarray or sparse matrix
        Finite-element stiffness matrix.
    sigma_i : float
        Intracellular conductivity.
    cm : float
        Membrane capacitance.
    chi : float
        Surface-to-volume ratio.
    Nx, Ny, Nz : int
        Number of spatial points in each (x,y,z) dimension.
    t_span : (float, float)
        Time interval for simulation.
    b : float
        Time integration weight coefficient.
        For Forward Euler method, b = 1. This corresponds to using only the current derivative
        in the general linear multistep or Runge-Kutta formula:
        u_{n+1} = u_n + b * dt * f(u_n)
        where f(u_n) is the derivative of the state at time step n.

    """

    def __init__(self, M_matrix, K_matrix, sigma_i, cm, chi,
                 Nx, Ny, Nz, t_span, b):
        self.sigma_i = sigma_i
        self.cm = cm
        self.chi = chi

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.t_span = t_span
        self.b = b

        self.M_matrix = M_matrix
        self.K_matrix = K_matrix

    # ----------------------------------------------------------------------
    # Initial Conditions
    # ----------------------------------------------------------------------
    def v_intial(self):
        """
        Create the initial condition for transmembrane potential.

        Returns
        -------
        v_initial : ndarray
            Flattened 3-D array of size (Nx*Ny*Nz).
        """
        # 3-D tensor for voltage initialization
        XX = np.zeros((self.Nx, self.Ny, self.Nz))

        # Set a block region to 1 to trigger activation
        XX[0:25, 0:25, :] = 1

        return XX.flatten()

    def s_intial(self):
        """
        Create the initial condition for the vector of cell-model state variables.

        Returns
        -------
        s_initial : ndarray
            Flattened 3-D array of size (Nx*Ny*Nz).
        """
        XXX = np.zeros((self.Nx, self.Ny, self.Nz))

        # Two regions with nonzero initial conditions
        XXX[0:25, 26:51, :] = 0.1
        XXX[26:51, 26:51, :] = 0.1

        return XXX.flatten()

    # ----------------------------------------------------------------------
    # Forward Euler Stability Estimate
    # ----------------------------------------------------------------------
    def step_size_FE(self):
        """
        Compute a stable Forward Euler time step.

        This uses the smallest eigenvalue of the diffusion operator:
            du/dt = A u
        where A is built from combinations of tensor products of M and K.

        Returns
        -------
        float
            The forward Euler time-step size: -2 / lambda_min.

        We compute the smallest eigenvalue of the 3-D operator

            A = c * (M^{-1} ⊗ M^{-1} ⊗ M^{-1}) *
                    (K ⊗ M ⊗ M + M ⊗ K ⊗ M + M ⊗ M ⊗ K)

        without ever forming any full Kronecker products.

        Why this technique is used:

        1. Avoids building huge 3-D matrices
           --------------------------------------------------
           The full operator A would be of size:
               (Nx*Ny*Nz) × (Nx*Ny*Nz)
           For Nx = Ny = Nz = 51 → dimension = 132,651.
           The Kronecker matrices would require several gigabytes
           of memory and are completely impractical to store.
           Instead, we apply each (K, M) factor directly using
           tensor contractions (tensordot), keeping memory usage low.

        2. Uses the 1-D LU factorization of M for fast solves
           --------------------------------------------------
           M^{-1} ⊗ M^{-1} ⊗ M^{-1} acts by solving three 1-D mass
           matrix systems along x, y, and z.
           Because M is the same 1-D matrix in every direction,
           we factor it once with LU (splu) and reuse the factorization.
           This reduces the cost from O(N^3) to O(3 * N^2).

        3. Operator is applied in matrix-free form
           --------------------------------------------------
           eigsh() from SciPy only needs matrix-vector products.
           By providing the operator as a LinearOperator with a custom
           matvec(), we avoid forming A explicitly and greatly reduce
           memory and computational overhead.

        4. Efficient computation of the smallest eigenvalue
           --------------------------------------------------
           eigsh() with which='SA' uses the Lanczos algorithm, which is
           well-suited for large symmetric operators. It requires only
           repeated matvec() operations, making the matrix-free approach ideal.

        Summary:
        --------
        This technique allows us to solve a very large 3-D generalized
        eigenvalue problem efficiently by:
        • avoiding full Kronecker matrices,
        • exploiting tensor structure,
        • reusing a 1-D LU factorization,
        • and using a matrix-free iterative eigensolver.

        The result is a method that is orders of magnitude faster and
        more memory-efficient than forming the full operator explicitly.
        """

        # Convert mass/stiffness matrices to sparse form
        M = csr_matrix(self.M_matrix)
        K = csr_matrix(self.K_matrix)

        # Scaling constant in the diffusion term
        c = -(self.sigma_i / (self.cm * self.chi))

        # LU factorization M⁻¹ for repeated use
        M_lu = splu(M)

        # Helpers for reshaping between vector and tensor
        def vec_to_tensor(v):
            return v.reshape((self.Nx, self.Ny, self.Nz))

        def tensor_to_vec(T):
            return T.reshape(-1)

        # ------------------------------------------------------
        # Matrix-free application of A*v
        # ------------------------------------------------------
        def apply_operator(v):
            """
            Apply the operator A to vector v, using tensor contractions
            instead of explicitly forming Kronecker products.
            """
            U = vec_to_tensor(v)

            # Term 1: (K ⊗ M ⊗ M) U
            T1 = np.tensordot(K.toarray(), U, axes=(1, 0))
            T1 = np.tensordot(M.toarray(), T1, axes=(1, 1))
            T1 = np.tensordot(M.toarray(), T1, axes=(1, 2))

            # Term 2: (M ⊗ K ⊗ M) U
            T2 = np.tensordot(M.toarray(), U, axes=(1, 0))
            T2 = np.tensordot(K.toarray(), T2, axes=(1, 1))
            T2 = np.tensordot(M.toarray(), T2, axes=(1, 2))

            # Term 3: (M ⊗ M ⊗ K) U
            T3 = np.tensordot(M.toarray(), U, axes=(1, 0))
            T3 = np.tensordot(M.toarray(), T3, axes=(1, 1))
            T3 = np.tensordot(K.toarray(), T3, axes=(1, 2))

            RHS = T1 + T2 + T3

            # Apply (M⁻¹ ⊗ M⁻¹ ⊗ M⁻¹)
            # Sequentially solve in each dimension
            for _ in range(3):
                RHS = (
                    M_lu.solve(RHS.reshape(self.Nx, -1))
                    .reshape(self.Nx, self.Ny, self.Nz)
                    .transpose((1, 2, 0))
                )

            return c * tensor_to_vec(RHS)

        # Wrap into a scipy LinearOperator (matrix-free)
        N = self.Nx * self.Ny * self.Nz
        A = LinearOperator((N, N), matvec=apply_operator, dtype=float)

        # Compute the smallest algebraic eigenvalue of A
        vals, _ = eigsh(A, k=1, which='SA')
        lambda_min = vals[0]

        # Forward Euler stability condition
        return -2 / lambda_min

    # ----------------------------------------------------------------------
    # Time Mesh
    # ----------------------------------------------------------------------
    def number_mesh_time(self):
        """
        Compute the total number of time steps required for the
        simulation interval using the stable Forward Euler time step.

        Returns
        -------
        int
            Number of time points.
        """
        dtt = self.step_size_FE()
        return round((self.t_span[1] - self.t_span[0]) / dtt)
