"""Nonlinear Poisson (Laplace) equation in 2D with Dirichlet BC.

PDE: -nabla . (q(u) nabla u) = 0   on Omega,  where q(u) = 1 + u^2
BC:  u = u_D                        on dOmega

Solved with Newton's method.

Data format: (sol, [bc])
    sol: (N, 3) array — columns [x, y, u]
    bc:  (M, 4) array — columns [x, y, u_D, 0]
"""

import random

import numpy as np
import ufl
from dolfinx import fem, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc

from .base import SPACE_SHIFT, setup_mesh

DEFAULT_CONFIG = {
    'num_polygons': 250,
    'num_batch': 10,
    'min_vertices': 3,
    'max_vertices': 12,
    'mesh_lc': 0.1,
    'output_prefix': 'nonlinear_poisson2d_simple',
}


def generate_solution(batch_pp):
    """Generate batch_pp nonlinear Poisson solutions on the current gmsh mesh.

    Returns:
        List of (sol, [bc]) tuples.
    """
    domain, V, boundary_facets, _, boundary_index, boundary_points = setup_mesh()

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
    uD = fem.Function(V)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = fem.Function(V)
    v = ufl.TestFunction(V)

    def q(u):
        return 1 + u**2

    F = q(u) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

    problem = NonlinearProblem(F, u, bcs=[bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "hypre"
    opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()

    samples = []
    for _ in range(batch_pp):
        uD.x.array[list(boundary_index)] = [random.random() for _ in boundary_index]

        log.set_log_level(log.LogLevel.INFO)
        n, converged = solver.solve(u)
        assert converged, f"Newton solver did not converge after {n} iterations"

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]] - SPACE_SHIFT,
            u.x.array[..., np.newaxis],
        ], axis=1)
        boundary_condition = np.concatenate([
            boundary_points[:, [0, 1]] - SPACE_SHIFT,
            uD.x.array[boundary_index, np.newaxis],
            np.zeros((len(boundary_index), 1)),
        ], axis=1)

        samples.append((sol, [boundary_condition]))

    return samples
