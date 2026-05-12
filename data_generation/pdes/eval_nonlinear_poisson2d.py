"""Evaluation data generation for NonlinearPoisson2d on domains A/B/C."""

import random

import numpy as np
import ufl
from dolfinx import fem, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from .base import setup_eval_mesh


def generate_eval_solution(domain_type, num_samples=100):
    """Generate num_samples nonlinear Poisson solutions on evaluation domain.

    Returns:
        List of (sol, [bc]) tuples.
    """
    domain, V, _, boundary_elems, boundary_index, boundary_points = setup_eval_mesh(domain_type)

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_elems)
    uD = fem.Function(V)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, ScalarType(0))

    def q(u):
        return 1 + u**2

    F = q(u) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

    problem = NonlinearProblem(F, u, bcs=[bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True
    solver.max_it = 1000

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
    for _ in range(num_samples):
        uD.x.array[list(boundary_index)] = [random.random() for _ in boundary_index]

        log.set_log_level(log.LogLevel.INFO)
        n, converged = solver.solve(u)
        assert converged, f"Newton solver did not converge after {n} iterations"

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]],
            u.x.array[..., np.newaxis],
        ], axis=1)
        boundary_condition = np.concatenate([
            boundary_points[:, [0, 1]],
            u.x.array[boundary_index, np.newaxis],
            np.zeros((len(boundary_index), 1)),
        ], axis=1)

        samples.append((sol, [boundary_condition]))

    return samples
