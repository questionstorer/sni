"""Evaluation data generation for Laplace2d with pure Dirichlet BC on domains A/B/C."""

import random

import numpy as np
import ufl
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from ufl import dx, grad, inner

from .base import setup_eval_mesh


def generate_eval_solution(domain_type, num_samples=100):
    """Generate num_samples solutions on evaluation domain.

    Returns:
        List of (sol, [bc]) tuples.
    """
    domain, V, _, boundary_elems, boundary_index, boundary_points = setup_eval_mesh(domain_type)

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_elems)
    uD = fem.Function(V)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, ScalarType(0))

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    problem = fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    samples = []
    for _ in range(num_samples):
        uD.x.array[list(boundary_index)] = [random.random() for _ in boundary_index]
        uh = problem.solve()

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]],
            uh.x.array[..., np.newaxis],
        ], axis=1)
        boundary_condition = np.concatenate([
            boundary_points[:, [0, 1]],
            uD.x.array[boundary_index, np.newaxis],
            np.zeros((len(boundary_index), 1)),
        ], axis=1)

        samples.append((sol, [boundary_condition]))

    return samples
