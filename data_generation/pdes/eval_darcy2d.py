"""Evaluation data generation for Darcy2d on domains A/B/C."""

import random

import numpy as np
import ufl
from dolfinx import fem
from ufl import dx, grad, inner

from .base import setup_eval_mesh


def generate_eval_solution(domain_type, num_samples=100):
    """Generate num_samples Darcy flow solutions on evaluation domain.

    Returns:
        List of (sol, [qf, bc]) tuples.
    """
    domain, V, _, boundary_elems, boundary_index, boundary_points = setup_eval_mesh(domain_type)

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_elems)
    uD = fem.Function(V)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    q = fem.Function(V)
    f = fem.Function(V)

    a = q * inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    problem = fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    samples = []
    for _ in range(num_samples):
        uD.x.array[list(boundary_index)] = [random.random() for _ in boundary_index]
        f.x.array[:] = [random.random() for _ in range(f.x.array.shape[0])]
        q.x.array[:] = [random.random() for _ in range(q.x.array.shape[0])]

        uh = problem.solve()

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]],
            uh.x.array[..., np.newaxis],
        ], axis=1)
        qf = np.concatenate([
            domain.geometry.x[:, [0, 1]],
            q.x.array[..., np.newaxis],
            f.x.array[..., np.newaxis],
        ], axis=1)
        boundary_condition = np.concatenate([
            boundary_points[:, [0, 1]],
            uh.x.array[boundary_index, np.newaxis],
            np.zeros((len(boundary_index), 1)),
        ], axis=1)

        samples.append((sol, [qf, boundary_condition]))

    return samples
