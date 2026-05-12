"""Evaluation data generation for Laplace2d with mixed Dirichlet/Neumann BC on domains A/B/C."""

import random

import numpy as np
import ufl
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner

from .base import setup_eval_mesh_mixed


def generate_eval_solution(domain_type, num_samples=100):
    """Generate num_samples solutions on evaluation domain with mixed BC.

    Returns:
        List of (sol, [bc]) tuples.
    """
    domain, V, _, db_elems, db_index, db_points, nb_elems, nb_index, nb_points = \
        setup_eval_mesh_mixed(domain_type)

    boundary_dofs = fem.locate_dofs_topological(V, 1, db_elems)
    uD = fem.Function(V)
    g = fem.Function(V)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, ScalarType(0))

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx - g * v * ds

    problem = fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    samples = []
    for _ in range(num_samples):
        uD.x.array[db_index] = [random.random() for _ in db_index]
        g.x.array[nb_index] = [random.random() for _ in nb_index]
        uh = problem.solve()

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]],
            uh.x.array[..., np.newaxis],
        ], axis=1)
        dbc = np.concatenate([
            db_points[:, [0, 1]],
            uD.x.array[db_index, np.newaxis],
            np.zeros((len(db_index), 1)),
        ], axis=1)
        nbc = np.concatenate([
            nb_points[:, [0, 1]],
            g.x.array[nb_index, np.newaxis],
            np.ones((len(nb_index), 1)),
        ], axis=1)
        bc_data = np.concatenate([dbc, nbc], axis=0)

        samples.append((sol, [bc_data]))

    return samples
