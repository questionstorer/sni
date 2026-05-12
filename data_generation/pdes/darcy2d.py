"""Darcy flow in 2D with coefficient field, source term, and Dirichlet BC.

PDE: -nabla . (a(x) nabla u) = f(x)   on Omega
BC:  u = u_D                           on dOmega

Data format: (sol, [qf, bc])
    sol: (N, 3) array — columns [x, y, u]
    qf:  (N, 4) array — columns [x, y, a, f]  (coefficient and source at mesh nodes)
    bc:  (M, 4) array — columns [x, y, u_D, 0]
"""

import random

import numpy as np
import ufl
from dolfinx import fem
from ufl import dx, grad, inner

from .base import SPACE_SHIFT, setup_mesh

DEFAULT_CONFIG = {
    'num_polygons': 250,
    'num_batch': 10,
    'min_vertices': 3,
    'max_vertices': 16,
    'mesh_lc': 0.1,
    'output_prefix': 'darcy2d_simple',
}


def generate_solution(batch_pp):
    """Generate batch_pp Darcy flow solutions on the current gmsh mesh.

    Returns:
        List of (sol, [qf, bc]) tuples.
    """
    domain, V, boundary_facets, _, boundary_index, boundary_points = setup_mesh()

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
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
    for _ in range(batch_pp):
        random_range = random.uniform(0.3, 1.0)
        uD.x.array[list(boundary_index)] = [random.random() * random_range for _ in boundary_index]
        f.x.array[:] = [-5 * random.random() for _ in range(f.x.array.shape[0])]
        q.x.array[:] = [random.random() for _ in range(q.x.array.shape[0])]

        uh = problem.solve()

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]] - SPACE_SHIFT,
            uh.x.array[..., np.newaxis],
        ], axis=1)
        qf = np.concatenate([
            domain.geometry.x[:, [0, 1]] - SPACE_SHIFT,
            q.x.array[..., np.newaxis],
            f.x.array[..., np.newaxis],
        ], axis=1)
        boundary_condition = np.concatenate([
            boundary_points[:, [0, 1]] - SPACE_SHIFT,
            uD.x.array[boundary_index, np.newaxis],
            np.zeros((len(boundary_index), 1)),
        ], axis=1)

        samples.append((sol, [qf, boundary_condition]))

    return samples
