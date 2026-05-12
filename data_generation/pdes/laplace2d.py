"""Laplace equation in 2D with pure Dirichlet boundary condition.

PDE: -nabla^2 u = 0      on Omega
BC:  u = u_D              on dOmega

Data format: (sol, [bc])
    sol: (N, 3) array — columns [x, y, u]
    bc:  (M, 4) array — columns [x, y, u_D, 0]  (last column is BC type: 0=Dirichlet)
"""

import random

import numpy as np
import ufl
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from ufl import dx, grad, inner

from .base import SPACE_SHIFT, setup_mesh

DEFAULT_CONFIG = {
    'num_polygons': 250,
    'num_batch': 10,
    'min_vertices': 3,
    'max_vertices': 12,
    'mesh_lc': 0.1,
    'output_prefix': 'laplace2d_simple',
}


def generate_solution(batch_pp):
    """Generate batch_pp solutions with random Dirichlet BC on the current gmsh mesh.

    Returns:
        List of (sol, [bc]) tuples.
    """
    domain, V, boundary_facets, _, boundary_index, boundary_points = setup_mesh()

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_facets)
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
    for _ in range(batch_pp):
        uD.x.array[list(boundary_index)] = [random.random() for _ in boundary_index]
        uh = problem.solve()

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]] - SPACE_SHIFT,
            uh.x.array[..., np.newaxis],
        ], axis=1)
        boundary_condition = np.concatenate([
            boundary_points[:, [0, 1]] - SPACE_SHIFT,
            uD.x.array[boundary_index, np.newaxis],
            np.zeros((len(boundary_index), 1)),
        ], axis=1)

        samples.append((sol, [boundary_condition]))

    return samples
