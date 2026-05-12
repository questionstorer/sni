"""Laplace equation in 2D with mixed Dirichlet/Neumann boundary condition.

PDE: -nabla^2 u = 0      on Omega
BC:  u = u_D              on Gamma_D  (Dirichlet)
     du/dn = g            on Gamma_N  (Neumann)

With probability (1-p), the boundary is split into contiguous Dirichlet and
Neumann segments. With probability p, pure Dirichlet BC is used.

Data format: (sol, [bc])
    sol: (N, 3) array — columns [x, y, u]
    bc:  (M, 4) array — columns [x, y, value, type]
         type=0 for Dirichlet, type=1 for Neumann
"""

import random

import numpy as np
import ufl
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner

from .base import SPACE_SHIFT, setup_mesh

DEFAULT_CONFIG = {
    'num_polygons': 10,
    'num_batch': 20,
    'min_vertices': 3,
    'max_vertices': 12,
    'mesh_lc': 0.1,
    'output_prefix': 'laplace2d_mixed_simple',
}


def generate_solution(batch_pp, p=0.2):
    """Generate batch_pp solutions with mixed BC on the current gmsh mesh.

    Args:
        batch_pp: Number of solutions to generate.
        p: Probability of pure Dirichlet BC (default: 0.2, so 80% mixed).

    Returns:
        List of (sol, [bc]) tuples.
    """
    domain, V, _, boundary_elems_with_vertices, boundary_index, boundary_points = setup_mesh()

    samples = []
    for _ in range(batch_pp):
        mixed = random.random() >= p

        if mixed:
            # Select contiguous Dirichlet segment; rest is Neumann
            start = random.randint(0, len(boundary_elems_with_vertices) - 1)
            end = (random.randint(len(boundary_elems_with_vertices) // 2,
                                  len(boundary_elems_with_vertices)) + start - 1) \
                  % len(boundary_elems_with_vertices)

            if end > start:
                db_elems = [e for e, v1, v2 in boundary_elems_with_vertices[start:end]]
                db_index = [v1 for e, v1, v2 in boundary_elems_with_vertices[start:end]]
                db_index.append(boundary_elems_with_vertices[end - 1][2])

                nb_elems = [e for e, v1, v2 in
                            boundary_elems_with_vertices[end:] + boundary_elems_with_vertices[:start]]
                nb_index = [v1 for e, v1, v2 in
                            boundary_elems_with_vertices[end:] + boundary_elems_with_vertices[:start]]
                nb_index.append(boundary_elems_with_vertices[start - 1][2])
            elif end < start:
                db_elems = [e for e, v1, v2 in
                            boundary_elems_with_vertices[start:] + boundary_elems_with_vertices[:end]]
                db_index = [v1 for e, v1, v2 in
                            boundary_elems_with_vertices[start:] + boundary_elems_with_vertices[:end]]
                db_index.append(boundary_elems_with_vertices[end - 1][2])

                nb_elems = [e for e, v1, v2 in boundary_elems_with_vertices[end:start]]
                nb_index = [v1 for e, v1, v2 in boundary_elems_with_vertices[end:start]]
                nb_index.append(boundary_elems_with_vertices[start - 1][2])
        else:
            db_elems = [e for e, v1, v2 in boundary_elems_with_vertices]
            db_index = [v1 for e, v1, v2 in boundary_elems_with_vertices]
            nb_elems = []
            nb_index = []

        db_points = domain.geometry.x[db_index]
        nb_points = domain.geometry.x[nb_index]

        boundary_dofs = fem.locate_dofs_topological(V, 1, db_elems)
        uD = fem.Function(V)
        bc = fem.dirichletbc(uD, boundary_dofs)
        g = fem.Function(V)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, ScalarType(0))

        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx - g * v * ds

        problem = fem.petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

        if len(nb_index) > 0:
            random_range = random.uniform(0.1, 1.0)
            if random.random() >= 0.5:
                uD.x.array[db_index] = [random.random() * random_range for _ in db_index]
                g.x.array[nb_index] = [random.random() for _ in nb_index]
            else:
                uD.x.array[db_index] = [random.random() for _ in db_index]
                g.x.array[nb_index] = [random.random() * random_range for _ in nb_index]
        else:
            uD.x.array[db_index] = [random.random() for _ in db_index]

        uh = problem.solve()

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]] - SPACE_SHIFT,
            uh.x.array[..., np.newaxis],
        ], axis=1)
        dbc = np.concatenate([
            db_points[:, [0, 1]] - SPACE_SHIFT,
            uD.x.array[db_index, np.newaxis],
            np.zeros((len(db_index), 1)),
        ], axis=1)
        nbc = np.concatenate([
            nb_points[:, [0, 1]] - SPACE_SHIFT,
            g.x.array[nb_index, np.newaxis],
            np.ones((len(nb_index), 1)),
        ], axis=1)
        bc_data = np.concatenate([dbc, nbc], axis=0)

        samples.append((sol, [bc_data]))

    return samples
