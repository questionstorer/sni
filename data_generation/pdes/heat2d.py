"""Time-dependent heat equation in 2D with Dirichlet BC.

PDE: du/dt - alpha * nabla^2 u = 0   on Omega x (0, T]
BC:  u = u_D(t)                       on dOmega
IC:  u(0) = u_0                       on Omega

Time discretization: implicit Euler with num_steps=10, T=0.1.

Data format: (sol, alpha, [bc])
    sol:   (N, 2+num_steps) array — columns [x, y, u_0, u_1, ..., u_{num_steps-1}]
    alpha: scalar thermal diffusivity
    bc:    (M, 2+num_steps+1) array — columns [x, y, bc_0, ..., bc_{num_steps-1}, 0]
"""

import random

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from .base import SPACE_SHIFT, setup_mesh

DEFAULT_CONFIG = {
    'num_polygons': 250,
    'num_batch': 50,
    'min_vertices': 3,
    'max_vertices': 12,
    'mesh_lc': 0.1,
    'output_prefix': 'heat2d_simple',
}


def generate_solution(batch_pp):
    """Generate batch_pp heat equation solutions on the current gmsh mesh.

    Returns:
        List of (sol, alpha, [bc]) tuples.
    """
    T = 0.1
    num_steps = 10
    dt = T / num_steps

    domain, V, boundary_facets, _, boundary_index, boundary_points = setup_mesh()

    u_n = fem.Function(V)

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_facets.tolist())
    uD = fem.Function(V)
    bc = fem.dirichletbc(uD, boundary_dofs)

    uh = fem.Function(V)
    uh.name = "uh"

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType(0))
    alpha = fem.Function(V)

    a = u * v * ufl.dx + alpha * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx

    samples = []
    for _ in range(batch_pp):
        series, bc_series = [], []
        alpha.x.array[:] = [random.uniform(0.1, 1.0)] * alpha.x.array.shape[0]

        bilinear_form = fem.form(a)
        linear_form = fem.form(L)
        b = create_vector(linear_form)
        A = assemble_matrix(bilinear_form, bcs=[bc])
        A.assemble()

        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        # Random initial condition
        u_n.x.array[:] = [random.random() for _ in range(u_n.x.array.shape[0])]
        uh.x.array[:] = u_n.x.array
        bc_series.append(np.copy(uh.x.array[list(boundary_index)]))

        for i in range(num_steps):
            # Random boundary condition at each time step
            uD.x.array[list(boundary_index)] = [random.random() for _ in boundary_index]

            series.append(np.copy(uh.x.array[:]))
            bc_series.append(np.copy(uD.x.array[list(boundary_index)]))

            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, linear_form)
            apply_lifting(b, [bilinear_form], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            solver.solve(b, uh.vector)
            uh.x.scatter_forward()
            u_n.x.array[:] = uh.x.array

        sol = np.concatenate([
            domain.geometry.x[:, [0, 1]] - SPACE_SHIFT,
            np.concatenate([s[..., np.newaxis] for s in series], axis=1),
        ], axis=1)
        bc_data = np.concatenate([
            boundary_points[:, [0, 1]] - SPACE_SHIFT,
            np.concatenate([s[..., np.newaxis] for s in bc_series[:-1]], axis=1),
            np.zeros((boundary_points.shape[0], 1)),
        ], axis=1)

        samples.append((sol, alpha.x.array[0], [bc_data]))

    return samples
