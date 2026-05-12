"""Evaluation data generation for Heat2d on domains A/B/C."""

import random

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from .base import setup_eval_mesh


def generate_eval_solution(domain_type, num_samples=100):
    """Generate num_samples heat equation solutions on evaluation domain.

    Returns:
        List of (sol, alpha, [bc]) tuples.
    """
    T = 0.5
    num_steps = 50
    dt = T / num_steps

    domain, V, _, boundary_elems, boundary_index, boundary_points = setup_eval_mesh(domain_type)

    u_n = fem.Function(V)

    boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_elems)
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

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)
    b = create_vector(linear_form)

    samples = []
    for _ in range(num_samples):
        series, bc_series = [], []
        alpha.x.array[:] = [1.0] * alpha.x.array.shape[0]

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

        # Fixed boundary condition across all time steps
        uD.x.array[list(boundary_index)] = [random.uniform(0.5, 1.0) for _ in boundary_index]

        for i in range(num_steps):
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
            domain.geometry.x[:, [0, 1]],
            np.concatenate([s[..., np.newaxis] for s in series], axis=1),
        ], axis=1)
        bc_data = np.concatenate([
            boundary_points[:, [0, 1]],
            np.concatenate([s[..., np.newaxis] for s in bc_series[:-1]], axis=1),
            np.zeros((boundary_points.shape[0], 1)),
        ], axis=1)

        samples.append((sol, alpha.x.array[0], [bc_data]))

    return samples
