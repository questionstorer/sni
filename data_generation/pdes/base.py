"""Common setup for PDE data generation.

Provides shared mesh setup and boundary extraction used by all PDE solvers.
"""

import os
import sys

# Add sni/ root to sys.path so that utils.polygon is importable
_sni_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _sni_root not in sys.path:
    sys.path.insert(0, _sni_root)

import gmsh
import numpy as np
from dolfinx import fem
from dolfinx.io import gmshio
from mpi4py import MPI

from utils.polygon import get_counterclockwise_dolphinx_boundary

SPACE_SHIFT = np.array([0.5, 0.5])

# Evaluation domain configs: mesh file and boundary facet markers
EVAL_DOMAINS = {
    'A': {
        'mesh_file': 'data/mesh/A-schwarz.msh',
        'boundary_markers': [21, 22],
        'dirichlet_markers': [21],
        'neumann_markers': [22],
    },
    'B': {
        'mesh_file': 'data/mesh/B-holes.msh',
        'boundary_markers': [11, 12, 13],
        'dirichlet_markers': [12, 13],
        'neumann_markers': [11],
    },
    'C': {
        'mesh_file': 'data/mesh/C-bosch.msh',
        'boundary_markers': [12, 13],
        'dirichlet_markers': [12],
        'neumann_markers': [13],
    },
}

EVAL_DOMAIN_NAMES = list(EVAL_DOMAINS.keys())


def setup_mesh():
    """Convert the current gmsh model 'p1' to a dolfinx mesh and extract boundary.

    Must be called after generate_polygon_mesh() which creates the 'p1' model.

    Returns:
        domain: dolfinx Mesh
        V: FunctionSpace (Lagrange, degree 1)
        boundary_facets: numpy array of boundary facet indices
        boundary_elems_with_vertices: list of (elem, v1, v2) tuples in CCW order
        boundary_index: list of boundary vertex indices in CCW order
        boundary_points: array of boundary vertex coordinates
    """
    gmsh.model.setCurrent('p1')
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    V = fem.FunctionSpace(domain, ("Lagrange", 1))

    boundary_facets = facet_markers.find(1)
    boundary_elems_with_vertices, boundary_index, boundary_points = \
        get_counterclockwise_dolphinx_boundary(domain, boundary_facets.tolist())

    return domain, V, boundary_facets, boundary_elems_with_vertices, boundary_index, boundary_points


def setup_eval_mesh(domain_type):
    """Load a pre-defined evaluation domain mesh and extract boundary info.

    Args:
        domain_type: one of 'A', 'B', 'C'

    Returns:
        domain: dolfinx Mesh
        V: FunctionSpace (Lagrange, degree 1)
        facet_markers: MeshTags for facets
        boundary_elems: list of all boundary facet indices
        boundary_index: sorted list of boundary vertex indices
        boundary_points: array of boundary vertex coordinates
    """
    cfg = EVAL_DOMAINS[domain_type]
    mesh_path = os.path.join(_sni_root, cfg['mesh_file'])
    gmsh.open(mesh_path)

    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    V = fem.FunctionSpace(domain, ("Lagrange", 1))

    boundary_elems = []
    for marker in cfg['boundary_markers']:
        boundary_elems.extend(facet_markers.find(marker).tolist())

    boundary_index = set()
    for i in boundary_elems:
        boundary_index = boundary_index.union(
            domain.topology.connectivity(1, 0).links(i).tolist()
        )
    boundary_index = sorted(list(boundary_index))
    boundary_points = domain.geometry.x[boundary_index]

    return domain, V, facet_markers, boundary_elems, boundary_index, boundary_points


def setup_eval_mesh_mixed(domain_type):
    """Load evaluation domain mesh with separate Dirichlet/Neumann boundaries.

    Args:
        domain_type: one of 'A', 'B', 'C'

    Returns:
        domain, V, facet_markers,
        db_elems, db_index, db_points,
        nb_elems, nb_index, nb_points
    """
    cfg = EVAL_DOMAINS[domain_type]
    mesh_path = os.path.join(_sni_root, cfg['mesh_file'])
    gmsh.open(mesh_path)

    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    V = fem.FunctionSpace(domain, ("Lagrange", 1))

    db_elems = []
    for marker in cfg['dirichlet_markers']:
        db_elems.extend(facet_markers.find(marker).tolist())
    nb_elems = []
    for marker in cfg['neumann_markers']:
        nb_elems.extend(facet_markers.find(marker).tolist())

    db_index = set()
    for i in db_elems:
        db_index = db_index.union(domain.topology.connectivity(1, 0).links(i).tolist())
    db_index = sorted(list(db_index))
    db_points = domain.geometry.x[db_index]

    nb_index = set()
    for i in nb_elems:
        nb_index = nb_index.union(domain.topology.connectivity(1, 0).links(i).tolist())
    nb_index = sorted(list(nb_index))
    nb_points = domain.geometry.x[nb_index]

    return domain, V, facet_markers, db_elems, db_index, db_points, nb_elems, nb_index, nb_points
