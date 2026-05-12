"""PDE solver registry for data generation.

Each training PDE module provides:
    - DEFAULT_CONFIG: dict with num_polygons, num_batch, min_vertices,
      max_vertices, mesh_lc, output_prefix
    - generate_solution(batch_pp): function that generates solutions on the
      current gmsh mesh, returns a list of data tuples

Each eval PDE module provides:
    - generate_eval_solution(domain_type, num_samples): function that generates
      solutions on a pre-defined evaluation domain (A/B/C)
"""

import importlib

PDE_NAMES = [
    'laplace2d',
    'laplace2d_mixed',
    'darcy2d',
    'heat2d',
    'nonlinear_poisson2d',
]

EVAL_PDE_NAMES = [
    'eval_laplace2d',
    'eval_laplace2d_mixed',
    'eval_darcy2d',
    'eval_heat2d',
    'eval_nonlinear_poisson2d',
]

# Maps CLI name -> eval module name
EVAL_PDE_MAP = {name: f'eval_{name}' for name in PDE_NAMES}


def get_pde(name):
    """Import and return the PDE module by name."""
    if name not in PDE_NAMES:
        raise ValueError(f"Unknown PDE: {name}. Available: {PDE_NAMES}")
    return importlib.import_module(f'.{name}', package=__name__)


def get_eval_pde(name):
    """Import and return the eval PDE module by PDE name."""
    if name not in PDE_NAMES:
        raise ValueError(f"Unknown PDE: {name}. Available: {PDE_NAMES}")
    module_name = EVAL_PDE_MAP[name]
    return importlib.import_module(f'.{module_name}', package=__name__)
