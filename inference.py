import os
import numpy as np
import torch
import sys
from tqdm import tqdm
import re
sys.path.append("./")
os.environ["METIS_DLL"]="./lib/libmetis.so"
from models.GNOT.data_utils import get_model, get_loss_func

from utils.logging_utils import resetLogger
from models.ddno import DDNO 
from args import get_inference_args

import plotly.io as pio
pio.renderers.default = 'iframe'

from utils.domain import DecomposedSimplePolygonMeshDomain, DecomposedSpaceTimeSimplePolygonMeshDomain
from trimesh.base import Trimesh
from utils.data_utils import (get_inference_boundary_marker, 
                              transform_gt, 
                              get_inference_dolphinx_dataset, 
                              get_inference_mesh, 
                              get_inference_normalizer)
import logging

logger = logging.getLogger(__name__)

PDE_NAMES = ['laplace2d', 'laplace2d_mixed', 'darcy2d', 'heat2d', 'nonlinear_poisson2d']

def get_pde_name(dataset):
    """Extract the PDE name from the dataset string (e.g. 'darcy2d_schwarz' -> 'darcy2d')."""
    for name in sorted(PDE_NAMES, key=len, reverse=True):
        if dataset.startswith(name):
            return name
    raise ValueError(f"Unknown PDE type for dataset: {dataset}")


def build_domain(pde_name, mesh, boundary_marker, args):
    if pde_name == 'heat2d':
        return DecomposedSpaceTimeSimplePolygonMeshDomain(
            mesh, dim=2, boundary_marker=boundary_marker,
            n_parts=args.n_parts, depth=args.depth,
            time_step=args.time_step, time_span=args.time_span)
    else:
        return DecomposedSimplePolygonMeshDomain(
            mesh, dim=2, boundary_marker=boundary_marker,
            n_parts=args.n_parts, depth=args.depth)


def build_model(pde_name, local_operator, domain, normalizer, args):
    if pde_name == 'heat2d':
        return DDNO(local_operator, domain, 2,
                    time_dependent=True, time_span=args.time_span,
                    normalizer=normalizer)
    else:
        return DDNO(local_operator, domain, 2, normalizer=normalizer)


def prepare_input_func(pde_name, model, inputs_f, device):
    """Build the input function list. Only darcy2d has non-empty input functions."""
    if pde_name == 'darcy2d':
        input_func = []
        for x in inputs_f.x:
            f = torch.zeros((model.domain.num_nodes, x.shape[1] - model.space_dim),
                            dtype=torch.float32).to(device)
            indices, func_value = model.map_input(x)
            f[indices] = func_value
            input_func.append(f)
        # throw away the last function which is boundary condition
        return input_func[:-1]
    return []


def prepare_boundary_conditions(pde_name, model, inputs_f):
    """Extract boundary (and initial) conditions from inputs_f."""
    if pde_name == 'heat2d':
        bc = model.map_boundary(inputs_f[1])
        ic = model.map_input(inputs_f[0])
        return (bc, ic)
    elif pde_name == 'darcy2d':
        bc = model.map_boundary(inputs_f[1])
        return (bc, None)
    else:
        bc = model.map_boundary(inputs_f[0])
        return (bc, None)


def schwarz_iterate(pde_name, model, sol, bic, u_p, input_func, tau):
    """Perform one Schwarz iteration and return the updated solution."""
    if pde_name == 'heat2d':
        p = model.domain.n_parts
        q = model.domain.num_interval
        temporal_local_sols = model(sol, bic, u_p, input_func)
        extended_temporal_sols = [
            sum([((model.rm[i].T @ temporal_local_sols[t][i] @ model.trm[t].T)
                  + (1 - model.masks[i] @ model.time_masks[t].T) * sol).to(sol.device)
                 for i, _ in enumerate(model.domain.subDomain)])
            for t, _ in enumerate(model.domain.subTimeInterval)
        ]
        return (1 - tau * (p * q)) * sol + tau * sum(extended_temporal_sols)
    else:
        p = model.domain.n_parts
        local_sols = model(sol, bic, u_p, input_func)
        extended_sols = [
            (model.rm[i].T @ v + (1 - model.masks[i]) * sol).to(sol.device)
            for i, v in enumerate(local_sols)
        ]
        return (1 - tau * p) * sol + tau * sum(extended_sols)


def get_metric_index(pde_name):
    """Return the index into the metric_func output tuple used for the loss."""
    return 0 if pde_name == 'heat2d' else 2


if __name__ == "__main__":

    resetLogger()
    args = get_inference_args()

    pde_name = get_pde_name(args.dataset)

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
        device = torch.device("cpu")

    args.test_num = int(args.test_num) if args.test_num not in ['all', 'none'] else args.test_num

    test_dataset = get_inference_dolphinx_dataset(args)
    args.dataset_config = test_dataset.config

    args.space_dim = int(re.search(r'\d', args.dataset).group())
    args.normalizer = test_dataset.y_normalizer.to(device) if test_dataset.y_normalizer is not None else None

    loss_func = get_loss_func(name=args.loss_name, args=args, regularizer=True, normalizer=args.normalizer)
    metric_func = get_loss_func(name='rel2', args=args, regularizer=False, normalizer=args.normalizer)

    gmesh, trimesh = get_inference_mesh(args)
    normalizer = get_inference_normalizer(args)(device)
    boundary_marker = get_inference_boundary_marker(args, gmesh)

    mesh = Trimesh(trimesh['vertices'], trimesh["faces"])
    domain = build_domain(pde_name, mesh, boundary_marker, args)

    local_operator = get_model(args)
    local_operator.load_state_dict(torch.load(args.model_path)["model"])
    model = build_model(pde_name, local_operator, domain, normalizer, args)
    model.to(device)

    metric_idx = get_metric_index(pde_name)
    losses = []

    for data in tqdm(test_dataset):
        graph, u_p, inputs_f = data

        input_func = prepare_input_func(pde_name, model, inputs_f, device)
        gt_sol = transform_gt(model, graph)

        inputs_f = inputs_f.to(device)
        u_p = u_p.to(device)
        graph = graph.to(device)
        gt_sol = gt_sol.to(device)

        epochs = args.epochs
        tau = args.tau

        loss = []
        with torch.no_grad():
            bic = prepare_boundary_conditions(pde_name, model, inputs_f)
            sol = model.initialize(inputs_f)

            for i in range(epochs):
                sol = schwarz_iterate(pde_name, model, sol, bic, u_p, input_func, tau)
                loss.append(round(float(metric_func(graph, sol, gt_sol)[metric_idx]), 4))

                if len(loss) >= 10 and loss[-1] == loss[-10]:
                    break

            losses.append(loss[-1])
            logger.info(loss[-1])
    logger.info(losses)
