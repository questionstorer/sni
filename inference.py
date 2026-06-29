import os
import numpy as np
import torch
import sys
from tqdm import tqdm
import re
import time
sys.path.append("./")
os.environ["METIS_DLL"]="./lib/libmetis.so"
from models.GNOT.data_utils import get_model, get_loss_func

from utils.logging_utils import resetLogger
from models.ddno import DDNO 
from args import get_inference_args

from utils.domain import DecomposedSimplePolygonMeshDomain, DecomposedSpaceTimeSimplePolygonMeshDomain
from trimesh.base import Trimesh
from utils.data_utils import (get_inference_boundary_marker, 
                              get_inference_dolphinx_dataset, 
                              get_inference_mesh, 
                              get_inference_normalizer)
import logging

logger = logging.getLogger(__name__)

PDE_NAMES = ['laplace2d', 'laplace2d_mixed', 'darcy2d', 'heat2d', 'nonlinear_poisson2d']
ITERATION_METHOD = 'schwarz'

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


def restore_physical_tensor(tensor, normalizer):
    if normalizer is None:
        return tensor
    return normalizer.transform(tensor, inverse=True)


def align_graph_solution_to_domain(model, graph, x_normalizer, y_normalizer):
    physical_x = restore_physical_tensor(graph.ndata['x'], x_normalizer)
    physical_y = restore_physical_tensor(graph.ndata['y'], y_normalizer)

    aligned_sol = torch.zeros(
        (model.domain.num_nodes, physical_y.shape[1]),
        dtype=physical_y.dtype,
        device=physical_y.device,
    )
    counts = torch.zeros(
        (model.domain.num_nodes, 1),
        dtype=physical_y.dtype,
        device=physical_y.device,
    )

    indices = []
    for point in physical_x:
        if model.space_dim == 2:
            query = np.concatenate([point[:2].detach().cpu().numpy(), np.zeros((1,))])
        else:
            query = point[:3].detach().cpu().numpy()
        _, index = model.domain.tree.query(query)
        indices.append(index)

    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=physical_y.device)
    aligned_sol.index_add_(0, index_tensor, physical_y)
    counts.index_add_(
        0,
        index_tensor,
        torch.ones((index_tensor.shape[0], 1), dtype=physical_y.dtype, device=physical_y.device),
    )

    return aligned_sol / counts.clamp_min(1)


def schwarz_iterate(pde_name, model, sol, bic, u_p, input_func, tau):
    """Evaluate one relaxed Schwarz fixed-point map."""
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
        local_sols = model(sol, bic, u_p, input_func)
        patched_sol = model.patch_local_sols(local_sols).to(sol.device)
        return (1 - tau) * sol + tau * patched_sol


def validate_iteration_args(args):
    if args.tau is None:
        raise ValueError("--tau is required for inference.")
    if args.stop_mode != 'metric_stagnation':
        raise ValueError("--stop-mode must be metric_stagnation.")


def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def tensor_norm(tensor):
    return torch.linalg.vector_norm(tensor.reshape(-1)).item()


def relative_norm(delta, reference, eps=1e-12):
    delta_norm = tensor_norm(delta)
    reference_norm = max(tensor_norm(reference), eps)
    return delta_norm, delta_norm / reference_norm


def metric_stagnated(metric_history, window=10, decimals=4):
    if len(metric_history) < window:
        return False
    return round(metric_history[-1], decimals) == round(metric_history[-window], decimals)


def get_stop_reason(metric_history):
    if metric_stagnated(metric_history):
        return 'metric_stagnation'
    return None


def get_metric_index(pde_name):
    """Return the index into the metric_func output tuple used for the loss."""
    return 0 if pde_name == 'heat2d' else 2


if __name__ == "__main__":

    resetLogger()
    args = get_inference_args()
    validate_iteration_args(args)

    pde_name = get_pde_name(args.dataset)

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
        device = torch.device("cpu")

    args.test_num = int(args.test_num) if args.test_num not in ['all', 'none'] else args.test_num

    test_dataset = get_inference_dolphinx_dataset(args)
    args.dataset_config = test_dataset.config

    args.space_dim = int(re.search(r'\d', args.dataset).group())
    metric_func = get_loss_func(name='rel2', args=args, regularizer=False, normalizer=None)

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
    sample_summaries = []
    for sample_idx, data in enumerate(tqdm(test_dataset), start=1):
        graph, u_p, inputs_f = data

        input_func = prepare_input_func(pde_name, model, inputs_f, device)
        gt_sol = align_graph_solution_to_domain(
            model,
            graph,
            test_dataset.x_normalizer,
            test_dataset.y_normalizer,
        )

        inputs_f = inputs_f.to(device)
        u_p = u_p.to(device)
        graph = graph.to(device)
        gt_sol = gt_sol.to(device)

        epochs = args.epochs
        tau = args.tau

        metric_history = []
        sample_start = time.perf_counter()
        total_map_wall_time = 0.0
        total_iter_wall_time = 0.0
        final_metric_value = float('nan')
        final_fp_residual_abs = float('nan')
        final_fp_residual_rel = float('nan')
        final_update_rel = float('nan')
        stop_reason = 'max_epochs'

        with torch.no_grad():
            bic = prepare_boundary_conditions(pde_name, model, inputs_f)
            sol = model.initialize(inputs_f)

            for i in range(epochs):
                prev_sol = sol
                synchronize_device(device)
                iter_start = time.perf_counter()
                mapped_sol = schwarz_iterate(pde_name, model, prev_sol, bic, u_p, input_func, tau)
                synchronize_device(device)
                map_end = time.perf_counter()

                map_wall_time = map_end - iter_start
                if not torch.isfinite(mapped_sol).all().item():
                    total_map_wall_time += map_wall_time
                    total_iter_wall_time += map_wall_time
                    final_metric_value = float('inf')
                    final_fp_residual_abs = float('inf')
                    final_fp_residual_rel = float('inf')
                    final_update_rel = float('inf')
                    stop_reason = 'nonfinite_mapped_sol'
                    break

                fixed_point_delta = mapped_sol - prev_sol
                fp_residual_abs, fp_residual_rel = relative_norm(fixed_point_delta, prev_sol)
                sol = mapped_sol
                if not torch.isfinite(sol).all().item():
                    total_map_wall_time += map_wall_time
                    total_iter_wall_time += map_wall_time
                    final_metric_value = float('inf')
                    final_fp_residual_abs = float('inf')
                    final_fp_residual_rel = float('inf')
                    final_update_rel = float('inf')
                    stop_reason = 'nonfinite_sol'
                    break

                metric_value = float(metric_func(graph, sol, gt_sol)[metric_idx])
                metric_history.append(metric_value)
                synchronize_device(device)
                iter_end = time.perf_counter()

                update_delta = sol - prev_sol
                _, update_rel = relative_norm(update_delta, prev_sol)
                iter_wall_time = iter_end - iter_start

                if not all(np.isfinite(value) for value in (
                    metric_value,
                    fp_residual_abs,
                    fp_residual_rel,
                    update_rel,
                )):
                    total_map_wall_time += map_wall_time
                    total_iter_wall_time += iter_wall_time
                    final_metric_value = metric_value
                    final_fp_residual_abs = fp_residual_abs
                    final_fp_residual_rel = fp_residual_rel
                    final_update_rel = update_rel
                    stop_reason = 'nonfinite_metric'
                    break

                total_map_wall_time += map_wall_time
                total_iter_wall_time += iter_wall_time
                final_metric_value = metric_value
                final_fp_residual_abs = fp_residual_abs
                final_fp_residual_rel = fp_residual_rel
                final_update_rel = update_rel

                stop_reason_candidate = get_stop_reason(metric_history)
                if stop_reason_candidate is not None:
                    stop_reason = stop_reason_candidate
                    break

        steps_taken = len(metric_history)
        sample_total_wall_time = time.perf_counter() - sample_start
        avg_iter_wall_time = total_iter_wall_time / steps_taken if steps_taken > 0 else 0.0
        losses.append(final_metric_value)
        sample_summary = {
            'sample_idx': sample_idx,
            'method': ITERATION_METHOD,
            'steps': steps_taken,
            'stop_reason': stop_reason,
            'final_error': final_metric_value,
            'final_fp_residual_abs': final_fp_residual_abs,
            'final_fp_residual_rel': final_fp_residual_rel,
            'final_update_rel': final_update_rel,
            'total_map_wall_time_s': total_map_wall_time,
            'total_wall_time_s': sample_total_wall_time,
            'avg_iter_wall_time_s': avg_iter_wall_time,
        }
        sample_summaries.append(sample_summary)
        logger.info(
            "sample=%d method=%s steps=%d stop_reason=%s final_error=%.6e final_fp_residual_abs=%.6e final_fp_residual_rel=%.6e final_update_rel=%.6e total_map_wall_time_s=%.6f total_wall_time_s=%.6f avg_iter_wall_time_s=%.6f",
            sample_idx,
            ITERATION_METHOD,
            steps_taken,
            stop_reason,
            final_metric_value,
            final_fp_residual_abs,
            final_fp_residual_rel,
            final_update_rel,
            total_map_wall_time,
            sample_total_wall_time,
            avg_iter_wall_time,
        )

    if sample_summaries:
        logger.info(
            "dataset=%s method=%s samples=%d avg_steps=%.2f avg_final_error=%.6e avg_final_fp_residual_rel=%.6e avg_total_wall_time_s=%.6f avg_iter_wall_time_s=%.6f",
            args.dataset,
            ITERATION_METHOD,
            len(sample_summaries),
            np.mean([summary['steps'] for summary in sample_summaries]),
            np.mean([summary['final_error'] for summary in sample_summaries]),
            np.mean([summary['final_fp_residual_rel'] for summary in sample_summaries]),
            np.mean([summary['total_wall_time_s'] for summary in sample_summaries]),
            np.mean([summary['avg_iter_wall_time_s'] for summary in sample_summaries]),
        )
    logger.info(losses)
