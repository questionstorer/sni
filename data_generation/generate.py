#!/usr/bin/env python
"""Unified training/test data generation for all PDEs.

Supports: laplace2d, laplace2d_mixed, darcy2d, heat2d, nonlinear_poisson2d

Due to dolfinx memory constraints, each process should generate a limited
number of samples. Multiple processes write temporary shards which are then
merged into a single output file per equation.

Usage (from the sni/ directory):

    # Generate one shard (process 1 of N)
    python data_generation/generate.py --pde laplace2d --mode train --count 1

    # Merge all shards for an equation into one pkl
    python data_generation/generate.py --pde laplace2d --mode train --merge

    # The shell script handles both steps automatically:
    bash scripts/generate_data.sh laplace2d train 8
"""

import argparse
import glob
import logging
import os
import pickle
import random
import shutil
import sys

# Add sni/ root to sys.path (must be before other imports)
_sni_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _sni_root)
# Add data_generation/ so that the 'pdes' package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gmsh
from tqdm import tqdm

from pdes import PDE_NAMES, get_pde
from utils.polygon import generate_polygon_mesh

logging.getLogger().setLevel(logging.ERROR)


def _shard_dir(output_dir, prefix, mode):
    """Return the temporary shard directory path."""
    return os.path.join(output_dir, f".shards_{prefix}_{mode}")


def generate(args):
    pde = get_pde(args.pde)
    config = pde.DEFAULT_CONFIG.copy()

    # Override config with CLI args
    for key in ['num_polygons', 'num_batch', 'min_vertices', 'max_vertices', 'mesh_lc']:
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    gmsh.initialize()

    datalist = []
    for _ in tqdm(range(config['num_polygons']), desc=f"Generating {args.pde} (shard {args.count})"):
        num_points = random.randrange(config['min_vertices'], config['max_vertices'])
        generate_polygon_mesh(num_points, 'simple', lc=config['mesh_lc'])
        samples = pde.generate_solution(config['num_batch'])
        datalist.extend(samples)

    # Save to a temporary shard file
    shard_path = _shard_dir(args.output_dir, config['output_prefix'], args.mode)
    os.makedirs(shard_path, exist_ok=True)
    shard_file = os.path.join(shard_path, f"shard_{args.count:03d}.pkl")
    with open(shard_file, 'wb') as f:
        pickle.dump(datalist, f)

    print(f"Saved {len(datalist)} samples to shard {shard_file}")


def merge(args):
    pde = get_pde(args.pde)
    config = pde.DEFAULT_CONFIG.copy()

    shard_path = _shard_dir(args.output_dir, config['output_prefix'], args.mode)
    shard_files = sorted(glob.glob(os.path.join(shard_path, "shard_*.pkl")))

    if not shard_files:
        print(f"No shards found in {shard_path}")
        sys.exit(1)

    datalist = []
    for sf in shard_files:
        with open(sf, 'rb') as f:
            datalist.extend(pickle.load(f))

    if args.output:
        output_path = args.output
    else:
        filename = f"{config['output_prefix']}_{len(datalist)}_{args.mode}.pkl"
        output_path = os.path.join(args.output_dir, filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(datalist, f)

    # Clean up shards
    shutil.rmtree(shard_path)
    print(f"Merged {len(shard_files)} shards ({len(datalist)} samples) into {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Training/test data generation for PDEs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--pde', type=str, required=True, choices=PDE_NAMES,
                        help='PDE to generate data for')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Data split (default: train)')
    parser.add_argument('--count', type=int, default=1,
                        help='Process/shard index (default: 1)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge all shards into a single output file')
    parser.add_argument('--num_polygons', type=int, default=None,
                        help='Number of random polygons (default: PDE-specific)')
    parser.add_argument('--num_batch', type=int, default=None,
                        help='Solutions per polygon (default: PDE-specific)')
    parser.add_argument('--min_vertices', type=int, default=None,
                        help='Min polygon vertices (default: PDE-specific)')
    parser.add_argument('--max_vertices', type=int, default=None,
                        help='Max polygon vertices (default: PDE-specific)')
    parser.add_argument('--mesh_lc', type=float, default=None,
                        help='Mesh characteristic length (default: PDE-specific)')
    parser.add_argument('--output_dir', type=str, default='data/2d',
                        help='Output directory (default: data/2d)')
    parser.add_argument('--output', type=str, default=None,
                        help='Full output path (overrides auto-naming)')
    args = parser.parse_args()

    if args.merge:
        merge(args)
    else:
        generate(args)


if __name__ == '__main__':
    main()
