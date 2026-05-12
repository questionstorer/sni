#!/usr/bin/env python
"""Evaluation data generation for all PDEs on pre-defined domains (A, B, C).

Generates test data on fixed evaluation meshes stored in data/mesh/.

Usage (from the sni/ directory):

    # Single PDE, single domain
    python data_generation/generate_eval.py --pde laplace2d --domain A

    # Single PDE, all domains
    python data_generation/generate_eval.py --pde laplace2d --domain all

    # All PDEs, all domains
    python data_generation/generate_eval.py --pde all --domain all

    # Custom number of samples
    python data_generation/generate_eval.py --pde darcy2d --domain B --num_samples 200

    # The shell script handles everything:
    bash scripts/generate_eval_data.sh
"""

import argparse
import logging
import os
import pickle
import sys

# Add sni/ root to sys.path (must be before other imports)
_sni_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _sni_root)
# Add data_generation/ so that the 'pdes' package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gmsh
from tqdm import tqdm

from pdes import PDE_NAMES, get_eval_pde
from pdes.base import EVAL_DOMAIN_NAMES

logging.getLogger().setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluation data generation on pre-defined domains',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--pde', type=str, required=True,
                        choices=PDE_NAMES + ['all'],
                        help='PDE to generate data for (or "all")')
    parser.add_argument('--domain', type=str, required=True,
                        choices=EVAL_DOMAIN_NAMES + ['all'],
                        help='Evaluation domain (A, B, C, or "all")')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per domain (default: 100)')
    parser.add_argument('--output_dir', type=str, default='data/2d',
                        help='Output directory (default: data/2d)')
    args = parser.parse_args()

    pdes = PDE_NAMES if args.pde == 'all' else [args.pde]
    domains = EVAL_DOMAIN_NAMES if args.domain == 'all' else [args.domain]

    gmsh.initialize()

    for pde_name in pdes:
        eval_pde = get_eval_pde(pde_name)

        for domain_type in domains:
            print(f"Generating {pde_name} on domain {domain_type} "
                  f"({args.num_samples} samples)...")
            datalist = eval_pde.generate_eval_solution(domain_type, args.num_samples)

            filename = f"{pde_name}_{domain_type}_{len(datalist)}_test.pkl"
            output_path = os.path.join(args.output_dir, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(datalist, f)

            print(f"  Saved {len(datalist)} samples to {output_path}")

    print("Done.")


if __name__ == '__main__':
    main()
