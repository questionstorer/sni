
# Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving

**[ICLR 2026]**  
Code and data for our paper: [Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving](https://arxiv.org/abs/2504.00510)

---

## Overview

Neural operators often struggle to generalize to new geometries that differ significantly from those seen during training. This limitation restricts their real-world applicability, especially in industrial settings where unseen geometries are common.

**Our solution:** We propose a local-to-global framework that combines operator learning with domain decomposition methods (DDMs) to enable geometry generalization in PDE solving.

<p align="center">
  <img src="./assets/framework.png" alt="Framework Overview" />
</p>

### Framework Components

1. **Training Data Generation**  
   - Generate random basic shapes
   - Impose appropriate boundary conditions
   - Create diverse training data for the neural operator

2. **Local Operator Learning**  
   - Train neural operators on basic shapes
   - Use data augmentation based on PDE symmetries
   - Capture intricate details and variations

3. **Schwarz Neural Inference (SNI)**  
   - Partition the computational domain into subdomains
   - Apply the learned operator locally
   - Iteratively stitch and update the global solution using additive Schwarz methods

---

## Getting Started

*Coming soon: Instructions for setup and usage.*

---

## Data & Data Generation

### Environment Setup

Install the FEM environment for data generation:

```bash
conda env create -f environment_fem.yml
conda activate fem
```

### Supported PDEs

| PDE Name | CLI Key | Description |
|---|---|---|
| Laplace2d-Dirichlet | `laplace2d` | Laplace equation with pure Dirichlet BC |
| Laplace2d-Mixed | `laplace2d_mixed` | Laplace equation with mixed Dirichlet/Neumann BC |
| Darcy2d | `darcy2d` | Darcy flow with coefficient field and source term |
| Heat2d | `heat2d` | Time-dependent heat equation with Dirichlet BC |
| NonlinearLaplace2d | `nonlinear_poisson2d` | Nonlinear Poisson equation (q(u)=1+u²) |

### Training Data Generation

All data generation is done through a single unified script. **Run all commands from the `sni/` directory.**

Due to dolfinx memory constraints, each process generates a limited number of samples (shards). After all processes finish, shards are merged into a single pkl per equation.

**Recommended: use the shell script** which handles sharding and merging automatically:

```bash
# All PDEs, training mode, 8 processes each
bash scripts/generate_data.sh all train 8

# Single PDE
bash scripts/generate_data.sh darcy2d train 8
```

**Manual step-by-step:**

```bash
# 1. Generate shards (one process per shard)
for n in $(seq 1 8); do
    python data_generation/generate.py --pde laplace2d --mode train --count $n
done

# 2. Merge shards into a single pkl file
python data_generation/generate.py --pde laplace2d --mode train --merge
```

#### CLI Options

| Argument | Description | Default |
|---|---|---|
| `--pde` | PDE type (required) | — |
| `--mode` | `train` or `test` | `train` |
| `--count` | Shard/process index | `1` |
| `--merge` | Merge all shards into one output file | — |
| `--num_polygons` | Random polygons per process | PDE-specific |
| `--num_batch` | Solutions per polygon | PDE-specific |
| `--min_vertices` | Min polygon vertices | PDE-specific |
| `--max_vertices` | Max polygon vertices | PDE-specific |
| `--mesh_lc` | Mesh characteristic length | PDE-specific |
| `--output_dir` | Output directory | `data/2d` |
| `--output` | Full output path (overrides auto-naming) | — |

### Test Data Generation

Test data on random polygons uses the same script with `--mode test`:

```bash
bash scripts/generate_data.sh laplace2d test 1
```

### Evaluation Data Generation

Evaluation data is generated on three pre-defined domains (A, B, C) stored as meshes in `data/mesh/`. These are the domains used in the paper's experiments.

| Domain | Mesh File | Description |
|---|---|---|
| A | `A-schwarz.msh` | Union of a disk and a rectangle |
| B | `B-holes.msh` | Square with two holes removed |
| C | `C-bosch.msh` | Disk with a complex shape removed |

**Recommended: use the shell script:**

```bash
# All PDEs, all domains, 100 samples each (default)
bash scripts/generate_eval_data.sh

# Single PDE, all domains
bash scripts/generate_eval_data.sh darcy2d

# Single PDE, single domain
bash scripts/generate_eval_data.sh laplace2d A

# Custom number of samples
bash scripts/generate_eval_data.sh all all 200
```

**Manual:**

```bash
# Single PDE, single domain
python data_generation/generate_eval.py --pde laplace2d --domain A

# All PDEs, all domains
python data_generation/generate_eval.py --pde all --domain all

# Custom samples
python data_generation/generate_eval.py --pde darcy2d --domain B --num_samples 200
```

### Data Format

Output files are pickled lists of tuples. The format depends on the PDE:

- **laplace2d, laplace2d_mixed, nonlinear_poisson2d**: `(sol, [bc])` — sol is `(N,3)` with `[x, y, u]`, bc is `(M,4)` with `[x, y, value, type]`
- **darcy2d**: `(sol, [qf, bc])` — qf is `(N,4)` with `[x, y, a, f]` (coefficient and source fields)
- **heat2d**: `(sol, alpha, [bc])` — sol has time steps as extra columns, alpha is the thermal diffusivity scalar

### Download Training Data and Test Data

*Coming soon.*

---

## Training

*Coming soon: Training procedures and configuration.*

---

## Inference on Unseen Geometries

*Coming soon: How to run inference on new geometries.*

---

## Results

<p align="center">
  <img src="./assets/main_result.png" alt="Main Results" />
</p>

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{huang2025operator,
  title={Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving},
  author={Huang, Jianing and Zhang, Kaixuan and Wu, Youjia and Cheng, Ze},
  journal={arXiv preprint arXiv:2504.00510},
  year={2025}
}
```