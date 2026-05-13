
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

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Conda or Mamba package manager

### Environment Setup

This project uses two conda environments:

| Environment | Config File | Purpose |
|---|---|---|
| `fem` | `environment_fem.yml` | FEM-based data generation (dolfinx, gmsh) |
| `pde` | `environment_pde.yml` | Model training and inference (PyTorch, DGL) |

```bash
# Data generation environment
conda env create -f environment_fem.yml

# Training & inference environment
conda env create -f environment_pde.yml
```

### Project Structure

```
sni/
├── train.py                 # Training script (all PDEs)
├── inference.py             # Inference script (all PDEs)
├── args.py                  # CLI argument definitions
├── models/
│   ├── ddno.py              # Domain Decomposed Neural Operator
│   ├── GNOT/                # GNOT model & data utilities
│   └── GeoFNO/              # GeoFNO model
├── utils/
│   ├── data_utils.py        # Dataset loading & normalization
│   ├── domain.py            # Domain decomposition classes
│   ├── augmentation.py      # PDE symmetry-based augmentations
│   ├── normalization.py     # PDE-specific normalizers
│   └── logging_utils.py     # Logging setup
├── data_generation/
│   ├── generate.py          # Training/test data generation
│   ├── generate_eval.py     # Evaluation domain data generation
│   └── pdes/                # PDE-specific solvers
├── scripts/
│   ├── generate_data.sh     # Parallel data generation
│   └── generate_eval_data.sh
├── data/
│   ├── 2d/                  # Generated datasets (.pkl)
│   ├── mesh/                # Pre-defined evaluation meshes (.msh)
│   └── chkpt/               # Model checkpoints
└── lib/
    └── libmetis.so          # METIS library for mesh partitioning
```

### Quick Start

The full pipeline has three stages. Run all commands from the `sni/` directory.

```bash
# 1. Generate training data (activate fem environment)
conda activate fem
bash scripts/generate_data.sh laplace2d train 8

# 2. Train the local operator (activate pde environment)
conda activate pde
python train.py --dataset laplace2d_simple --epochs 500

# 3. Run inference on unseen geometries
bash scripts/generate_eval_data.sh laplace2d    # generate eval data (fem env)
conda activate pde
python inference.py --dataset laplace2d_schwarz --model-path data/chkpt/<checkpoint>.pt --tau 0.1
```

### Supported PDEs

| PDE Name | CLI Key | Description |
|---|---|---|
| Laplace2d-Dirichlet | `laplace2d` | Laplace equation with pure Dirichlet BC |
| Laplace2d-Mixed | `laplace2d_mixed` | Laplace equation with mixed Dirichlet/Neumann BC |
| Darcy2d | `darcy2d` | Darcy flow with coefficient field and source term |
| Heat2d | `heat2d` | Time-dependent heat equation with Dirichlet BC |
| NonlinearLaplace2d | `nonlinear_poisson2d` | Nonlinear Poisson equation (q(u)=1+u²) |

### Download Data

Pre-generated training and evaluation data are available on Hugging Face:

```bash
# Download and extract to sni/data/2d/
huggingface-cli download questionstorer/sni --repo-type dataset --local-dir data/2d/
```

Alternatively, you can generate the data yourself (see [Data & Data Generation](#data--data-generation)).

---

## Data & Data Generation

**Environment:** activate `fem` before running data generation commands.

```bash
conda activate fem
```

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

---

## Training

**Environment:** activate `pde` before running training commands.

```bash
conda activate pde
```

### Training the Local Operator

All PDEs use a single training script. **Run all commands from the `sni/` directory.**

```bash
# Laplace equation (Dirichlet BC)
python train.py --dataset laplace2d_simple --epochs 500

# Laplace equation (mixed Dirichlet/Neumann BC)
python train.py --dataset laplace2d_mixed_simple --epochs 500

# Darcy flow
python train.py --dataset darcy2d_simple --epochs 500

# Heat equation (time-dependent)
python train.py --dataset heat2d_simple --epochs 500 --time-step 5

# Nonlinear Poisson
python train.py --dataset nonlinear_poisson2d_simple --epochs 500
```

### Resuming Training

To resume from a checkpoint:

```bash
python train.py --dataset laplace2d_simple --resume data/chkpt/<checkpoint>.pt
```

### Training CLI Options

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset/PDE type (see table below) | — |
| `--epochs` | Number of training epochs | `500` |
| `--batch-size` | Training batch size | `4` |
| `--val-batch-size` | Validation batch size | `8` |
| `--lr` | Max learning rate | `0.001` |
| `--lr-method` | LR schedule: `cycle`, `step`, `warmup` | `cycle` |
| `--optimizer` | Optimizer: `Adam`, `AdamW` | `AdamW` |
| `--model-name` | Model architecture: `GNOT`, `CGPT` | `GNOT` |
| `--n-hidden` | Hidden dimension size | `64` |
| `--n-layers` | Number of layers | `3` |
| `--loss-name` | Loss function: `rel2`, `rel1`, `l2`, `l1` | `rel2` |
| `--time-step` | Time basis dimension (heat2d only) | `1` |
| `--resume` | Path to checkpoint to resume from | — |
| `--gpu` | GPU device id | `0` |
| `--no-cuda` | Disable CUDA | `False` |
| `--use-tb` | Enable TensorBoard logging | `0` |

### Training Datasets

| PDE | Dataset Key |
|---|---|
| Laplace2d (Dirichlet) | `laplace2d_simple` |
| Laplace2d (Mixed) | `laplace2d_mixed_simple` |
| Darcy2d | `darcy2d_simple` |
| Heat2d | `heat2d_simple` |
| Nonlinear Poisson2d | `nonlinear_poisson2d_simple` |

Checkpoints are saved to `data/chkpt/` automatically when validation metric improves.

---

## Inference on Unseen Geometries

All PDEs are evaluated through a single unified script `inference.py`. The PDE type is automatically inferred from the `--dataset` argument.

**Run all commands from the `sni/` directory.**

```bash
# Laplace equation (Dirichlet BC)
python inference.py --dataset laplace2d_schwarz --model-path <path> --tau 0.1

# Laplace equation (mixed Dirichlet/Neumann BC)
python inference.py --dataset laplace2d_mixed_holes --model-path <path> --tau 0.1

# Darcy flow
python inference.py --dataset darcy2d_bosch --model-path <path> --tau 0.1

# Nonlinear Poisson
python inference.py --dataset nonlinear_poisson2d_schwarz --model-path <path> --tau 0.1

# Heat equation (time-dependent)
python inference.py --dataset heat2d_schwarz --model-path <path> --tau 0.1 \
    --time-span 10 --time-step 5
```

### Inference CLI Options

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name (encodes PDE + domain, e.g. `darcy2d_bosch`) | `laplace2d_schwarz` |
| `--model-path` | Path to trained model checkpoint | — |
| `--tau` | Schwarz iteration relaxation parameter | — |
| `--n-parts` | Number of subdomains | `10` |
| `--depth` | Extension depth for overlapping partitions | `2` |
| `--epochs` | Max Schwarz iterations | `5000` |
| `--time-span` | Total time steps (heat2d only) | `1` |
| `--time-step` | Basis output dimension (heat2d only) | `5` |
| `--gpu` | GPU device id | `0` |
| `--no-cuda` | Disable CUDA | `False` |

### Supported Datasets

| PDE | Available Datasets |
|---|---|
| Laplace2d (Dirichlet) | `laplace2d_schwarz`, `laplace2d_holes`, `laplace2d_bosch`, `laplace2d_dolphin`, `laplace2d_disk` |
| Laplace2d (Mixed) | `laplace2d_mixed_schwarz`, `laplace2d_mixed_holes`, `laplace2d_mixed_bosch` |
| Darcy2d | `darcy2d_schwarz`, `darcy2d_holes`, `darcy2d_bosch`, `darcy2d_negative_triangle` |
| Heat2d | `heat2d_schwarz`, `heat2d_holes`, `heat2d_bosch` |
| Nonlinear Poisson2d | `nonlinear_poisson2d_schwarz`, `nonlinear_poisson2d_holes`, `nonlinear_poisson2d_bosch` |

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