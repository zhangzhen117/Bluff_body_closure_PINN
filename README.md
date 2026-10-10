# Bluff-body turbulence closures trained by a PINN

Code and data for *Generalizable turbulence closures across bluff-body shapes by PINN-based
solver-agnostic training* ([arXiv:2607.04491](https://arxiv.org/abs/2607.04491); Journal of Fluid Mechanics, under review).

Closures for the steady two-dimensional RANS equations are identified with a physics-informed
neural network (PINN). Training uses only the RANS residual and observations, with no flow solver
in the loop. The trained closure is then frozen and deployed in a finite-element RANS solver
(FEniCSx). Four closures are provided:

| tag   | model | closure |
|-------|-------|---------|
| `loc` | M1 | local tensor-basis Reynolds stress |
| `alg` | M2 | transported $k$, algebraic length scale $\ell$ |
| `nn`  | M3 | transported $k$, learned length scale $\ell$ |
| `fc`  | M4 | Reynolds force $\mathbf F=-\nabla\cdot\boldsymbol\tau$ with eddy viscosity and energy-neutral residual |

## Repository layout

```
pinn/             PINN training (JAX): networks, RANS residuals, losses, sampling, SOAP optimizer
  train.py          entry point
  config.py         shapes, splits, paths and all default hyperparameters
fem/              frozen-closure deployment (FEniCSx)
  rans_fem.py       M1 (local) solver
  rans_fem_nl.py    M2/M3 solver with transported k
  rans_fem_force.py M4 (force) solver
  deployment_checks.py  physical input filter, residual monitor, convergence criterion
  positive_k.py     positivity-preserving k operator (graph diffusion, lumped reaction)
  newton_solver.py  Newton-GMRES fallback for M2-M4
  mesh2d.py         body-fitted gmsh meshes
postprocessing/
  evaluate.py       relative L2 errors against DNS and control-volume drag
  plot_sixway.py    six-way field figures (DNS, M1-M4, SST)
scripts/          train.sh and deploy.sh with the settings used in the paper
checkpoints/      trained closures used in the paper (see below)
data/             DNS, PIV and SST data (see below)
environment/      conda environments (jax.yml for training/postprocessing, fenicsx.yml for deployment)
results/          output of the deployment and postprocessing
```

## Installation

```bash
conda env create -f environment/jax.yml        # training and postprocessing (CUDA 12 GPU build of JAX)
conda env create -f environment/fenicsx.yml    # deployment (FEniCSx 0.9, gmsh, PETSc)
```

## Data

All quantities are non-dimensionalized by the body size $D$ and the free-stream velocity
$U_\infty$; $\mathrm{Re}=U_\infty D/\nu=10^4$.

| file | content |
|------|---------|
| `data/dns/circle.csv.gz`, `square_45_degree.csv.gz` (diamond), `tri_long_0_dns.csv.gz` (long triangle) | time- and spanwise-averaged DNS (nekRS) on the spectral-element points: `x,y,meanU,meanV,meanP,uu,uv,vv,ww,pp` |
| `data/piv/circle.h5` | patched PIV mean field of the circular cylinder on a structured grid: `x,y,U,V,uu,uv,vv,mask` (`mask=1` marks valid data) |
| `data/sst/circle.npz` | steady SST $k$–$\omega$ solution (OpenFOAM) of the circular cylinder on a structured grid: `X,Y,U,V,speed,p,k,nut,valid` |

The DNS of the square, the ellipse and the equilateral triangle, and the PIV data of the other
shapes, are available from the corresponding author on request. Leave-one-shape-out training
(`lo`) uses all six DNS shapes, so it needs these files as well
(`data/dns/<shape>.csv.gz`, names in `pinn/config.py`).

## Trained closures

`checkpoints/` contains the closures deployed in the paper for the three shapes distributed here:
`<protocol>_<model>_<code>_params.npz`, with protocol `is` (in-sample) or `lo` (leave-one-shape-out),
model `loc|alg|nn|fc` and shape code `cir` (circle), `dia` (diamond), `trl` (long triangle), plus the
PIV-trained cylinder closures `is_<model>_piv_cir`. They can be deployed directly.

## Usage

```bash
# 1. train (JAX environment, one GPU; ~15 min in-sample on an NVIDIA L40S)
bash scripts/train.sh nn circle is          # M3, in-sample on the cylinder
bash scripts/train.sh fc circle piv         # M4, trained on the PIV cylinder

# 2. deploy the frozen closure (FEniCSx environment, CPU; a few minutes)
bash scripts/deploy.sh nn circle is

# 3. errors against DNS, drag and figures (JAX environment)
python postprocessing/evaluate.py --mode is --shapes circle
python postprocessing/plot_sixway.py --shape circle --mode is
```

Training overwrites the checkpoint with the same tag. The deployment settings are those of the
paper: medium mesh (wall, wake and far-field spacings 0.02, 0.08, 0.8), physical input filter
$\delta_f/D = C_f/(1+sD/U_\infty)$ with $C_f=0.2$, SUPG on momentum, mass-weighted residuals
relative to their initial values below $10^{-5}$, at most 500 Picard iterations. M2–M4 use the
positivity-preserving $k$ operator and fall back to Newton–GMRES if Picard does not converge.
`plot_sixway.py` uses the M4 field as the reference grid, so deploy M4 first.

## Citation

If you use this code or data, please cite

> Z. Zhang, T. Käufer, L. Ronglan, M. S. Triantafyllou and G. E. Karniadakis,
> Generalizable turbulence closures across bluff-body shapes by PINN-based solver-agnostic training,
> arXiv:2607.04491 (2026). https://arxiv.org/abs/2607.04491

```bibtex
@article{zhang2026generalizable,
  title         = {Generalizable turbulence closures across bluff-body shapes by {PINN}-based solver-agnostic training},
  author        = {Zhang, Zhen and K{\"a}ufer, Theo and Ronglan, Louise and Triantafyllou, Michael S. and Karniadakis, George Em},
  journal       = {arXiv preprint arXiv:2607.04491},
  year          = {2026},
  eprint        = {2607.04491},
  archivePrefix = {arXiv},
  primaryClass  = {physics.flu-dyn},
  url           = {https://arxiv.org/abs/2607.04491}
}
```

## License

The code is released under the MIT License (see `LICENSE`).
