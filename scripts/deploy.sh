#!/bin/bash
# Deploy a trained closure, frozen, in the FEniCSx RANS solver with the settings of the paper:
# medium mesh, physical input filter C_f = 0.2, SUPG on momentum, at most 500 Picard iterations,
# relative residual threshold 1e-5. M2-M4 solve a positivity-preserving k equation and fall back
# to Newton-GMRES if Picard does not converge. Run in the FEniCSx environment (environment/fenicsx.yml).
#
#   bash scripts/deploy.sh <model> <shape> [is|lo|piv]      e.g.  bash scripts/deploy.sh fc circle is
#
# Reads checkpoints/<protocol>_<model>_<code>_params.npz; writes results/fem/<shape>_<tag>_*.npz
# and the convergence history <shape>_<tag>_*_convergence.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/fem"
model=$1; shape=$2; protocol=${3:-is}
declare -A CODE=([circle]=cir [square_0_degree]=sq0 [ellipse_0_degree]=ell
                 [tri_long_0_dns]=trl [tri_equi_0_dns]=tre [square_45_degree]=dia)
tag="${protocol}_${model}_${CODE[$shape]}"
[ "$protocol" = piv ] && tag="is_${model}_piv_cir"
case "$model" in
  loc) solver=rans_fem.py;       flags="--filt 0 --out_filt 0.03" ;;
  alg) solver=rans_fem_nl.py;    flags="--nut_kl --use_ww" ;;
  nn)  solver=rans_fem_nl.py;    flags="--nut_kl --use_ww --ell_nn" ;;
  fc)  solver=rans_fem_force.py; flags="--nut_kl --ell_nn --force_proj --filt 0.05" ;;
  *) echo "unknown model $model"; exit 1 ;;
esac
[ "$protocol" = piv ] && flags=${flags/--use_ww/}       # PIV-trained closures have no tau_zz output
out="$ROOT/results/fem"
mkdir -p "$out"
python -u "$solver" --shape "$shape" --ckpt "$tag" $flags \
  --filter_mode strain --filter_cf 0.2 --supg --picard 500 --tol 1e-4 \
  --residual_rtol 1e-5 --residual_atol 1e-8 \
  --hwall 0.02 --wake_h 0.08 --hfar 0.8 --output_dir "$out"
