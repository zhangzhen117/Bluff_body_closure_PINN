#!/bin/bash
# Train one closure with the settings of the paper (50 000 SOAP steps, 2048 collocation and
# 2048 data points per shape and step). Run in the JAX environment (environment/jax.yml).
#
#   bash scripts/train.sh <model> <shape> [protocol] [steps]
#     model     loc (M1, local) | alg (M2, algebraic ell) | nn (M3, learned ell) | fc (M4, force)
#     shape     circle | square_45_degree | tri_long_0_dns   (DNS distributed with this repository)
#               square_0_degree | ellipse_0_degree | tri_equi_0_dns   (DNS on request)
#     protocol  is  (in-sample, default) | lo (leave-one-shape-out; needs all six DNS files)
#               piv (in-sample on the PIV cylinder, shape must be circle)
#
# Writes checkpoints/<protocol>_<model>_<code>_params.npz, the tag read by scripts/deploy.sh.
set -euo pipefail
cd "$(dirname "$0")/../pinn"
model=$1; shape=$2; protocol=${3:-is}; steps=${4:-50000}

declare -A CODE=([circle]=cir [square_0_degree]=sq0 [ellipse_0_degree]=ell
                 [tri_long_0_dns]=trl [tri_equi_0_dns]=tre [square_45_degree]=dia)
declare -A SPLIT=([circle]=cylinder [square_0_degree]=square0_dns [ellipse_0_degree]=ellipse0_dns
                  [tri_long_0_dns]=tri_dns [tri_equi_0_dns]=triequi_dns [square_45_degree]=diamond)
case "$model" in
  loc) flags="--lip_weight 0.03" ;;
  alg) flags="--use_keq 1 --use_ww 1 --nut_kl 1 --lip_weight 0.03 --k_weight 1.0 --tau_weight 0.1" ;;
  nn)  flags="--use_keq 1 --use_ww 1 --nut_kl 1 --ell_nn 1 --lip_weight 0.03 --k_weight 1.0 --tau_weight 0.1" ;;
  fc)  flags="--use_keq 1 --use_ww 1 --nut_kl 1 --ell_nn 1 --use_force 1 --force_proj 1 --lip_weight 0.03 --k_weight 1.0 --force_sup 1.0" ;;
  *) echo "unknown model $model"; exit 1 ;;
esac
common="--steps $steps --n_coll 2048 --n_data 2048"

case "$protocol" in
  is)  python -u train.py --tag "is_${model}_${CODE[$shape]}" --split "${SPLIT[$shape]}" \
           --dns_csv "../data/dns/${shape}.csv.gz" $flags $common ;;
  lo)  python -u train.py --tag "lo_${model}_${CODE[$shape]}" --split dns_loso --loso_shape "$shape" \
           --use_dns $flags $common ;;
  piv) # PIV gives no pressure and no out-of-plane stress: no tau_zz output, no force supervision
       flags=${flags/--use_ww 1/--use_ww 0}; flags=${flags/--force_sup 1.0/--force_sup 0}
       python -u train.py --tag "is_${model}_piv_cir" --split cylinder $flags $common ;;
  *) echo "unknown protocol $protocol"; exit 1 ;;
esac
