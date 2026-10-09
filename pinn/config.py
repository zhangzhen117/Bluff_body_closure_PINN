"""Configuration for the shape-generalizing steady-RANS closure model.

Single source of truth for: shape registry, train/test splits, the unconfined
computational domain, normalization constants, and default hyperparameters.
"""
import os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # repository root
DATA_DIR = os.path.join(ROOT, "data", "piv")         # PIV mean fields <shape>.h5
DNS_DIR = os.path.join(ROOT, "data", "dns")          # DNS spanwise/time averages <shape>.csv.gz
SST_DIR = os.path.join(ROOT, "data", "sst")          # SST k-omega fields <shape>.npz
CKPT_DIR = os.path.join(ROOT, "checkpoints")         # trained parameters <tag>_params.npz
RESULTS_DIR = os.path.join(ROOT, "results")          # FEM fields and evaluation tables

# ---- physical / normalization constants (data already normalized D=1, U_inf=1) ----
NU = 1e-4            # kinematic viscosity, Re = U*D/nu = 1e4
RE = 1.0e4

# ---- unconfined computational domain (much larger than the PIV window) ----
# inlet at X0 (u=1,v=0); symmetry at y=+-Y1; zero-pressure outlet at X1.
X0, X1 = -10.0, 15.0
Y0, Y1 = -10.0, 10.0

# ---- shapes ----
# NOTE: square_0_degree is EXCLUDED (the PIV experiment for it has an issue).
ALL_SHAPES = [
    "circle",
    "ellipse_0_degree", "ellipse_90_degree",
    "long_triangle_0_degree", "long_triangle_180_degree",
    "short_triangle_0_degree", "short_triangle_180_degree",
    "square_45_degree",
]

# train/test splits (config-driven). Stage 1 = cylinder only.
SPLITS = {
    # Stage 1 sanity check: train on cylinder, test on cylinder.
    "cylinder": {
        "train": ["circle"],
        "test": ["circle"],
    },
    # Stage 2 default: canonical orientations train, rotated/unseen variants test.
    # (square_0_degree excluded due to a data issue; square_45 stays as a test shape.)
    "canonical": {
        "train": ["circle", "ellipse_0_degree", "long_triangle_0_degree",
                  "short_triangle_0_degree"],
        "test": ["ellipse_90_degree", "long_triangle_180_degree",
                 "short_triangle_180_degree", "square_45_degree"],
    },
    # single-shape DNS benchmark: long triangle (sharp-corner test)
    "tri_dns": {
        "train": ["tri_long_0_dns"],
        "test": ["tri_long_0_dns"],
    },
    # single PIV shape: diamond (square rotated 45 deg) -- transfer-learning target
    "diamond": {
        "train": ["square_45_degree"],
        "test": ["square_45_degree"],
    },
    # single-shape DNS benchmarks (use with --dns_csv dns/<file>)
    "square0_dns": {"train": ["square_0_degree"], "test": ["square_0_degree"]},
    "ellipse0_dns": {"train": ["ellipse_0_degree"], "test": ["ellipse_0_degree"]},
    "triequi_dns": {"train": ["tri_equi_0_dns"], "test": ["tri_equi_0_dns"]},
    # multi-shape DNS generalization (use with --use_dns): train 3, test 3.
    "dns3": {
        "train": ["circle", "square_0_degree", "tri_long_0_dns"],
        "test": ["ellipse_0_degree", "square_45_degree", "tri_equi_0_dns"],
    },
    # in-sample Phase-B test: frozen dns3 closure re-solves its OWN training shapes
    "dns3_insample": {
        "train": ["circle", "square_0_degree", "tri_long_0_dns"],
        "test": ["circle", "square_0_degree", "tri_long_0_dns"],
    },
    # in-sample check: frozen-closure solve ON the training shapes themselves
    "canonical_insample": {
        "train": ["circle", "ellipse_0_degree", "long_triangle_0_degree",
                  "short_triangle_0_degree"],
        "test": ["circle", "ellipse_0_degree", "long_triangle_0_degree",
                 "short_triangle_0_degree"],
    },
}

# DNS file of each shape in DNS_DIR. Only circle, square_45_degree (diamond) and tri_long_0_dns
# are distributed with this repository; the other three are available on request.
DNS_LINKS = {
    "circle": "circle.csv.gz",
    "tri_long_0_dns": "tri_long_0_dns.csv.gz",
    "square_45_degree": "square_45_degree.csv.gz",
    "square_0_degree": "square_0_degree.csv.gz",
    "ellipse_0_degree": "ellipse_0_degree.csv.gz",
    "tri_equi_0_dns": "tri_equi_0_dns.csv.gz",
}


def dns_path(shape):
    return os.path.join(DNS_DIR, DNS_LINKS[shape])


# the 6 shapes with DNS data (for leave-one-out generalization, use with --use_dns)
DNS_SHAPES = ["circle", "square_0_degree", "ellipse_0_degree",
              "tri_long_0_dns", "tri_equi_0_dns", "square_45_degree"]


def split_lists(split, loso_shape=None):
    """Return (train_shapes, test_shapes) for a named split.

    split='loso' / 'dns_loso' do leave-one-shape-out over ALL_SHAPES / DNS_SHAPES:
    train on the rest, test on `loso_shape`."""
    if split == "loso":
        assert loso_shape in ALL_SHAPES, f"--loso_shape must be in {ALL_SHAPES}"
        return [s for s in ALL_SHAPES if s != loso_shape], [loso_shape]
    if split == "dns_loso":
        assert loso_shape in DNS_SHAPES, f"--loso_shape must be in {DNS_SHAPES}"
        return [s for s in DNS_SHAPES if s != loso_shape], [loso_shape]
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choices: {list(SPLITS)+['loso']}")
    return list(SPLITS[split]["train"]), list(SPLITS[split]["test"])


# ---- default hyperparameters (overridable via CLI in train.py/eval.py) ----
DEFAULTS = dict(
    # mean-field net (per shape)
    mf_hidden=64, mf_layers=4, mf_embed_scale=2.0, mf_embed_dim=64,
    # shared closure net
    cl_hidden=24, cl_layers=2,
    # closure input limiter f/(|f|+ref) reference values (~ typical magnitude)
    ref_grad=1.0, ref_inv=1.0, ref_sdf=0.5,
    g2_scale=1.0,     # scale on the energy-neutral residual coefficient g2
    tau_weight=0.1,   # multiplier on the tau-consistency terms so the PDE/data win
    lip_weight=0.03,  # unified closure-Lipschitz penalty ||d(closure coeffs)/d(grad u)||_F^2 for ALL
                      #   methods M1-M4 (a single physically-meaningful Lipschitz constant; 0=off). For
                      #   the force model the coeffs include the residual force (Rx,Ry), so this also
                      #   subsumes the former force_reg gain penalty.
    pde_weight=1.0,   # multiplier on the PDE-residual loss terms (set 0 for pure-supervised ablation)
    nut_max=0.0,      # if >0, nu_t = nut_max*sigmoid (bounded) instead of softplus (unbounded)
    gate_w0=0.0,      # if >0, gate (k,nu_t,g2) by tanh(|Omega|/gate_w0)^2 -> 0 in irrotational regions
    wall_d0=0.0,      # if >0, wall-damp the closure by tanh(sdf/wall_d0) IN training (deployment-consistent)
    # --- non-local k-equation closure (use_keq): k is a transported field (4th mean-field
    #     output) feeding nu_t,g2; the closure NN outputs only (nu_t,g2). Local model untouched. ---
    use_keq=False,    # master flag: transported-k (non-local) closure
    use_ww=False,     # (requires use_keq) full 3D TKE k=1/2(uu+vv+ww): closure gains a 3rd
                      # output G0 for the T0=diag(1/6,1/6,-1/3) redistribution tensor; tau_zz
                      # is reconstructed (tau=(2/3)k I -2nu_t S + g2 T2 + G0 T0) and supervised
    kappa=0.41,       # von Karman constant; mixing length ell = min(kappa*max(d,d_floor), ell_max)
    ell_max=1.5,      # far-field cap on the mixing length (so k^1.5/ell doesn't vanish in freestream)
    d_floor=1e-3,     # near-wall floor on d (so k^1.5/ell isn't singular at the wall)
    # learned mixing length (requires use_keq): ell = clip(kappa*max(d,d_floor)*exp(ell_span*tanh(NN)),
    #   d_floor, ell_cap) -- a bounded log-correction on the kappa*d baseline (the closure gains a
    #   final output). Keeps near-wall scaling, lets the wake length scale grow (less destruction).
    learn_ell=False,
    ell_span=1.386,   # tanh -> factor exp(+-1.386) = [0.25, 4.0] around the kappa*d baseline
    ell_cap=6.0,      # cap on the LEARNED mixing length (larger than ell_max so the wake ell can grow)
    # ell_nn: learn ell from a SEPARATE small net fed only the 7 kinematic features (NO k), so the
    #   destruction k^{3/2}/ell stays cleanly monotonic in k. Same bounded log-correction on kappa*d.
    ell_nn=False,
    ell_hidden=16, ell_layers=2,
    # Prandtl-Kolmogorov eddy viscosity (requires use_keq): nu_t = softplus(NN)*sqrt(k+k_eps)*ell
    #   instead of a free softplus(NN). Anchors nu_t to the smooth transported k and the length
    #   scale -> nu_t->0 where k->0 (freestream/wall), the NN only learns a bounded C_mu-like factor.
    nut_kl=False,
    k_eps=1e-6,       # floor inside sqrt(k) so nu_t is Lipschitz at k=0
    # calibrated k-l constants (active with nut_kl): the standard model has
    #   nu_t = cmu^{1/4} sqrt(k) l   and   destruction = cmu^{3/4} k^{3/2}/l.
    # cmu^{1/4}~0.55 centers the NN multiplier; cmu^{3/4}~0.16 in destruction fixes the
    # ~6x over-destruction of the bare k^{3/2}/l (the likely cause of wake-k under-prediction).
    cmu=0.09,
    cnu_span=1.609,   # NN multiplier exp(cnu_span*tanh) in [0.2, 5.0] around cmu^{1/4}
    ref_k=0.1,        # rational-limiter reference for the k input feature
    k_weight=1.0,     # weight on the k-equation residual / k-data / k-BC loss terms
    # forcing-vector closure (use_force, with use_keq+nut_kl+ell_nn): the closure outputs the
    #   mean-momentum Reynolds force F=(Fx,Fy) directly (slots 2,3) instead of (g2,G0); the
    #   momentum residual uses -F instead of div(tau). nu_t (PK) is kept only for the k-equation.
    use_force=False,
    force_scale=1.0,  # output scale on the predicted (Fx, Fy)
    # NOTE: the former per-force gain/magnitude penalties (force_reg, force_mag) are retired -- the
    #   residual-force gain is now regularized by the unified `lip_weight` above (closure_fwd returns
    #   (Rx,Ry) in the coefficient slots for the force model), and `force_proj` supplies the
    #   dissipativity guarantee that makes the underdetermined nu_t/R split well-posed.
    # energy-consistent force projection (force_proj): split the closure force into a DISSIPATIVE
    #   channel (the eddy stress -2 nu_t S, nu_t>=0, which drains mean KE: -2 nu_t |S|^2 <= 0) and
    #   an ENERGY-NEUTRAL residual R projected perpendicular to the mean velocity (R.U=0, so it does
    #   no work on the mean flow -- pure momentum redistribution). This removes the anti-dissipative
    #   (backscatter) component of R that destabilizes the a-posteriori fixed point on sharp bodies,
    #   making the deployed closure provably mean-KE dissipative. The DNS force is ~70% perpendicular
    #   and ~95% dissipative already, so the projection keeps the dominant physics.
    force_proj=False,
    force_proj_eps=1e-3,   # |U|^2 floor in the perpendicular projection (smooth at stagnation)
    # direct DNS-force supervision (force_sup): weight on ||F_closure - F_DNS||^2 at data points,
    #   where F_closure = div(2 nu_t S) + R is the model's total Reynolds force and F_DNS = -div<u'u'>
    #   is the filtered measured force (data.data_F). Complements the residual-driven (implicit) target
    #   with a direct local one, pinning the dissipative/neutral split where the PDE residual is weak.
    force_sup=0.0,
    # batch sizes (per shape, per step) -- pinned to the campaign recipe (n_coll=2048)
    n_coll=2048, n_data=2048, n_bc_wall=512, n_bc_in=256, n_bc_out=256, n_bc_sym=256,
    near_wall_band=0.3,        # collocation densification band 0<sdf<band
    # optimizer
    lr=3e-3, warmup=2000, decay_steps=4000, decay_rate=0.9, precond_freq=5,
    w_update_every=500, w_momentum=0.9,
    pt_update_every=200, pt_momentum=0.7,
    # schedule -- pinned: all M1-M4 campaigns train 50k steps
    steps=50000, eval_steps=40000, seed=42,
    # ablation knobs (analysis only; defaults reproduce the pinned behaviour):
    opt="soap",          # optimizer: "soap" (default) or "adam"
    arch="modified",     # network: "modified" (ModifiedMlp, default) or "mlp" (plain MLP)
    no_gradnorm=0,       # 1 -> disable grad-norm adaptive loss-weight balancing (uniform weights)
    no_pt=0,             # 1 -> disable pseudo-time stepping (theta held at 1)
    err_every=5000,   # print field rel-L2 (all/near-wall) vs data every N steps
)


def base_cfg(**overrides):
    """A mutable namespace of the default hyperparameters, with overrides."""
    d = dict(DEFAULTS)
    d.update(overrides)
    return SimpleNamespace(**d)
