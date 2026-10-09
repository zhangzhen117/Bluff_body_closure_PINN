"""Pure-numpy port of the tensor-basis closure (no JAX), for use in the FEniCS
solver. Loads the trained closure weights from history/<tag>_params.npz and
reproduces model.py exactly: ModifiedMlp forward -> (k, nu_t, g2) -> build_tau.

Verify against the JAX model with `python closure_np.py --verify` (in an env that
has both jax and numpy, e.g. jaxpi).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinn"))
import config  # pure (os + SimpleNamespace), importable without jax

CFG = config.base_cfg()   # ref_grad, ref_inv, ref_sdf, g2_scale (defaults)


def load_closure(npz_path):
    z = np.load(npz_path)
    pre = "closure/params/"
    W = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
    return W


def _softplus(x):
    return np.logaddexp(0.0, x)


def _lim(f, ref):
    return f / (np.abs(f) + ref)


def features(ux, uy, vx, vy, d, cfg=CFG):
    s_mag = np.sqrt(ux ** 2 + vy ** 2 + 0.5 * (uy + vx) ** 2 + 1e-12)
    w_mag = np.sqrt(0.5 * (uy - vx) ** 2 + 1e-12)
    return np.stack([_lim(ux, cfg.ref_grad), _lim(uy, cfg.ref_grad),
                     _lim(vx, cfg.ref_grad), _lim(vy, cfg.ref_grad),
                     _lim(s_mag, cfg.ref_inv), _lim(w_mag, cfg.ref_inv),
                     _lim(d, cfg.ref_sdf)], axis=-1)   # (...,7)


def _mlp(feat, W):
    u = np.tanh(feat @ W["Dense_0/kernel"] + W["Dense_0/bias"])
    v = np.tanh(feat @ W["Dense_1/kernel"] + W["Dense_1/bias"])
    x = feat
    for L in ["Dense_2", "Dense_3"]:        # num_layers=2 gated loop
        x = np.tanh(x @ W[L + "/kernel"] + W[L + "/bias"])
        x = x * u + (1.0 - x) * v
    return x @ W["Dense_4/kernel"] + W["Dense_4/bias"]   # (...,3)


def coeffs(W, ux, uy, vx, vy, d, cfg=CFG):
    """-> (k, nu_t, g2), each same shape as inputs. Mirrors model.closure_fwd
    including optional bounded nu_t / wall-damping / turbulence gate."""
    out = _mlp(features(ux, uy, vx, vy, d, cfg), W)
    k = _softplus(out[..., 0])
    nut_max = getattr(cfg, "nut_max", 0.0)
    nu_t = nut_max / (1.0 + np.exp(-out[..., 1])) if nut_max > 0 else _softplus(out[..., 1])
    g2 = cfg.g2_scale * np.tanh(out[..., 2])
    if getattr(cfg, "wall_d0", 0.0) > 0:
        phi = np.tanh(np.clip(d, 0.0, None) / cfg.wall_d0)
        k, nu_t, g2 = phi * k, phi * nu_t, phi * g2
    if getattr(cfg, "gate_w0", 0.0) > 0:
        enst = 0.5 * (uy - vx) ** 2                  # |Omega|^2 (smooth, no sqrt -> Lipschitz-safe)
        gate = np.tanh(enst / cfg.gate_w0 ** 2)
        k, nu_t, g2 = gate * k, gate * nu_t, gate * g2
    return k, nu_t, g2


def features_nl(ux, uy, vx, vy, d, k, cfg=CFG):
    """8 features: the 7 local ones + the transported-k feature (use_keq)."""
    base = features(ux, uy, vx, vy, d, cfg)              # (...,7)
    kf = _lim(k, cfg.ref_k)[..., None]                   # (...,1)
    return np.concatenate([base, kf], axis=-1)           # (...,8)


def _ell_mlp(feat, W, layers):
    """Plain tanh-MLP forward for the k-free ell subnet (ell_nn)."""
    x = feat
    for i in range(layers):
        x = np.tanh(x @ W[f"Dense_{i}/kernel"] + W[f"Dense_{i}/bias"])
    return x @ W[f"Dense_{layers}/kernel"] + W[f"Dense_{layers}/bias"]   # (...,1)


def coeffs_nl(W, ux, uy, vx, vy, d, k, cfg=CFG):
    """Non-local closure (use_keq): 8-feature head -> (nu_t, g2[, G0][, ell-corr]).
    k is the FEM transported field (NOT from the NN). Returns (k, nu_t, g2, g0, ell);
    g0=0 unless use_ww; ell is algebraic unless learn_ell/ell_nn. Mirrors model.closure_fwd."""
    feats8 = features_nl(ux, uy, vx, vy, d, k, cfg)
    if getattr(cfg, "ell_nn", False):                  # composite: main(8 feat) + k-free ell subnet
        mainW = {kk[len("ModifiedMlp_0/"):]: v for kk, v in W.items() if kk.startswith("ModifiedMlp_0/")}
        ellW = {kk: v for kk, v in W.items() if not kk.startswith("ModifiedMlp_0/")}
        feats7 = features(ux, uy, vx, vy, d, cfg)      # kinematic only (no k)
        out = np.concatenate([_mlp(feats8, mainW), _ell_mlp(feats7, ellW, cfg.ell_layers)], axis=-1)
    else:
        out = _mlp(feats8, W)                          # (...,2..4)
    g2 = cfg.g2_scale * np.tanh(out[..., 1])
    i = 2
    if getattr(cfg, "use_ww", False):
        g0 = 2.0 * k * np.tanh(out[..., i]); i += 1
    else:
        g0 = np.zeros_like(g2)
    learn_l = getattr(cfg, "learn_ell", False) or getattr(cfg, "ell_nn", False)
    ell = mixing_length(d, cfg, out[..., i] if learn_l else None)
    nut_max = getattr(cfg, "nut_max", 0.0)
    if getattr(cfg, "nut_kl", False):                       # Prandtl-Kolmogorov nu_t
        mult = cfg.cmu ** 0.25 * np.exp(cfg.cnu_span * np.tanh(out[..., 0]))
        nu_t = mult * np.sqrt(np.maximum(k, 0.0) + cfg.k_eps) * ell
    elif nut_max > 0:
        nu_t = nut_max / (1.0 + np.exp(-out[..., 0]))
    else:
        nu_t = _softplus(out[..., 0])
    if getattr(cfg, "wall_d0", 0.0) > 0:
        phi = np.tanh(np.clip(d, 0.0, None) / cfg.wall_d0)
        nu_t, g2 = phi * nu_t, phi * g2
    if getattr(cfg, "gate_w0", 0.0) > 0:
        enst = 0.5 * (uy - vx) ** 2
        gate = np.tanh(enst / cfg.gate_w0 ** 2)
        nu_t, g2 = gate * nu_t, gate * g2
    return k, nu_t, g2, g0, ell


def coeffs_force(W, ux, uy, vx, vy, d, k, cfg=CFG, u=None, v=None):
    """Forcing-vector closure (use_force): 8-feature head -> (m, Fx, Fy) [+ k-free ell subnet].
    Returns (Fx, Fy, nu_t, ell). nu_t (Prandtl-Kolmogorov) is used only by the k-equation; the
    momentum force is (Fx, Fy). With cfg.force_proj and the mean velocity (u,v) supplied, the
    residual force is projected perpendicular to U (R.U=0, energy-neutral). Mirrors closure_fwd."""
    feats8 = features_nl(ux, uy, vx, vy, d, k, cfg)
    if getattr(cfg, "ell_nn", False):                  # composite: main(8 feat) + k-free ell subnet
        mainW = {kk[len("ModifiedMlp_0/"):]: v for kk, v in W.items() if kk.startswith("ModifiedMlp_0/")}
        ellW = {kk: v for kk, v in W.items() if not kk.startswith("ModifiedMlp_0/")}
        feats7 = features(ux, uy, vx, vy, d, cfg)
        out = np.concatenate([_mlp(feats8, mainW), _ell_mlp(feats7, ellW, cfg.ell_layers)], axis=-1)
    else:
        out = _mlp(feats8, W)
    fs = getattr(cfg, "force_scale", 1.0)
    Fx, Fy = fs * out[..., 1], fs * out[..., 2]
    if getattr(cfg, "force_proj", False) and u is not None:     # energy-neutral: project R perp U
        eps = getattr(cfg, "force_proj_eps", 1e-3)
        w = (Fx * u + Fy * v) / (u * u + v * v + eps)
        Fx, Fy = Fx - w * u, Fy - w * v
    learn_l = getattr(cfg, "learn_ell", False) or getattr(cfg, "ell_nn", False)
    ell = mixing_length(d, cfg, out[..., 3] if learn_l else None)
    if getattr(cfg, "nut_kl", False):
        mult = cfg.cmu ** 0.25 * np.exp(cfg.cnu_span * np.tanh(out[..., 0]))
        nu_t = mult * np.sqrt(np.maximum(k, 0.0) + cfg.k_eps) * ell
    else:
        nu_t = _softplus(out[..., 0])
    return Fx, Fy, nu_t, ell


def build_tau_ww(k, nu_t, g2, g0, ux, uy, vx, vy):
    """3D-aware realizable stress for 2D mean flow (mirror model.build_tau_ww).
    Returns (tau_uu, tau_uv, tau_vv, tau_zz)."""
    a = 0.5 * (ux - vy); b = 0.5 * (uy + vx); w = 0.5 * (uy - vx)
    iso = (2.0 / 3.0) * k + g0 / 6.0
    B11 = -2.0 * nu_t * a + g2 * (-2.0 * b * w)
    B12 = -2.0 * nu_t * b + g2 * (2.0 * a * w)
    lam = np.sqrt(B11 ** 2 + B12 ** 2 + 1e-30)
    scale = iso * np.tanh(lam / (iso + 1e-9)) / (lam + 1e-30)
    B11, B12 = B11 * scale, B12 * scale
    tau_zz = (2.0 / 3.0) * k - g0 / 3.0
    return iso + B11, B12, iso - B11, tau_zz


def mixing_length(d, cfg=CFG, ell_out=None):
    """Mixing length (mirror model.mixing_length). Algebraic min(kappa*max(d,d_floor),
    ell_max), or with learn_ell a bounded log-correction on the kappa*d baseline."""
    base = cfg.kappa * np.maximum(d, cfg.d_floor)
    if ell_out is None:
        return np.minimum(base, cfg.ell_max)
    return np.clip(base * np.exp(cfg.ell_span * np.tanh(ell_out)), cfg.d_floor, cfg.ell_cap)


def build_tau(k, nu_t, g2, ux, uy, vx, vy):
    """Realizable tensor-basis tau = k*I - 2*nu_t*S + g2*(S*Om-Om*S) (mirror model.build_tau)."""
    a = 0.5 * (ux - vy); b = 0.5 * (uy + vx); w = 0.5 * (uy - vx)
    B11 = -2.0 * nu_t * a + g2 * (-2.0 * b * w)
    B12 = -2.0 * nu_t * b + g2 * (2.0 * a * w)
    lam = np.sqrt(B11 ** 2 + B12 ** 2 + 1e-30)
    scale = k * np.tanh(lam / (k + 1e-9)) / (lam + 1e-30)
    B11, B12 = B11 * scale, B12 * scale
    return k + B11, B12, k - B11   # tau_uu, tau_uv, tau_vv


def wall_damp(d, d0=0.12):
    """Near-wall damping of the closure: phi -> 0 at the surface (d=0), -> 1 for
    d >> d0. Turbulence vanishes at the wall and both the PINN mean field and the
    closure are least reliable in that thin band, so we taper the Reynolds stress
    there. d is the (positive-in-fluid) signed distance / SDF."""
    return np.tanh(np.clip(d, 0.0, None) / d0)


def _verify(npz_path):
    import jax, jax.numpy as jnp
    import model as M
    from jax import vmap, random
    cfg = config.base_cfg()
    CL = M.make_closure_arch(cfg)
    cl = M.__dict__  # placeholder
    # load JAX closure params from the same npz via common
    import common
    clp = common.load_params(npz_path, {"closure": CL.init(random.PRNGKey(0), jnp.ones(7))})["closure"]
    W = load_closure(npz_path)
    rng = np.random.default_rng(0)
    ux, uy, vx, vy, d = [rng.normal(0, 2, 5000) for _ in range(5)]
    # numpy
    kn, nn_, gn = coeffs(W, ux, uy, vx, vy, d, cfg)
    tuu_n, tuv_n, tvv_n = build_tau(kn, nn_, gn, ux, uy, vx, vy)
    # jax: tau_at needs a mean-field; instead call closure_fwd + build_tau directly
    def jx(uxi, uyi, vxi, vyi, di):
        k, nu, g, _, _ = M.closure_fwd(CL, clp, uxi, uyi, vxi, vyi, di, cfg)
        return jnp.array([k, nu, g, *M.build_tau(k, nu, g, uxi, uyi, vxi, vyi)])
    J = np.asarray(vmap(jx)(jnp.asarray(ux), jnp.asarray(uy), jnp.asarray(vx), jnp.asarray(vy), jnp.asarray(d)))
    kj, nuj, gj, tuu_j, tuv_j, tvv_j = [J[:, i] for i in range(6)]
    def rel(a, b): return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))
    print("max rel diff vs JAX:  k %.2e  nu_t %.2e  g2 %.2e | tau_uu %.2e tau_uv %.2e tau_vv %.2e" % (
        rel(kn, kj), rel(nn_, nuj), rel(gn, gj), rel(tuu_n, tuu_j), rel(tuv_n, tuv_j), rel(tvv_n, tvv_j)))


def _verify_nl(npz_path, use_ww=False):
    """numpy<->jax parity for the non-local closure (8-feat; use_ww -> 3-output + tau_zz)."""
    import jax, jax.numpy as jnp
    import model as M
    from jax import vmap, random
    import common
    cfg = config.base_cfg(use_keq=True, use_ww=use_ww)
    CL = M.make_closure_arch(cfg)
    clp = common.load_params(npz_path,
                             {"closure": CL.init(random.PRNGKey(0), jnp.ones(8))})["closure"]
    W = load_closure(npz_path)
    rng = np.random.default_rng(0)
    ux, uy, vx, vy, d, k = [rng.normal(0, 2, 5000) for _ in range(6)]
    k = np.abs(k)                                          # k >= 0
    kn, nn_, gn, g0n, elln = coeffs_nl(W, ux, uy, vx, vy, d, k, cfg)
    if use_ww:
        tuu_n, tuv_n, tvv_n, tzz_n = build_tau_ww(kn, nn_, gn, g0n, ux, uy, vx, vy)
    else:
        tuu_n, tuv_n, tvv_n = build_tau(kn, nn_, gn, ux, uy, vx, vy); tzz_n = np.zeros_like(kn)

    def jx(uxi, uyi, vxi, vyi, di, ki):
        kk, nu, g, g0, _ = M.closure_fwd(CL, clp, uxi, uyi, vxi, vyi, di, cfg, k=ki)
        if use_ww:
            t = M.build_tau_ww(kk, nu, g, g0, uxi, uyi, vxi, vyi)
        else:
            t = (*M.build_tau(kk, nu, g, uxi, uyi, vxi, vyi), 0.0)
        return jnp.array([nu, g, g0, *t])
    J = np.asarray(vmap(jx)(*(jnp.asarray(a) for a in (ux, uy, vx, vy, d, k))))
    nuj, gj, g0j, tuu_j, tuv_j, tvv_j, tzz_j = [J[:, i] for i in range(7)]
    def rel(a, b): return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))
    print("NL%s max rel diff vs JAX:  nu_t %.2e g2 %.2e g0 %.2e | tau_uu %.2e uv %.2e vv %.2e zz %.2e" % (
        "(ww)" if use_ww else "", rel(nn_, nuj), rel(gn, gj), rel(g0n, g0j),
        rel(tuu_n, tuu_j), rel(tuv_n, tuv_j), rel(tvv_n, tvv_j), rel(tzz_n, tzz_j)))


def _verify_force(npz_path):
    """numpy<->jax parity for the forcing-vector closure (use_force; outputs Fx,Fy,nu_t,ell)."""
    import jax, jax.numpy as jnp
    import model as M
    from jax import vmap, random
    import common
    cfg = config.base_cfg(use_keq=True, nut_kl=True, ell_nn=True, use_force=True)
    CL = M.make_closure_arch(cfg)
    clp = common.load_params(npz_path,
                             {"closure": CL.init(random.PRNGKey(0), jnp.ones(8))})["closure"]
    W = load_closure(npz_path)
    rng = np.random.default_rng(0)
    ux, uy, vx, vy, d, k = [rng.normal(0, 2, 5000) for _ in range(6)]
    k = np.abs(k)
    Fxn, Fyn, nun, elln = coeffs_force(W, ux, uy, vx, vy, d, k, cfg)

    def jx(uxi, uyi, vxi, vyi, di, ki):
        kk, nu, Fx, Fy, ell = M.closure_fwd(CL, clp, uxi, uyi, vxi, vyi, di, cfg, k=ki)
        return jnp.array([Fx, Fy, nu, ell])
    J = np.asarray(vmap(jx)(*(jnp.asarray(a) for a in (ux, uy, vx, vy, d, k))))
    Fxj, Fyj, nuj, ellj = [J[:, i] for i in range(4)]
    def rel(a, b): return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))
    print("FORCE max rel diff vs JAX:  Fx %.2e  Fy %.2e  nu_t %.2e  ell %.2e" % (
        rel(Fxn, Fxj), rel(Fyn, Fyj), rel(nun, nuj), rel(elln, ellj)))


if __name__ == "__main__":
    if "--verify" in sys.argv:
        _verify(os.path.join(config.CKPT_DIR, "tb_tri_params.npz"))
    if "--verify_force" in sys.argv:
        i = sys.argv.index("--verify_force")
        tag = sys.argv[i + 1] if i + 1 < len(sys.argv) else "fc_is_cir"
        _verify_force(os.path.join(config.CKPT_DIR, f"{tag}_params.npz"))
    if "--verify_nl" in sys.argv:
        i = sys.argv.index("--verify_nl")
        tag = sys.argv[i + 1] if i + 1 < len(sys.argv) else "nl_cir"
        _verify_nl(os.path.join(config.CKPT_DIR, f"{tag}_params.npz"))
    if "--verify_ww" in sys.argv:
        i = sys.argv.index("--verify_ww")
        tag = sys.argv[i + 1] if i + 1 < len(sys.argv) else "nl3_cir"
        _verify_nl(os.path.join(config.CKPT_DIR, f"{tag}_params.npz"), use_ww=True)
