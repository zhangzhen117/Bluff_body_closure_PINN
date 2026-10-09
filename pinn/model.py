"""Networks for the closure model.

  * mean-field net  N_s(x,y) -> (u, v, p)   : ModifiedMlp, one per shape.
  * shared closure  C(features) -> (tau_uu, tau_uv, tau_vv) : ModifiedMlp.

Closure input features = [u_x, u_y, v_x, v_y, d]  (velocity gradients + signed
wall distance / SDF), all normalized; NO raw (x,y) so the closure is
geometry-agnostic and transferable across shapes.

Forward helpers take the flax module + params explicitly (the module is a static
pytree, fine to close over inside jit). Coordinate normalization maps the
near-body / wake region to ~[-1,1] while the large far-field domain stays O(1-3).
"""
import jax
import jax.numpy as jnp
import flax.linen as fnn
from archs import ModifiedMlp, Mlp


class ClosureEllNet(fnn.Module):
    """Composite closure: a main ModifiedMlp on all features + a small MLP for the
    mixing-length correction fed only the first 7 (kinematic) features -- NO k. The
    ell output is concatenated last, so closure_fwd reads it like learn_ell's out[i]."""
    main_layers: int
    main_hidden: int
    main_out: int
    ell_layers: int
    ell_hidden: int

    @fnn.compact
    def __call__(self, feats):
        main = ModifiedMlp(num_layers=self.main_layers, hidden_dim=self.main_hidden,
                           out_dim=self.main_out, activation="tanh", periodicity=None,
                           fourier_emb=None)(feats)
        e = feats[:7]                                   # kinematic features only (drop k)
        for _ in range(self.ell_layers):
            e = fnn.tanh(fnn.Dense(self.ell_hidden)(e))
        e = fnn.Dense(1)(e)
        return jnp.concatenate([main, e])

# coordinate normalization: keep the wake (x in [-1,6], |y|<2.5) near [-1,1]
# while the unconfined domain x in [-10,15], y in [-10,10] stays O(2-3).
XC, XS = 2.0, 6.0
YC, YS = 0.0, 3.0


def make_meanfield_arch(cfg):
    fourier = {"embed_scale": cfg.mf_embed_scale, "embed_dim": cfg.mf_embed_dim}
    # use_keq: 4th output is the transported TKE field k (non-local closure)
    out_dim = 4 if getattr(cfg, "use_keq", False) else 3
    # ablation: plain MLP instead of the default ModifiedMlp (analysis only; closure stays ModifiedMlp)
    Net = Mlp if getattr(cfg, "arch", "modified") == "mlp" else ModifiedMlp
    return Net(num_layers=cfg.mf_layers, hidden_dim=cfg.mf_hidden,
               out_dim=out_dim, activation="tanh", periodicity=None,
               fourier_emb=fourier)


def make_closure_arch(cfg):
    # use_force: (m, Fx, Fy); use_ww: (nu_t, g2, G0); use_keq: (nu_t, g2); local: (k, nu_t, g2);
    # learn_ell appends one more output (the mixing-length log-correction).
    if getattr(cfg, "use_force", False):
        base = 3
    elif getattr(cfg, "use_ww", False):
        base = 3
    elif getattr(cfg, "use_keq", False):
        base = 2
    else:
        base = 3
    # ell_nn: ell from a separate k-free subnet (composite); the concatenated output is base+1.
    if getattr(cfg, "ell_nn", False) and getattr(cfg, "use_keq", False):
        return ClosureEllNet(main_layers=cfg.cl_layers, main_hidden=cfg.cl_hidden,
                             main_out=base, ell_layers=cfg.ell_layers, ell_hidden=cfg.ell_hidden)
    out_dim = base
    if getattr(cfg, "learn_ell", False) and getattr(cfg, "use_keq", False):
        out_dim += 1
    return ModifiedMlp(num_layers=cfg.cl_layers, hidden_dim=cfg.cl_hidden,
                       out_dim=out_dim, activation="tanh", periodicity=None,
                       fourier_emb=None)


# ---- mean-field forward (u, v, p) and scalar component accessors ----
def mf_fwd(MF, params, x, y):
    xh = (x - XC) / XS
    yh = (y - YC) / YS
    out = MF.apply(params, jnp.stack([xh, yh]))
    return out[0], out[1], out[2]


def mf_u(MF, params, x, y):
    return mf_fwd(MF, params, x, y)[0]


def mf_v(MF, params, x, y):
    return mf_fwd(MF, params, x, y)[1]


def mf_p(MF, params, x, y):
    return mf_fwd(MF, params, x, y)[2]


def mf_k(MF, params, x, y):
    """Transported TKE field (4th mean-field output, use_keq); softplus -> k >= 0."""
    xh = (x - XC) / XS
    yh = (y - YC) / YS
    out = MF.apply(params, jnp.stack([xh, yh]))
    return jax.nn.softplus(out[3])


# ---- closure forward: features -> coefficients (k, nu_t, g2) -> tensor-basis tau ----
# inputs: raw gradients + 2 invariants + sdf (7 features), rational-limited
N_CLOSURE_FEATURES = 7


def n_closure_features(cfg):
    """Closure input width: 8 with the transported-k feature (use_keq), else 7."""
    return 8 if getattr(cfg, "use_keq", False) else N_CLOSURE_FEATURES


def _lim(f, ref):
    """rational/softsign limiter f/(|f|+ref): ~linear for |f|<<ref, ->+-1 for >>ref."""
    return f / (jnp.abs(f) + ref)


def closure_features(u_x, u_y, v_x, v_y, d, ref_grad, ref_inv, ref_sdf,
                     k=None, ref_k=0.1):
    # Same 7 features, but bounded with f/(|f|+ref) instead of tanh -> no extreme
    # closure response on out-of-distribution gradients. |S|,|Omega| are the
    # rotation invariants (nonlinear info the first linear layer cannot form).
    s_mag = jnp.sqrt(u_x ** 2 + v_y ** 2 + 0.5 * (u_y + v_x) ** 2 + 1e-12)
    w_mag = jnp.sqrt(0.5 * (u_y - v_x) ** 2 + 1e-12)
    feats = [_lim(u_x, ref_grad), _lim(u_y, ref_grad),
             _lim(v_x, ref_grad), _lim(v_y, ref_grad),
             _lim(s_mag, ref_inv), _lim(w_mag, ref_inv),
             _lim(d, ref_sdf)]
    if k is not None:                               # 8th feature: transported TKE (use_keq)
        feats.append(_lim(k, ref_k))
    return jnp.stack(feats)


def mixing_length(cfg, d, ell_out=None):
    """Destruction mixing length for the k-equation. Algebraic baseline
    ell0 = min(kappa*max(d,d_floor), ell_max). With learn_ell, ell_out is the
    closure's raw output and ell = clip(kappa*max(d,d_floor)*exp(span*tanh(ell_out)),
    d_floor, ell_cap) -- a bounded log-correction that keeps the near-wall scaling
    but lets the wake length scale grow."""
    base = cfg.kappa * jnp.maximum(d, cfg.d_floor)
    if ell_out is None:
        return jnp.minimum(base, cfg.ell_max)
    corr = jnp.exp(cfg.ell_span * jnp.tanh(ell_out))
    return jnp.clip(base * corr, cfg.d_floor, cfg.ell_cap)


def closure_fwd(CL, cparams, u_x, u_y, v_x, v_y, d, cfg, k=None):
    """Return the tensor-basis coefficients (k, nu_t, g2, g0). g0 is the coefficient
    of the out-of-plane redistribution tensor T0=diag(1/6,1/6,-1/3) and is 0 unless
    use_ww. Optional deployment-consistent processing (bounded nu_t, wall-damping,
    turbulence gate) keeps the trained operator identical to the FEM solver's.

    use_keq: pass the transported field `k`; the closure NN takes it as an extra
    feature. use_keq -> outputs (nu_t, g2); use_ww -> outputs (nu_t, g2, G0)."""
    if getattr(cfg, "use_keq", False):
        feats = closure_features(u_x, u_y, v_x, v_y, d, cfg.ref_grad, cfg.ref_inv,
                                 cfg.ref_sdf, k=k, ref_k=cfg.ref_k)
        out = CL.apply(cparams, feats)
        # slots 2,3 carry (g2, G0) for the stress models, or (Fx, Fy) for the force model (use_force).
        if getattr(cfg, "use_force", False):
            fs = getattr(cfg, "force_scale", 1.0)
            c2, c3 = fs * out[1], fs * out[2]            # Reynolds-force vector (Fx, Fy)
            i = 3
        else:
            c2 = cfg.g2_scale * jnp.tanh(out[1])         # g2
            i = 2
            if getattr(cfg, "use_ww", False):            # G0 = 2k*tanh(.) (realizable, range +-2k)
                c3 = 2.0 * k * jnp.tanh(out[i]); i += 1
            else:
                c3 = 0.0
        learn_l = getattr(cfg, "learn_ell", False) or getattr(cfg, "ell_nn", False)
        ell = mixing_length(cfg, d, out[i] if learn_l else None)
        # eddy viscosity: Prandtl-Kolmogorov nu_t = softplus(NN)*sqrt(k+eps)*ell (nut_kl),
        # else a free (optionally bounded) softplus/sigmoid of the NN output.
        nut_max = getattr(cfg, "nut_max", 0.0)
        if getattr(cfg, "nut_kl", False):
            # nu_t = cmu^{1/4} * exp(span*tanh(NN)) * sqrt(k+eps) * ell
            #   -> NN learns a bounded correction (init ~1) around the calibrated C_mu^{1/4}.
            mult = cfg.cmu ** 0.25 * jnp.exp(cfg.cnu_span * jnp.tanh(out[0]))
            nu_t = mult * jnp.sqrt(k + cfg.k_eps) * ell
        elif nut_max > 0:
            nu_t = nut_max * jax.nn.sigmoid(out[0])
        else:
            nu_t = jax.nn.softplus(out[0])
        if getattr(cfg, "wall_d0", 0.0) > 0:
            phi = jnp.tanh(jnp.maximum(d, 0.0) / cfg.wall_d0)
            nu_t, c2 = phi * nu_t, phi * c2
        if getattr(cfg, "gate_w0", 0.0) > 0:
            enst = 0.5 * (u_y - v_x) ** 2
            gate = jnp.tanh(enst / cfg.gate_w0 ** 2)
            nu_t, c2 = gate * nu_t, gate * c2
        return k, nu_t, c2, c3, ell
    feats = closure_features(u_x, u_y, v_x, v_y, d, cfg.ref_grad, cfg.ref_inv, cfg.ref_sdf)
    out = CL.apply(cparams, feats)
    k = jax.nn.softplus(out[0])                 # in-plane TKE >= 0 (sets trace)
    nut_max = getattr(cfg, "nut_max", 0.0)
    nu_t = nut_max * jax.nn.sigmoid(out[1]) if nut_max > 0 else jax.nn.softplus(out[1])
    g2 = cfg.g2_scale * jnp.tanh(out[2])        # energy-neutral residual coeff
    # wall-damping (turbulence vanishes at the wall; PINN least reliable there)
    if getattr(cfg, "wall_d0", 0.0) > 0:
        phi = jnp.tanh(jnp.maximum(d, 0.0) / cfg.wall_d0)
        k, nu_t, g2 = phi * k, phi * nu_t, phi * g2
    # turbulence gate (attractive near zero): no stress in irrotational regions, so
    # weak off-body velocity ripples cannot be turned into spurious stress. Gates
    # all three since realizability ties the deviatoric part to k anyway.
    if getattr(cfg, "gate_w0", 0.0) > 0:
        enst = 0.5 * (u_y - v_x) ** 2                # |Omega|^2 (smooth, no sqrt -> Lipschitz-safe)
        gate = jnp.tanh(enst / cfg.gate_w0 ** 2)
        k, nu_t, g2 = gate * k, gate * nu_t, gate * g2
    return k, nu_t, g2, 0.0, 0.0                     # g0=0, ell unused: local model has no k-equation


def build_tau(k, nu_t, g2, u_x, u_y, v_x, v_y):
    """Realizable 2D tensor-basis Reynolds stress:
        tau = k*I - 2*nu_t*S + g2*(S*Omega - Omega*S)
    S=[[a,b],[b,-a]] (deviatoric strain), Omega=[[0,w],[-w,0]];
    T2 = S*Omega-Omega*S = [[-2bw, 2aw],[2aw, 2bw]] is traceless symmetric with
    T2:S = 0 (energy-neutral). PSD-capped so eigenvalues k +/- |B| >= 0."""
    a = 0.5 * (u_x - v_y)
    b = 0.5 * (u_y + v_x)
    w = 0.5 * (u_y - v_x)
    # deviatoric anisotropy B = -2 nu_t S + g2 T2  (B = [[B11,B12],[B12,-B11]])
    B11 = -2.0 * nu_t * a + g2 * (-2.0 * b * w)
    B12 = -2.0 * nu_t * b + g2 * (2.0 * a * w)
    # realizability: deviatoric eigenvalue lam = sqrt(B11^2+B12^2) must be <= k
    lam = jnp.sqrt(B11 ** 2 + B12 ** 2 + 1e-30)
    scale = k * jnp.tanh(lam / (k + 1e-9)) / (lam + 1e-30)
    B11, B12 = B11 * scale, B12 * scale
    return k + B11, B12, k - B11                # tau_uu, tau_uv, tau_vv


def build_tau_ww(k, nu_t, g2, g0, u_x, u_y, v_x, v_y):
    """Realizable 3D-aware stress for 2D mean flow (use_ww), k=1/2(uu+vv+ww):
        tau = (2/3)k I - 2 nu_t S + g2 (S*Om-Om*S) + g0 T0,  T0=diag(1/6,1/6,-1/3)
    -> in-plane isotropic level iso=(2/3)k+g0/6, tau_zz=(2/3)k-g0/3 (out of plane).
    With g0=2k tanh(.) the state is realizable: iso>=k/3>0, tau_zz>=0; the in-plane
    deviatoric eigenvalue is capped at iso. Returns (tau_uu, tau_uv, tau_vv, tau_zz)."""
    a = 0.5 * (u_x - v_y)
    b = 0.5 * (u_y + v_x)
    w = 0.5 * (u_y - v_x)
    iso = (2.0 / 3.0) * k + g0 / 6.0                # in-plane isotropic level
    B11 = -2.0 * nu_t * a + g2 * (-2.0 * b * w)
    B12 = -2.0 * nu_t * b + g2 * (2.0 * a * w)
    lam = jnp.sqrt(B11 ** 2 + B12 ** 2 + 1e-30)     # cap deviatoric eigenvalue at iso
    scale = iso * jnp.tanh(lam / (iso + 1e-9)) / (lam + 1e-30)
    B11, B12 = B11 * scale, B12 * scale
    tau_zz = (2.0 / 3.0) * k - g0 / 3.0
    return iso + B11, B12, iso - B11, tau_zz        # tau_uu, tau_uv, tau_vv, tau_zz
