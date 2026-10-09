"""Steady RANS physics + losses for the closure model.

Mean momentum (normalized, nu=NU, tau_ij = <u'_i u'_j> with positive diagonal):
  continuity : u_x + v_y = 0
  x-mom : u*u_x + v*u_y + p_x - nu*(u_xx+u_yy) + d/dx(tau_uu) + d/dy(tau_uv) = 0
  y-mom : u*v_x + v*v_y + p_y - nu*(v_xx+v_yy) + d/dx(tau_uv) + d/dy(tau_vv) = 0

tau(x,y) = C(grad u(x,y), sdf(x,y)); the Reynolds-stress divergence is obtained
by autodiff (jacrev) through that composition.  The SAME tau (and sign) is used
in the residual and in the stress-consistency data loss.

These functions take the flax modules MF, CL, the cfg, and the (python) per-shape
`sdf` as leading STATIC args (closed over / specialized at trace time), and the
traced params/coords as the rest.  vmap/jit happen in train.py / eval.py.
"""
import jax
import jax.numpy as jnp
from jax import grad, jacrev, vmap
from jax.tree_util import tree_leaves

import config
import model as M

NU = config.NU

# loss keys
PDE_KEYS = ["ru", "rv", "rc"]
DATA_KEYS = ["data_u", "data_v", "tau_uu", "tau_uv", "tau_vv"]
BC_KEYS = ["u_in", "v_in", "p_out", "sym_v", "sym_uy", "u_wall", "v_wall", "tau_free"]
TRAIN_KEYS = PDE_KEYS + DATA_KEYS + BC_KEYS         # Phase A
EVAL_KEYS = PDE_KEYS + BC_KEYS                       # Phase B (no data/stress)

# non-local (use_keq) transported-k loss keys: k-transport residual, k-data, k-BCs
K_PDE_KEYS = ["rk"]
K_DATA_KEYS = ["data_k"]
K_BC_KEYS = ["k_wall", "k_in"]
K_ALL_KEYS = K_PDE_KEYS + K_DATA_KEYS + K_BC_KEYS   # all weighted by cfg.k_weight
WW_KEYS = ["tau_zz"]                                  # out-of-plane stress (use_ww), tau-weighted
# forcing-vector closure (use_force): no stress terms; supervise the mean field + the momentum
# residual (which drives F to the DNS balance) + the transported k; anchor F~0 in the freestream.
FORCE_DATA_KEYS = ["data_u", "data_v"]
FORCE_BC_KEYS = ["u_in", "v_in", "p_out", "sym_v", "sym_uy", "u_wall", "v_wall", "F_free"]


def train_keys(cfg):
    if getattr(cfg, "use_force", False):
        return PDE_KEYS + K_PDE_KEYS + FORCE_DATA_KEYS + K_DATA_KEYS + FORCE_BC_KEYS + K_BC_KEYS
    if getattr(cfg, "use_keq", False):
        keys = PDE_KEYS + K_PDE_KEYS + DATA_KEYS + K_DATA_KEYS + BC_KEYS + K_BC_KEYS
        if getattr(cfg, "use_ww", False):
            keys = keys + WW_KEYS
        return keys
    return TRAIN_KEYS


def eval_keys(cfg):
    """Phase-B / PINN-deploy keys: PDE residuals + BCs, NO data terms (with_data=False) -- the
    data-free analogue of train_keys, so a frozen non-local/force closure is solved with the full
    coupled u-k system (momentum + continuity + k-transport + velocity/k BCs)."""
    if getattr(cfg, "use_force", False):
        return PDE_KEYS + K_PDE_KEYS + FORCE_BC_KEYS + K_BC_KEYS
    if getattr(cfg, "use_keq", False):
        return PDE_KEYS + K_PDE_KEYS + BC_KEYS + K_BC_KEYS
    return EVAL_KEYS


# ======================================================================
#  Per-point physics
# ======================================================================
def velocity_grads(MF, mf_params, x, y):
    u_x, u_y = grad(lambda xx, yy: M.mf_u(MF, mf_params, xx, yy), (0, 1))(x, y)
    v_x, v_y = grad(lambda xx, yy: M.mf_v(MF, mf_params, xx, yy), (0, 1))(x, y)
    return u_x, u_y, v_x, v_y


def _k_at(MF, cfg, mf_params, x, y):
    """Transported TKE field (use_keq) or None for the local model."""
    return M.mf_k(MF, mf_params, x, y) if getattr(cfg, "use_keq", False) else None


def tau_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale=1.0):
    """In-plane Reynolds stress (tau_uu, tau_uv, tau_vv) as a function of (x,y).
    tau_scale=0 -> laminar baseline (no closure)."""
    u_x, u_y, v_x, v_y = velocity_grads(MF, mf_params, x, y)
    d = sdf(x, y)
    kt = _k_at(MF, cfg, mf_params, x, y)             # transported k (use_keq) else None
    k, nu_t, g2, g0, _ = M.closure_fwd(CL, cl_params, u_x, u_y, v_x, v_y, d, cfg, k=kt)
    if getattr(cfg, "use_ww", False):
        tuu, tuv, tvv, _ = M.build_tau_ww(k, nu_t, g2, g0, u_x, u_y, v_x, v_y)
    else:
        tuu, tuv, tvv = M.build_tau(k, nu_t, g2, u_x, u_y, v_x, v_y)
    return tau_scale * jnp.array([tuu, tuv, tvv])


def tau_zz_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y):
    """Out-of-plane normal stress tau_zz=ww at (x,y) (use_ww)."""
    u_x, u_y, v_x, v_y = velocity_grads(MF, mf_params, x, y)
    d = sdf(x, y)
    kt = _k_at(MF, cfg, mf_params, x, y)
    k, nu_t, g2, g0, _ = M.closure_fwd(CL, cl_params, u_x, u_y, v_x, v_y, d, cfg, k=kt)
    return M.build_tau_ww(k, nu_t, g2, g0, u_x, u_y, v_x, v_y)[3]


def force_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y):
    """Predicted RESIDUAL Reynolds force R=(Rx,Ry) at (x,y) (hybrid use_force):
    closure slots 2,3 carry the residual force instead of (g2,G0). The full Reynolds
    force is F = -div(tau_e) + R with tau_e=-2 nu_t S the Boussinesq eddy stress, so the
    NN only models the NON-Boussinesq remainder while nu_t carries the (stabilizing)
    turbulent diffusion. R -> 0 in the freestream (nu_t -> 0 there too)."""
    u_x, u_y, v_x, v_y = velocity_grads(MF, mf_params, x, y)
    d = sdf(x, y)
    kt = _k_at(MF, cfg, mf_params, x, y)
    _, _, Rx, Ry, _ = M.closure_fwd(CL, cl_params, u_x, u_y, v_x, v_y, d, cfg, k=kt)
    if getattr(cfg, "force_proj", False):
        # energy-neutral projection: remove the component of R along the mean velocity so R.U=0
        # (does no work on the mean flow). nu_t>=0 then carries all the dissipation.
        u, v, _ = M.mf_fwd(MF, mf_params, x, y)
        eps = getattr(cfg, "force_proj_eps", 1e-3)
        w = (Rx * u + Ry * v) / (u * u + v * v + eps)
        Rx, Ry = Rx - w * u, Ry - w * v
    return Rx, Ry


def tau_eddy_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale=1.0):
    """Boussinesq eddy-viscosity stress tau_e = -2 nu_t S (in-plane) as a function of (x,y),
    for the hybrid force closure. The (2/3)k I isotropic part is a gradient absorbed into the
    pressure (and into R), so only the deviatoric -2 nu_t S is kept here."""
    u_x, u_y, v_x, v_y = velocity_grads(MF, mf_params, x, y)
    d = sdf(x, y)
    kt = _k_at(MF, cfg, mf_params, x, y)
    _, nu_t, _, _, _ = M.closure_fwd(CL, cl_params, u_x, u_y, v_x, v_y, d, cfg, k=kt)
    e_uu = -2.0 * nu_t * u_x
    e_uv = -nu_t * (u_y + v_x)
    e_vv = -2.0 * nu_t * v_y
    return tau_scale * jnp.array([e_uu, e_uv, e_vv])


def tau_eddy_div(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale=1.0):
    """(div_x, div_y) of the eddy stress tau_e via jacrev through tau_e(x,y)."""
    J = jacrev(lambda xx, yy: tau_eddy_at(MF, CL, cfg, sdf, mf_params, cl_params, xx, yy, tau_scale),
               argnums=(0, 1))(x, y)
    dtdx, dtdy = J
    div_x = dtdx[0] + dtdy[1]
    div_y = dtdx[1] + dtdy[2]
    return div_x, div_y


def force_closure_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y):
    """Total closure Reynolds force F = div(2 nu_t S) + R = -div(tau_e) + R (the quantity that
    replaces -div(tau) in the momentum balance), for direct supervision against the DNS force."""
    dvx, dvy = tau_eddy_div(MF, CL, cfg, sdf, mf_params, cl_params, x, y)
    Rx, Ry = force_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y)
    return -dvx + Rx, -dvy + Ry


def _mixing_length(cfg, d):
    """ell = min(kappa*max(d,d_floor), ell_max): floored at the wall, capped far-field."""
    return jnp.minimum(cfg.kappa * jnp.maximum(d, cfg.d_floor), cfg.ell_max)


def nut_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y):
    """Eddy viscosity nu_t at a point (uses the transported k feature; use_keq)."""
    u_x, u_y, v_x, v_y = velocity_grads(MF, mf_params, x, y)
    d = sdf(x, y)
    kt = _k_at(MF, cfg, mf_params, x, y)
    _, nu_t, _, _, _ = M.closure_fwd(CL, cl_params, u_x, u_y, v_x, v_y, d, cfg, k=kt)
    return nu_t


def k_residual_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y):
    """Steady transported-k residual (use_keq):
        rk = u*k_x + v*k_y - P + k^{3/2}/ell - div(nu_t grad k)
    P = 4 nu_t (a^2+b^2) = -tau:grad u; returns (rk, k)."""
    u, v, _ = M.mf_fwd(MF, mf_params, x, y)
    k = M.mf_k(MF, mf_params, x, y)
    k_x, k_y = grad(lambda xx, yy: M.mf_k(MF, mf_params, xx, yy), (0, 1))(x, y)
    u_x, u_y, v_x, v_y = velocity_grads(MF, mf_params, x, y)
    d = sdf(x, y)
    _, nu_t, _, _, ell = M.closure_fwd(CL, cl_params, u_x, u_y, v_x, v_y, d, cfg, k=k)
    a = 0.5 * (u_x - v_y)
    b = 0.5 * (u_y + v_x)
    prod = 4.0 * nu_t * (a ** 2 + b ** 2)            # production P = -tau:grad u
    cd = cfg.cmu ** 0.75 if getattr(cfg, "nut_kl", False) else 1.0   # calibrated dissipation const
    dest = cd * k ** 1.5 / ell                       # destruction cmu^{3/4} k^{3/2}/ell

    def k_flux(xx, yy):                              # F = nu_t * grad k
        kx, ky = grad(lambda p, q: M.mf_k(MF, mf_params, p, q), (0, 1))(xx, yy)
        nt = nut_at(MF, CL, cfg, sdf, mf_params, cl_params, xx, yy)
        return nt * jnp.array([kx, ky])
    Jf = jacrev(k_flux, argnums=(0, 1))(x, y)
    dFx, dFy = Jf
    div_diff = dFx[0] + dFy[1]                        # div(nu_t grad k)
    rk = u * k_x + v * k_y - prod + dest - div_diff
    return rk, k


def tau_div(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale=1.0):
    """(d/dx, d/dy) of tau via jacrev through tau(x,y); returns (div_x, div_y)."""
    J = jacrev(lambda xx, yy: tau_at(MF, CL, cfg, sdf, mf_params, cl_params, xx, yy, tau_scale),
               argnums=(0, 1))(x, y)
    dtdx, dtdy = J                      # each shape (3,)
    div_x = dtdx[0] + dtdy[1]           # d/dx tau_uu + d/dy tau_uv
    div_y = dtdx[1] + dtdy[2]           # d/dx tau_uv + d/dy tau_vv
    return div_x, div_y


def residual_and_state(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale=1.0):
    """(ru, rv, rc, u, v, p, rk, k) at a scalar point. rk,k are 0 for the local
    model; with use_keq they carry the transported-k residual and field."""
    u, v, p = M.mf_fwd(MF, mf_params, x, y)
    u_x, u_y, v_x, v_y = velocity_grads(MF, mf_params, x, y)
    p_x, p_y = grad(lambda xx, yy: M.mf_p(MF, mf_params, xx, yy), (0, 1))(x, y)
    u_hess = jax.hessian(lambda xx, yy: M.mf_u(MF, mf_params, xx, yy), (0, 1))(x, y)
    v_hess = jax.hessian(lambda xx, yy: M.mf_v(MF, mf_params, xx, yy), (0, 1))(x, y)
    u_xx, u_yy = u_hess[0][0], u_hess[1][1]
    v_xx, v_yy = v_hess[0][0], v_hess[1][1]
    if getattr(cfg, "use_force", False):
        # hybrid force closure: the momentum diffusion is carried by the Boussinesq eddy
        # stress tau_e=-2 nu_t S (kept implicit -> well-posed deployment), and the NN predicts
        # only the residual force R. Balance: conv + grad p - visc + div(tau_e) - R = 0, which
        # equals conv + grad p - visc - F = 0 with the full force F = -div(tau_e) + R.
        Rx, Ry = force_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y)
        div_x, div_y = tau_eddy_div(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale)
        ru = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) + div_x - tau_scale * Rx
        rv = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) + div_y - tau_scale * Ry
    else:
        div_x, div_y = tau_div(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale)
        ru = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) + div_x
        rv = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) + div_y
    rc = u_x + v_y
    if getattr(cfg, "use_keq", False):
        rk, k = k_residual_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y)
    else:
        rk, k = 0.0, 0.0
    return ru, rv, rc, u, v, p, rk, k


# ======================================================================
#  Loss terms for one shape
# ======================================================================
def loss_terms(MF, CL, cfg, sdf, cl_params, mf_params, batch, data,
               sol_prev, theta, with_data=True, tau_scale=1.0):
    """Dict of scalar (unweighted) loss terms for one shape.

    batch: dict of sampled arrays (coll, wall, in_y, out_y, sym_x).
    data:  dict with xy (Nd,2), uv (Nd,2), tau (Nd,3) -- gathered PIV points.
    sol_prev: (Ncoll,3) frozen previous (u,v,p) at coll pts (pseudo-time level).
    theta: (3,) pseudo-time weights.
    """
    coll = batch["coll"]
    cx, cy = coll[:, 0], coll[:, 1]

    def res_pt(x, y):
        return residual_and_state(MF, CL, cfg, sdf, mf_params, cl_params, x, y, tau_scale)
    ru, rv, rc, u, v, p, rk, kfield = vmap(res_pt)(cx, cy)

    # pseudo-time augmentation (steady relaxation toward previous iterate)
    au = ru + theta[0] * (u - sol_prev[:, 0])
    av = rv + theta[1] * (v - sol_prev[:, 1])
    ac = rc + theta[2] * (p - sol_prev[:, 2])
    d = {"ru": jnp.mean(au ** 2), "rv": jnp.mean(av ** 2), "rc": jnp.mean(ac ** 2)}
    if getattr(cfg, "use_keq", False):
        d["rk"] = jnp.mean(rk ** 2)              # transported-k collocation residual

    fwd = lambda x, y: M.mf_fwd(MF, mf_params, x, y)

    # inlet x=X0: u=1, v=0
    iy = batch["in_y"]
    ix = jnp.full_like(iy, config.X0)
    ui, vi, _ = vmap(fwd)(ix, iy)
    d["u_in"] = jnp.mean((ui - 1.0) ** 2)
    d["v_in"] = jnp.mean(vi ** 2)

    # outlet x=X1: pressure gauge p=0
    oy = batch["out_y"]
    ox = jnp.full_like(oy, config.X1)
    _, _, po = vmap(fwd)(ox, oy)
    d["p_out"] = jnp.mean(po ** 2)

    # symmetry y=+-Y1: v=0 and u_y=0
    sxs = batch["sym_x"]
    yhi = jnp.full_like(sxs, config.Y1)
    ylo = jnp.full_like(sxs, config.Y0)
    _, vhi, _ = vmap(fwd)(sxs, yhi)
    _, vlo, _ = vmap(fwd)(sxs, ylo)
    d["sym_v"] = jnp.mean(vhi ** 2) + jnp.mean(vlo ** 2)
    uy_hi = vmap(lambda x, y: grad(lambda yy: M.mf_u(MF, mf_params, x, yy))(y))(sxs, yhi)
    uy_lo = vmap(lambda x, y: grad(lambda yy: M.mf_u(MF, mf_params, x, yy))(y))(sxs, ylo)
    d["sym_uy"] = jnp.mean(uy_hi ** 2) + jnp.mean(uy_lo ** 2)

    # no-slip body wall: u=v=0
    wx, wy = batch["wall"][:, 0], batch["wall"][:, 1]
    uw, vw, _ = vmap(fwd)(wx, wy)
    d["u_wall"] = jnp.mean(uw ** 2)
    d["v_wall"] = jnp.mean(vw ** 2)

    # freestream closure anchor at the inlet (far upstream): the Reynolds force ~0 for the
    # force model, the stress ~0 otherwise.
    if getattr(cfg, "use_force", False):
        Fin = vmap(lambda x, y: jnp.array(force_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y)))(ix, iy)
        d["F_free"] = jnp.mean(Fin ** 2)
    else:
        tau_in = vmap(lambda x, y: tau_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y))(ix, iy)
        d["tau_free"] = jnp.mean(tau_in ** 2)

    # closure Lipschitz penalty: ||d(k,nu_t,g2)/d(grad u)||_F^2 at coll points.
    # Limits how fast the LEARNED coefficients swing with the velocity gradients
    # (the tensor-basis strain dependence in build_tau is intentionally NOT
    # penalized). Smaller -> smoother closure -> more contractive FEM Picard map
    # and less DG0 source noise. grad u / sdf are stop_gradient'd so the penalty
    # shapes only the closure, not the mean field. Kept out of TRAIN_KEYS.
    # Unified closure-Lipschitz penalty (lip_weight), k-aware and method-agnostic: penalizes
    # ||d(closure coefficient outputs)/d(grad u)||_F^2 at collocation points, where closure_fwd
    # returns (k, nu_t, c2, c3, ell) -- c2,c3 = (g2,G0) for the stress models, (Rx,Ry) for the
    # force model. So a single lip_weight bounds the strain-sensitivity of every closure coefficient
    # (and, for M4, the residual force R) -> smoother closure -> contractive a-posteriori Picard map
    # without constraining the coefficient MAGNITUDES. The transported k (use_keq) is passed and
    # stop_gradient'd; for use_keq it is a passthrough so d k/d(grad u)=0 (only the local model's
    # k contributes). grad u / sdf / k are stop_gradient'd so the penalty shapes only the closure.
    if cfg.lip_weight > 0.0:
        def coeffs_vec(ux, uy, vx, vy, dd, kk):
            return jnp.array(M.closure_fwd(CL, cl_params, ux, uy, vx, vy, dd, cfg, k=kk))

        def lip_pt(x, y):
            ux, uy, vx, vy = velocity_grads(MF, mf_params, x, y)
            ux, uy, vx, vy = (jax.lax.stop_gradient(g) for g in (ux, uy, vx, vy))
            dd = jax.lax.stop_gradient(sdf(x, y))
            kk = _k_at(MF, cfg, mf_params, x, y)
            kk = None if kk is None else jax.lax.stop_gradient(kk)
            J = jacrev(coeffs_vec, argnums=(0, 1, 2, 3))(ux, uy, vx, vy, dd, kk)
            return sum(jnp.sum(j ** 2) for j in J)
        d["lip"] = jnp.mean(vmap(lip_pt)(cx, cy))
    else:
        d["lip"] = 0.0

    if with_data:
        dx, dy = data["xy"][:, 0], data["xy"][:, 1]
        ud, vd, _ = vmap(fwd)(dx, dy)
        d["data_u"] = jnp.mean((ud - data["uv"][:, 0]) ** 2)
        d["data_v"] = jnp.mean((vd - data["uv"][:, 1]) ** 2)
        if not getattr(cfg, "use_force", False):     # force model has no modelled stress to match
            tau_d = vmap(lambda x, y: tau_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y))(dx, dy)
            d["tau_uu"] = jnp.mean((tau_d[:, 0] - data["tau"][:, 0]) ** 2)
            d["tau_uv"] = jnp.mean((tau_d[:, 1] - data["tau"][:, 1]) ** 2)
            d["tau_vv"] = jnp.mean((tau_d[:, 2] - data["tau"][:, 2]) ** 2)
        # direct DNS-force supervision: match the total closure force F=div(2 nu_t S)+R to the
        # filtered measured force F_DNS=-div<u'u'> at the data points (force_sup).
        if getattr(cfg, "use_force", False) and getattr(cfg, "force_sup", 0.0) > 0.0 and "F" in data:
            Fc = vmap(lambda x, y: jnp.array(
                force_closure_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y)))(dx, dy)
            d["data_F"] = jnp.mean((Fc - data["F"]) ** 2)
        else:
            d["data_F"] = 0.0

    # transported-k supervision + BCs (use_keq). k(x,y)=mf_k; target 1/2(uu+vv).
    if getattr(cfg, "use_keq", False):
        mfk = lambda x, y: M.mf_k(MF, mf_params, x, y)
        d["k_wall"] = jnp.mean(vmap(mfk)(wx, wy) ** 2)         # k=0 on the body
        d["k_in"] = jnp.mean(vmap(mfk)(ix, iy) ** 2)           # k~0 at the inlet
        if with_data and "k" in data:
            kd = vmap(mfk)(data["xy"][:, 0], data["xy"][:, 1])
            d["data_k"] = jnp.mean((kd - data["k"]) ** 2)
        else:
            d["data_k"] = 0.0
        # tau_zz supervision is a STRESS-model term (build_tau_ww); the force model has no
        # modelled stress, so for use_force, use_ww only promotes the k target to the full 3D
        # TKE 1/2(uu+vv+ww) (set in sample_data) and skips tau_zz entirely.
        if getattr(cfg, "use_ww", False) and not getattr(cfg, "use_force", False):
            if with_data and "ww" in data:
                zz = vmap(lambda x, y: tau_zz_at(MF, CL, cfg, sdf, mf_params, cl_params, x, y))(
                    data["xy"][:, 0], data["xy"][:, 1])
                d["tau_zz"] = jnp.mean((zz - data["ww"]) ** 2)
            else:
                d["tau_zz"] = 0.0
    return d


# ======================================================================
#  Pseudo-time weights (cosine shrink) -- adapted from cyl_pinn_pt.py:205-226
# ======================================================================
def residual_mse_per_comp(MF, CL, cfg, sdf, cl_params, mf_params, coll):
    def res_pt(x, y):
        ru, rv, rc, *_ = residual_and_state(MF, CL, cfg, sdf, mf_params, cl_params, x, y)
        return ru, rv, rc
    ru, rv, rc = vmap(res_pt)(coll[:, 0], coll[:, 1])
    return jnp.array([jnp.mean(ru ** 2), jnp.mean(rv ** 2), jnp.mean(rc ** 2)])


def pseudo_time_weights(MF, CL, cfg, sdf, cl_params, mf_params, prev_mf_params,
                        coll, losses0, start_log_drop=2.0, end_log_drop=6.0, min_factor=0.1):
    x, y = coll[:, 0], coll[:, 1]

    def res_state(mp, x, y):
        ru, rv, rc, u, v, p, *_ = residual_and_state(MF, CL, cfg, sdf, mp, cl_params, x, y)
        return jnp.array([ru, rv, rc]), jnp.array([u, v, p])
    R, S = vmap(lambda xx, yy: res_state(mf_params, xx, yy))(x, y)
    Rp, Sp = vmap(lambda xx, yy: res_state(prev_mf_params, xx, yy))(x, y)
    res_diff = (R - Rp).T          # (3, N)
    sol_diff = (S - Sp).T          # (3, N)
    losses = jnp.mean(R ** 2, axis=0)        # (3,)
    log_drop = jnp.log10((losses0 + 1e-8) / (losses + 1e-8))
    pgr = jnp.clip((log_drop - start_log_drop) / (end_log_drop - start_log_drop), 0.0, 1.0)
    factor = min_factor + (1.0 - min_factor) * 0.5 * (1.0 + jnp.cos(jnp.pi * pgr))
    theta = jnp.linalg.norm(res_diff, axis=1) / (jnp.linalg.norm(sol_diff, axis=1) + 1e-8) * factor
    return jnp.clip(theta, 1e-2, 100.0)
