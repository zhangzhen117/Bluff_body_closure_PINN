"""Collocation / BC / data sampling for one shape.

We precompute (once, on host) a large pool of fluid collocation points
(sdf>0), mixing uniform-over-domain points with a denser near-wall band, and a
fixed set of no-slip wall points.  Each training step gathers a random subset
(fast, fresh-ish) and samples inlet/outlet/symmetry boundary points in jax.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import config
import geometry as geo


def precompute_pools(name, cfg, seed=0):
    """Return dict of jnp arrays: coll_pool (M,2), wall (Nw,2)."""
    rng = np.random.default_rng(seed)
    sdf_v = jax.vmap(geo.make_sdf(name))

    def keep_valid(xy, smin, smax):
        d = np.asarray(sdf_v(jnp.asarray(xy[:, 0]), jnp.asarray(xy[:, 1])))
        m = (d > smin) & (d < smax)
        return xy[m]

    # uniform over the full unconfined domain (far field), sdf>0
    n_far = 400_000
    far = np.stack([rng.uniform(config.X0, config.X1, n_far),
                    rng.uniform(config.Y0, config.Y1, n_far)], axis=1)
    far = keep_valid(far, 0.0, np.inf)

    # dense near-body box, keep a near-wall band + near-body interior fluid
    n_near = 400_000
    near = np.stack([rng.uniform(-3.0, 7.0, n_near),
                     rng.uniform(-3.0, 3.0, n_near)], axis=1)
    near = keep_valid(near, 0.0, np.inf)

    coll_pool = np.concatenate([far, near], axis=0).astype(np.float32)

    wall = geo.boundary_points(name, 20_000, rng).astype(np.float32)
    return dict(coll_pool=jnp.asarray(coll_pool), wall=jnp.asarray(wall))


def sample_step(key, pools, cfg):
    """Gather one step's batch of points (jax)."""
    k_c, k_w, k_in, k_out, k_sym = random.split(key, 5)
    M = pools["coll_pool"].shape[0]
    Nw = pools["wall"].shape[0]
    ci = random.randint(k_c, (cfg.n_coll,), 0, M)
    wi = random.randint(k_w, (cfg.n_bc_wall,), 0, Nw)
    coll = pools["coll_pool"][ci]
    wall = pools["wall"][wi]
    in_y = random.uniform(k_in, (cfg.n_bc_in,), minval=config.Y0, maxval=config.Y1)
    out_y = random.uniform(k_out, (cfg.n_bc_out,), minval=config.Y0, maxval=config.Y1)
    sym_x = random.uniform(k_sym, (cfg.n_bc_sym,), minval=config.X0, maxval=config.X1)
    return dict(coll=coll, wall=wall, in_y=in_y, out_y=out_y, sym_x=sym_x)


def sample_data(key, sdata, cfg):
    """Gather n_data random valid PIV points for a shape (Phase A supervision)."""
    Nd = sdata.data_xy.shape[0]
    di = random.randint(key, (cfg.n_data,), 0, Nd)
    out = dict(xy=sdata.data_xy[di], uv=sdata.data_uv[di], tau=sdata.data_tau[di])
    if getattr(cfg, "use_keq", False) and sdata.data_k is not None:
        if getattr(cfg, "use_ww", False) and sdata.data_ww is not None:
            out["k"] = sdata.data_k[di] + 0.5 * sdata.data_ww[di]   # 1/2(uu+vv+ww) full TKE
            out["ww"] = sdata.data_ww[di]                            # tau_zz target
        else:
            out["k"] = sdata.data_k[di]           # 1/2(uu+vv) target for transported k
    if getattr(cfg, "force_sup", 0.0) > 0.0 and sdata.data_F is not None:
        out["F"] = sdata.data_F[di]               # filtered DNS Reynolds force target
    return out
