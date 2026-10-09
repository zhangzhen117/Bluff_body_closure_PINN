"""Shared utilities: checkpoint IO (nested param pytree) and the field metric."""
import numpy as np
import jax.numpy as jnp
from jax import vmap
from flax.core import unfreeze, freeze

import model as M


def save_params(path, params):
    """Flatten an arbitrarily-nested param mapping to '/'-joined keys and savez."""
    flat = {}

    def rec(d, prefix):
        for k, vv in d.items():
            kk = f"{prefix}/{k}" if prefix else str(k)
            if hasattr(vv, "items"):
                rec(vv, kk)
            else:
                flat[kk] = np.asarray(vv)
    rec(unfreeze(params), "")
    np.savez(path, **flat)


def load_params(path, template):
    z = np.load(path)
    tmpl = unfreeze(template)

    def rec(d, prefix):
        for k, vv in d.items():
            kk = f"{prefix}/{k}" if prefix else str(k)
            if hasattr(vv, "items"):
                rec(vv, kk)
            else:
                d[k] = jnp.asarray(z[kk])
    rec(tmpl, "")
    return tmpl


def _pred_uv(MF, mf_params, xy):
    u, v, _ = vmap(lambda x, y: M.mf_fwd(MF, mf_params, x, y))(xy[:, 0], xy[:, 1])
    return jnp.stack([u, v], axis=1)


def field_rel_l2(MF, mf_params, sdata, sel=None):
    """Relative-L2 of (u,v) over (a subset `sel` of) the valid data points."""
    xy, ref = sdata.data_xy, sdata.data_uv
    if sel is not None:
        xy, ref = xy[sel], ref[sel]
    pred = _pred_uv(MF, mf_params, xy)
    return float(jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref))


def near_wall_rel_l2(MF, mf_params, sdata, sdf, band=0.3):
    """Relative-L2 of (u,v) restricted to a near-wall band (analytic sdf < band)."""
    import numpy as np
    from jax import vmap as _vmap
    xy = np.asarray(sdata.data_xy)
    d = np.asarray(_vmap(sdf)(jnp.asarray(xy[:, 0]), jnp.asarray(xy[:, 1])))
    sel = d < band
    return field_rel_l2(MF, mf_params, sdata, sel=jnp.asarray(sel))
