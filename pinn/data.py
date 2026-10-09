"""Load the processed PIV mean-field shapes into device arrays.

Each .h5 holds structured (Ny,Nx) grids: x,y (normalized D=1), U,V (U_inf=1),
in-plane Reynolds stresses uu,uv,vv, and `mask` (1=valid PIV data, 0=FOV/occluded
region -- NOT the body). We expose:
  * the full structured grids (for plotting / metric over valid points), and
  * a flattened array of VALID interior data points (mask==1 and finite) used as
    the data-supervision / stress-consistency samples in Phase A.
"""
import os
from dataclasses import dataclass

import h5py
import numpy as np
import jax.numpy as jnp

import config


@dataclass
class ShapeData:
    name: str
    x: np.ndarray      # (Ny,Nx) structured PIV, or (Nd,) scattered DNS
    y: np.ndarray
    U: np.ndarray
    V: np.ndarray
    uu: np.ndarray
    uv: np.ndarray
    vv: np.ndarray
    mask: np.ndarray   # uint8 (Ny,Nx) for PIV; None for DNS (all valid)
    # flattened valid data points (mask==1 and all fields finite)
    data_xy: jnp.ndarray      # (Nd,2)
    data_uv: jnp.ndarray      # (Nd,2)  U,V
    data_tau: jnp.ndarray     # (Nd,3)  uu,uv,vv
    attrs: dict
    P: np.ndarray = None      # pressure (DNS only)
    data_p: jnp.ndarray = None
    data_k: jnp.ndarray = None  # (Nd,) in-plane TKE 1/2(uu+vv) target (use_keq)
    data_ww: jnp.ndarray = None  # (Nd,) out-of-plane normal stress ww=tau_zz target (use_ww)
    data_F: jnp.ndarray = None  # (Nd,2) filtered DNS Reynolds force -div<u'u'> target (force_sup)
    source: str = "piv"


def _dns_force_at_points(x, y, uu, uv, vv, sigma=1.2, nx=360, ny=200):
    """Filtered DNS Reynolds force F=-div<u'u'> sampled at the scattered DNS points.
    Grids the stresses, computes -div(tau) (one derivative, ->0 at the wall), Gaussian-denoises,
    and interpolates back to the data points -- the direct supervision target for the force model."""
    from scipy.interpolate import griddata, RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    xs = np.linspace(config.X0, config.X1, nx); ys = np.linspace(config.Y0, config.Y1, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([x, y], 1)
    g = lambda v: np.nan_to_num(griddata(pts, v, (XX, YY), "linear"))
    UU, UV, VV = g(uu), g(uv), g(vv)
    Fx = -(np.gradient(UU, xs, axis=1) + np.gradient(UV, ys, axis=0))
    Fy = -(np.gradient(UV, xs, axis=1) + np.gradient(VV, ys, axis=0))
    Fx = gaussian_filter(Fx, sigma); Fy = gaussian_filter(Fy, sigma)
    fxi = RegularGridInterpolator((ys, xs), Fx, bounds_error=False, fill_value=0.0)
    fyi = RegularGridInterpolator((ys, xs), Fy, bounds_error=False, fill_value=0.0)
    q = np.stack([y, x], 1)
    return np.stack([fxi(q), fyi(q)], 1).astype(np.float32)


def load_shape(name, data_dir=config.DATA_DIR):
    path = os.path.join(data_dir, f"{name}.h5")
    with h5py.File(path, "r") as h:
        d = {k: h[k][:] for k in ["x", "y", "U", "V", "uu", "uv", "vv", "mask"]}
        attrs = dict(h.attrs)
    valid = (d["mask"] == 1)
    for k in ["U", "V", "uu", "uv", "vv"]:
        valid &= np.isfinite(d[k])
    idx = np.where(valid)
    data_xy = np.stack([d["x"][idx], d["y"][idx]], axis=1).astype(np.float32)
    data_uv = np.stack([d["U"][idx], d["V"][idx]], axis=1).astype(np.float32)
    data_tau = np.stack([d["uu"][idx], d["uv"][idx], d["vv"][idx]], axis=1).astype(np.float32)
    data_k = (0.5 * (d["uu"][idx] + d["vv"][idx])).astype(np.float32)
    return ShapeData(
        name=name,
        x=d["x"], y=d["y"], U=d["U"], V=d["V"],
        uu=d["uu"], uv=d["uv"], vv=d["vv"], mask=d["mask"],
        data_xy=jnp.asarray(data_xy),
        data_uv=jnp.asarray(data_uv),
        data_tau=jnp.asarray(data_tau),
        data_k=jnp.asarray(data_k),
        attrs=attrs,
    )


def load_shapes(names):
    return {n: load_shape(n) for n in names}


def load_dns(path, name="circle", clip_domain=True):
    """Load a scattered DNS spanwise-avg CSV (x,y,meanU,meanV,meanP,uu,uv,vv,ww,pp).

    No mask: every point is valid fluid (mesh starts at the body surface). Pressure
    is available. Optionally clip to the computational domain [X0,X1]x[Y0,Y1]."""
    cache = path + ".npy"
    if os.path.exists(cache):
        a = np.load(cache)
    else:
        a = np.loadtxt(path, delimiter=",", skiprows=1)
        np.save(cache, a)
    x, y = a[:, 0], a[:, 1]
    U, V, P = a[:, 2], a[:, 3], a[:, 4]
    uu, uv, vv = a[:, 5], a[:, 6], a[:, 7]
    ww = a[:, 8]                                  # spanwise normal stress (tau_zz)
    if clip_domain:
        keep = ((x >= config.X0) & (x <= config.X1) &
                (y >= config.Y0) & (y <= config.Y1))
        x, y, U, V, P, uu, uv, vv, ww = (v[keep] for v in (x, y, U, V, P, uu, uv, vv, ww))
    data_xy = np.stack([x, y], axis=1).astype(np.float32)
    data_uv = np.stack([U, V], axis=1).astype(np.float32)
    data_tau = np.stack([uu, uv, vv], axis=1).astype(np.float32)
    data_k = (0.5 * (uu + vv)).astype(np.float32)         # in-plane (use_keq)
    data_ww = ww.astype(np.float32)                       # tau_zz target (use_ww)
    fcache = path + ".force.npy"                           # cache the (slow) gridded force target
    if os.path.exists(fcache):
        data_F = np.load(fcache)
    else:
        data_F = _dns_force_at_points(x, y, uu, uv, vv)
        np.save(fcache, data_F)
    return ShapeData(
        name=name, x=x, y=y, U=U, V=V, uu=uu, uv=uv, vv=vv, mask=None,
        data_xy=jnp.asarray(data_xy), data_uv=jnp.asarray(data_uv),
        data_tau=jnp.asarray(data_tau), attrs=dict(Re=1e4, source=path),
        P=P, data_p=jnp.asarray(P.astype(np.float32)),
        data_k=jnp.asarray(data_k), data_ww=jnp.asarray(data_ww),
        data_F=jnp.asarray(data_F), source="dns")


if __name__ == "__main__":
    for n in config.ALL_SHAPES:
        s = load_shape(n)
        nd = s.data_xy.shape[0]
        print(f"{n:28s} grid={s.U.shape} valid_pts={nd:6d} "
              f"U[{float(s.data_uv[:,0].min()):+.2f},{float(s.data_uv[:,0].max()):+.2f}] "
              f"uu_max={float(s.data_tau[:,0].max()):.3f} Re={s.attrs.get('Re')}")
