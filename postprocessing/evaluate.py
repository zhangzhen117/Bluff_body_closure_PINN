"""Six-way evaluation (DNS, M1 local, M2 algebraic ell, M3 learned ell, M4 force, SST).

Computes the relative L2 error against DNS of U,V,p,k,tau_xx,tau_xy,tau_yy,tau_zz,F_x,F_y through
one pipeline for all models, plus the control-volume drag Cd. FEM fields share the (200,360)
common grid; SST is on its own grid. Forces are F=-div(tau) with a light Gaussian denoise; k is
the full TKE 1/2(uu+vv+ww). Pressure is compared after removing its mean over the window.

Usage: python postprocessing/evaluate.py --mode is --shapes circle square_45_degree tri_long_0_dns
Missing model fields (not yet deployed) are skipped. Writes results/errors_<mode>.csv.
"""
import os, sys, csv
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import jax.numpy as jnp
from jax import vmap
import argparse, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pinn")); sys.path.insert(0, HERE)
import config, geometry as geo, data as dataio
from forces import reynolds_force_from_stress
from compute_cd_force import cd_force

NU = config.NU; FSMOOTH = 1.2
SHAPES = [("circle", "cir"), ("square_0_degree", "sq0"), ("ellipse_0_degree", "ell"),
          ("tri_long_0_dns", "trl"), ("tri_equi_0_dns", "tre"), ("square_45_degree", "dia")]
CODES = dict(SHAPES)
METHODS = ["loc", "alg", "nn", "fc", "sst"]          # M1, M2, M3, M4, SST
QTYS = ["U", "V", "p", "k", "tau_uu", "tau_uv", "tau_vv", "tau_zz", "Fx", "Fy"]


def eddy_tau(XX, YY, U, V, nut, k=None):
    xs = XX[0, :]; ys = YY[:, 0]
    Ux = np.gradient(np.nan_to_num(U), xs, axis=1); Uy = np.gradient(np.nan_to_num(U), ys, axis=0)
    Vx = np.gradient(np.nan_to_num(V), xs, axis=1); Vy = np.gradient(np.nan_to_num(V), ys, axis=0)
    kk = 0.0 if k is None else (2.0 / 3.0) * np.nan_to_num(k)
    return kk - 2 * nut * Ux, -2 * nut * 0.5 * (Uy + Vx), kk - 2 * nut * Vy


def field_path(method, shape, mode):
    """Deployed field of a model (results/fem/<shape>_<mode>_<method>_<code>*.npz) or the SST field."""
    if method == "sst":
        p = os.path.join(config.SST_DIR, f"{shape}.npz")
        return p if os.path.exists(p) else None
    pat = os.path.join(config.RESULTS_DIR, "fem", f"{shape}_{mode}_{method}_{CODES[shape]}*.npz")
    hits = [p for p in sorted(glob.glob(pat)) if not p.endswith(("_dofs.npz", "_diag.npz"))]
    return hits[0] if hits else None


def load(method, shape, code, mode):
    p = field_path(method, shape, mode)
    if p is None:
        raise FileNotFoundError(f"no field for {method} / {shape} / {mode}")
    return np.load(p)


def method_fields(method, z):
    """Physical fields on the method's own grid; missing quantities -> None."""
    XX, YY = z["X"], z["Y"]; f = {q: None for q in QTYS}
    f["U"], f["V"], f["p"] = z["U"], z["V"], z["p"]
    if method in ("loc", "alg", "nn"):
        uu, uv, vv = z["uu"], z["uv"], z["vv"]
        f["tau_uu"], f["tau_uv"], f["tau_vv"] = uu, uv, vv
        f["Fx"], f["Fy"] = reynolds_force_from_stress(XX, YY, uu, uv, vv)
        f["k"] = z["k"] if "k" in z.files else 0.5 * (uu + vv)
        if "ww" in z.files:
            f["tau_zz"] = z["ww"]
    elif method == "fc":
        e = eddy_tau(XX, YY, z["U"], z["V"], z["nut"]); Mex, Mey = reynolds_force_from_stress(XX, YY, *e)
        f["Fx"], f["Fy"] = Mex + z["Fx"], Mey + z["Fy"]; f["k"] = z["k"]
    elif method == "sst":
        e = eddy_tau(XX, YY, z["U"], z["V"], z["nut"], k=z["k"])
        f["tau_uu"], f["tau_uv"], f["tau_vv"] = e
        f["Fx"], f["Fy"] = reynolds_force_from_stress(XX, YY, *e)
        f["k"] = z["k"]; f["tau_zz"] = (2.0 / 3.0) * z["k"]
    return XX, YY, f


def dns_on(XX, YY, d):
    pts = np.stack([np.asarray(d.x), np.asarray(d.y)], 1)
    G = lambda v: griddata(pts, np.asarray(v), (XX, YY), "linear")
    uu, uv, vv, ww = G(d.uu), G(d.uv), G(d.vv), G(np.asarray(d.data_ww))
    Fx, Fy = reynolds_force_from_stress(XX, YY, uu, uv, vv)
    return dict(U=G(d.U), V=G(d.V), p=G(d.P), k=0.5 * (uu + vv + ww),
                tau_uu=uu, tau_uv=uv, tau_vv=vv, tau_zz=ww, Fx=Fx, Fy=Fy)


def relL2(pred, ref, m, q):
    if pred is None or ref is None:
        return None
    P, R = np.array(pred, float), np.array(ref, float)
    if q in ("Fx", "Fy"):
        P, R = gaussian_filter(np.nan_to_num(P), FSMOOTH), gaussian_filter(np.nan_to_num(R), FSMOOTH)
    if q == "p":
        P = P - np.nanmean(P[m]); R = R - np.nanmean(R[m])
    mm = m & np.isfinite(P) & np.isfinite(R)
    return float(np.linalg.norm((P - R)[mm]) / (np.linalg.norm(R[mm]) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="is", choices=("is", "lo"), help="in-sample or leave-one-shape-out")
    ap.add_argument("--shapes", nargs="+", default=["circle", "square_45_degree", "tri_long_0_dns"])
    a = ap.parse_args()
    rows = []
    for shape in a.shapes:
        d = dataio.load_dns(config.dns_path(shape), name=shape)
        sdf = geo.make_sdf(shape)
        dns_cache = {}
        for method in METHODS:
            if field_path(method, shape, a.mode) is None:
                print(f"  skip {method} / {shape}: no field"); continue
            XX, YY, f = method_fields(method, load(method, shape, CODES[shape], a.mode))
            key = (XX.shape, float(XX[0, 0]), float(YY[-1, -1]))
            if key not in dns_cache:
                dns_cache[key] = dns_on(XX, YY, d)
            D = dns_cache[key]
            body = np.asarray(vmap(sdf)(jnp.asarray(XX.ravel()), jnp.asarray(YY.ravel()))).reshape(XX.shape) < 0
            m = (~body) & np.isfinite(D["U"]) & np.isfinite(f["U"])
            row = dict(shape=shape, model=method)
            for q in QTYS:
                e = relL2(f[q], D[q], m, q)
                row[q] = "" if e is None else f"{e:.4f}"
            row["Cd"] = f"{cd_force(XX, YY, f['U'], f['V'], f['p'], f['Fx'], f['Fy']):.4f}"
            row["Cd_DNS"] = f"{cd_force(XX, YY, D['U'], D['V'], D['p'], D['Fx'], D['Fy']):.4f}"
            rows.append(row)
            print("  " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, f"errors_{a.mode}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["shape", "model"] + QTYS + ["Cd", "Cd_DNS"])
        w.writeheader(); w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
