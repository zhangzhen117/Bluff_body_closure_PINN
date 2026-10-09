"""Six-way field figure (DNS, M1--M4, SST): rows U,V,p,k,tau_xx,tau_xy,tau_yy,tau_zz,F_x,F_y.

The number above each panel is the relative L2 error of that field against DNS. Blank panels are
quantities a model does not represent. Columns without a deployed field are omitted.

Usage: python postprocessing/plot_sixway.py --shape circle --mode is
Writes results/figures/sixway_<mode>_<shape>.png.
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pinn")); sys.path.insert(0, HERE)
import config, geometry as geo, data as dataio
from evaluate import load, field_path, method_fields, dns_on, METHODS, CODES, QTYS, FSMOOTH

# --- LaTeX label rendering. usetex=True needs a system LaTeX install; mathtext (default) renders
#     the same $...$ markup with Computer-Modern fonts and always works. Flip USETEX if you have LaTeX. ---
USETEX = False
plt.rcParams.update({'text.usetex': USETEX,
                     'font.family': 'serif' if USETEX else 'sans-serif',
                     'mathtext.fontset': 'cm'})

# method column labels (LaTeX) and quantity row labels (LaTeX) -- keys are the plain field names
LABEL = {'loc': r'M1 (local)', 'alg': r'M2 (alg-$\ell$)', 'nn': r'M3 (NN-$\ell$)',
         'fc': r'M4 (force)', 'sst': r'SST $k$-$\omega$'}
PRETTY = {'U': r'$U$', 'V': r'$V$', 'p': r'$p$', 'k': r'$k$',
          'tau_uu': r'$\tau_{xx}$', 'tau_uv': r'$\tau_{xy}$', 'tau_vv': r'$\tau_{yy}$',
          'tau_zz': r'$\tau_{zz}$', 'Fx': r'$F_x$', 'Fy': r'$F_y$'}
DIVERGING = {'V', 'p', 'tau_uv', 'Fx', 'Fy'}     # symmetric +/- colour scale
NONNEG = {'k', 'tau_uu', 'tau_vv', 'tau_zz'}      # clamp colour floor to 0
_DNS_CACHE = {}


def _body_mask(sdf, XX, YY):
    s = np.asarray(vmap(sdf)(jnp.asarray(XX.ravel()), jnp.asarray(YY.ravel()))).reshape(XX.shape)
    return s < 0

def _dns_cached(shape, XX, YY, d):
    key = (shape, XX.shape, round(float(XX[0, 0]), 6), round(float(XX[0, -1]), 6),
           round(float(YY[0, 0]), 6), round(float(YY[-1, 0]), 6))
    if key not in _DNS_CACHE:
        _DNS_CACHE[key] = dns_on(XX, YY, d)
    return _DNS_CACHE[key]

def relL2(P, R, m, q):
    if P is None or R is None:
        return None
    P = np.array(P, float); R = np.array(R, float)
    if q in ('Fx', 'Fy'):
        P = gaussian_filter(np.nan_to_num(P), FSMOOTH); R = gaussian_filter(np.nan_to_num(R), FSMOOTH)
    if q == 'p':
        P = P - np.nanmean(P[m]); R = R - np.nanmean(R[m])
    mm = m & np.isfinite(P) & np.isfinite(R)
    return float(np.linalg.norm((P - R)[mm]) / (np.linalg.norm(R[mm]) + 1e-30))

def plot_6way(shape, mode='is', qtys=None, save=None):
    """6 columns (DNS, M1, M2, M3, M4, SST) x len(qtys) rows; titles = rel-L2 vs DNS."""
    qtys = qtys or QTYS
    code = CODES[shape]
    d = dataio.load_dns(config.dns_path(shape), name=shape)
    sdf = geo.make_sdf(shape)
    XR, YR, _ = method_fields('fc', load('fc', shape, code, mode))      # reference (M4) grid
    DREF = _dns_cached(shape, XR, YR, d)
    # columns: (label, XX, YY, fields, mask, DNS-on-this-grid)
    cols = [('DNS', XR, YR, DREF, ~_body_mask(sdf, XR, YR), DREF)]
    for mth in METHODS:
        if field_path(mth, shape, mode) is None:
            print(f'  skip {mth}: no field'); continue
        XX, YY, f = method_fields(mth, load(mth, shape, code, mode))
        same = (XX.shape == XR.shape and np.allclose(XX, XR) and np.allclose(YY, YR))
        Dg = DREF if same else _dns_cached(shape, XX, YY, d)            # DNS on this column's grid
        cols.append((LABEL[mth], XX, YY, f, ~_body_mask(sdf, XX, YY), Dg))
    nrow, ncol = len(qtys), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 1.2 * nrow), squeeze=False)
    for i, q in enumerate(qtys):
        ref = np.array(DREF[q], float); ref[~cols[0][4]] = np.nan; fin = ref[np.isfinite(ref)]
        if q in DIVERGING:
            hi = np.nanpercentile(np.abs(fin), 98); lo = -hi
        else:
            lo = 0.0 if q in NONNEG else np.nanpercentile(fin, 2)
            hi = np.nanpercentile(fin, 98)
        im = None
        for j, (lab, XX, YY, f, m, Dg) in enumerate(cols):
            ax = axes[i][j]; ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            F = f.get(q)
            if F is None:
                ax.axis('off')
            else:
                S = np.array(F, float); S[~m] = np.nan
                if q == 'p':
                    S = S - np.nanmean(S[m])
                im = ax.pcolormesh(XX, YY, np.ma.masked_invalid(S), cmap='jet',
                                   vmin=lo, vmax=hi, shading='auto', rasterized=True)
                if j > 0:
                    e = relL2(F, Dg[q], m, q)
                    if e is not None:
                        ax.set_title(f'{e:.3f}', fontsize=10)
            if i == 0:
                ax.set_title((ax.get_title() + '\n' if ax.get_title() else '') + lab, fontsize=10)
            if j == 0:
                ax.set_ylabel(PRETTY[q], rotation=0, labelpad=12, va='center', fontsize=12)
        if im is not None:
            fig.colorbar(im, ax=[axes[i][k] for k in range(ncol)], fraction=0.012, pad=0.01)
    #fig.suptitle(shape.replace('_', ' ') + f'  ({mode}):  panel titles = rel-$L_2$ vs DNS', y=0.995, fontsize=11)
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight'); print('saved', save)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="circle")
    ap.add_argument("--mode", default="is", choices=("is", "lo"))
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "figures")
    os.makedirs(out, exist_ok=True)
    plot_6way(a.shape, mode=a.mode, save=os.path.join(out, f"sixway_{a.mode}_{a.shape}.png"))
