"""Control-volume drag coefficient.

The CV drag splits into a mean-flow surface flux and a Reynolds contribution. By the divergence
theorem the Reynolds part equals the volume integral of the streamwise Reynolds force,
-oint tau_xj n_j = int F_x dV, so the same mean-flow-surface + force-volume scheme is applied to
DNS, SST and all closures (F reconstructed from the stress where needed). Cd = 2 D_x (U_inf=D=1).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinn"))
import config

NU = config.NU
trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
BOX = dict(x1=-1.5, x2=6.0, y0=2.0)


def cd_force(XX, YY, U, V, p, Fx, Fy, box=BOX):
    """Cd = 2 D_x, D_x = (mean-flow momentum flux through the box) + (int F_x dV over the box)."""
    xs = XX[0, :]; ys = YY[:, 0]
    U = np.nan_to_num(U); V = np.nan_to_num(V); p = np.nan_to_num(p)
    Ux = np.gradient(U, xs, axis=1); Uy = np.gradient(U, ys, axis=0); Vx = np.gradient(V, xs, axis=1)
    Gx = -p + 2 * NU * Ux - U * U                 # mean-flow x-momentum flux, x-faces (no Reynolds)
    Gy = NU * (Uy + Vx) - U * V                   # mean-flow x-momentum flux, y-faces
    jx1 = int(np.argmin(np.abs(xs - box["x1"]))); jx2 = int(np.argmin(np.abs(xs - box["x2"])))
    iyl = int(np.argmin(np.abs(ys + box["y0"]))); iyh = int(np.argmin(np.abs(ys - box["y0"])))
    ym = (ys >= ys[iyl]) & (ys <= ys[iyh]); xm = (xs >= xs[jx1]) & (xs <= xs[jx2])
    Dmean = (trapz(Gx[ym, jx2], ys[ym]) - trapz(Gx[ym, jx1], ys[ym])
             + trapz(Gy[iyh, xm], xs[xm]) - trapz(Gy[iyl, xm], xs[xm]))
    Fb = np.nan_to_num(Fx)[iyl:iyh + 1, jx1:jx2 + 1]     # streamwise Reynolds force in the box
    Dforce = trapz(trapz(Fb, xs[jx1:jx2 + 1], axis=1), ys[iyl:iyh + 1])
    return 2.0 * (Dmean + Dforce)
