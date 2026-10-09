"""Mean-momentum Reynolds force F=(Fx,Fy) from a gridded steady mean field.

The steady incompressible RANS momentum balance is
    (U.grad)U + grad P - nu lap U - F = 0,   so   F = (U.grad)U + grad P - nu lap U,
where F = -div(tau) is the divergence of the Reynolds stress (the forcing-vector form
of the closure). Computing F the SAME way for DNS, SST and the M4 FEM field gives a
like-for-like comparison of the closure target, independent of how each model represents
the stress internally. All fields are (Ny,Nx) on the structured grid XX,YY.
Run in the jaxpi env (numpy/scipy only)."""
import numpy as np


def reynolds_force(XX, YY, U, V, P, nu):
    """F=(Fx,Fy) = conv + grad P - nu lap U on the structured grid (np.gradient)."""
    xs = XX[0, :]; ys = YY[:, 0]
    U = np.nan_to_num(U); V = np.nan_to_num(V); P = np.nan_to_num(P)
    Ux = np.gradient(U, xs, axis=1); Uy = np.gradient(U, ys, axis=0)
    Vx = np.gradient(V, xs, axis=1); Vy = np.gradient(V, ys, axis=0)
    Px = np.gradient(P, xs, axis=1); Py = np.gradient(P, ys, axis=0)
    Uxx = np.gradient(Ux, xs, axis=1); Uyy = np.gradient(Uy, ys, axis=0)
    Vxx = np.gradient(Vx, xs, axis=1); Vyy = np.gradient(Vy, ys, axis=0)
    Fx = U * Ux + V * Uy + Px - nu * (Uxx + Uyy)
    Fy = U * Vx + V * Vy + Py - nu * (Vxx + Vyy)
    return Fx, Fy


def reynolds_force_from_stress(XX, YY, uu, uv, vv):
    """Reynolds force F = -div<u'u'> directly from the Reynolds-stress field (uu,uv,vv).
    Identical to reynolds_force in the exact mean balance (conv+gradP-nu lap U = -div<u'u'>),
    but uses ONE derivative of the measured stress instead of two of the velocity, and -> 0 at
    a no-slip wall where the turbulence vanishes (uu,uv,vv -> 0). This removes the spurious
    near-body peaks that twice-differentiating an interpolated DNS velocity produces, so it is
    the clean way to build the DNS reference force. Fields are (Ny,Nx) on the grid XX,YY."""
    xs = XX[0, :]; ys = YY[:, 0]
    uu = np.nan_to_num(uu); uv = np.nan_to_num(uv); vv = np.nan_to_num(vv)
    Fx = -(np.gradient(uu, xs, axis=1) + np.gradient(uv, ys, axis=0))
    Fy = -(np.gradient(uv, xs, axis=1) + np.gradient(vv, ys, axis=0))
    return Fx, Fy
