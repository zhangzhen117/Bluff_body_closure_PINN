"""Numpy-only shape geometry (SDF + half-domain upper-chain) for the FEniCS solver.
Ported from ../geometry.py and ../openfoam/gen_geo.py (no jax). SDF: negative
inside body, positive in fluid. Vectorized over arrays of (x,y)."""
import numpy as np

SQRT2_2 = float(np.sqrt(2.0) / 2.0)
SHAPES = {
    "circle":            dict(kind="circle", r=0.5),
    "ellipse_0_degree":  dict(kind="ellipse", ax=1.0, ay=0.5),
    "ellipse_90_degree": dict(kind="ellipse", ax=0.5, ay=1.0),
    "square_0_degree":   dict(kind="polygon", verts=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]),
    "square_45_degree":  dict(kind="polygon", verts=[(SQRT2_2, 0.0), (0.0, SQRT2_2), (-SQRT2_2, 0.0), (0.0, -SQRT2_2)]),
    "short_triangle_0_degree":   dict(kind="polygon", verts=[(-1.0, 0.0), (0.0, -0.5), (0.0, 0.5)]),
    "short_triangle_180_degree": dict(kind="polygon", verts=[(0.0, 0.0), (-1.0, -0.5), (-1.0, 0.5)]),
    "long_triangle_0_degree":    dict(kind="polygon", verts=[(-2.0, 0.0), (0.0, -0.5), (0.0, 0.5)]),
    "long_triangle_180_degree":  dict(kind="polygon", verts=[(0.0, 0.0), (-2.0, -0.5), (-2.0, 0.5)]),
    "tri_long_0_dns":            dict(kind="polygon", verts=[(-1.5, 0.0), (0.5, -0.5), (0.5, 0.5)]),
    "tri_equi_0_dns":            dict(kind="polygon", verts=[(-0.36603, 0.0), (0.5, -0.5), (0.5, 0.5)]),
    "tri_short_0_dns":           dict(kind="polygon", verts=[(-0.5, 0.0), (0.5, -0.5), (0.5, 0.5)]),
    # 180-degree (apex downstream) DNS-placed triangles; CCW. Provisional (partial DNS runs, deferred).
    "tri_short_180_dns":         dict(kind="polygon", verts=[(0.5, 0.0), (-0.5, 0.5), (-0.5, -0.5)]),
    "tri_long_180_dns":          dict(kind="polygon", verts=[(1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5)]),
}


def _sdf_polygon_np(verts):
    V = np.asarray(verts, float); n = len(V)
    def sdf(x, y):
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        P = np.stack([x, y], -1)                       # (N,2)
        d = np.sum((P - V[0]) ** 2, -1)
        s = np.ones_like(x)
        for i in range(n):
            j = (i - 1) % n
            e = V[j] - V[i]; w = P - V[i]
            t = np.clip((w @ e) / (e @ e + 1e-30), 0.0, 1.0)
            b = w - np.outer(t, e)
            d = np.minimum(d, np.sum(b ** 2, -1))
            c1 = P[:, 1] >= V[i, 1]; c2 = P[:, 1] < V[j, 1]
            c3 = (e[0] * w[:, 1] > e[1] * w[:, 0])
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            s = np.where(flip, -s, s)
        return s * np.sqrt(d + 1e-30)
    return sdf


def make_sdf(name):
    g = SHAPES[name]
    if g["kind"] == "circle":
        return lambda x, y: np.sqrt(np.asarray(x, float) ** 2 + np.asarray(y, float) ** 2 + 1e-30) - g["r"]
    if g["kind"] == "ellipse":
        ax, ay = g["ax"], g["ay"]
        def sdf(x, y):
            x = np.asarray(x, float); y = np.asarray(y, float)
            gx, gy = x / ax, y / ay
            G = np.sqrt(gx ** 2 + gy ** 2 + 1e-30)
            grad = np.sqrt((x / ax ** 2) ** 2 + (y / ay ** 2) ** 2 + 1e-30)
            return (G - 1.0) * G / grad
        return sdf
    return _sdf_polygon_np(g["verts"])


def full_ring(name, n=200):
    """Closed CCW boundary points of the body, for a FULL-domain body-fitted mesh hole.
    Polygons return their stored corner loop (already CCW); circle/ellipse are sampled at n points
    over the full 2*pi. No duplicated closing point (mesh builder wraps modulo len)."""
    g = SHAPES[name]
    if g["kind"] in ("circle", "ellipse"):
        ax = g.get("r", g.get("ax")); ay = g.get("r", g.get("ay"))
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([ax * np.cos(t), ay * np.sin(t)], 1)
    return np.asarray([tuple(map(float, p)) for p in g["verts"]], float)


def upper_chain(name, narc=160):
    """Ordered (x,y) along the body's upper-half perimeter, downstream (max x) ->
    upstream (min x). Used to build the half-domain mesh boundary."""
    g = SHAPES[name]
    if g["kind"] in ("circle", "ellipse"):
        ax = g.get("r", g.get("ax")); ay = g.get("r", g.get("ay"))
        t = np.linspace(0.0, np.pi, narc)
        return np.stack([ax * np.cos(t), ay * np.sin(t)], 1)
    V = [tuple(map(float, p)) for p in g["verts"]]; n = len(V)
    aug = []
    for i in range(n):
        x0, y0 = V[i]; x1, y1 = V[(i + 1) % n]
        aug.append((x0, y0))
        if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
            tt = y0 / (y0 - y1); aug.append((x0 + tt * (x1 - x0), 0.0))
    m = len(aug); yv = [p[1] for p in aug]
    if all(v >= -1e-12 for v in yv):
        run = list(range(m))
    else:
        start = next(i for i in range(m) if yv[i] < -1e-12)
        run = [(start + 1 + k) % m for k in range(m) if yv[(start + 1 + k) % m] >= -1e-12]
    chain = [aug[i] for i in run]
    if chain[0][0] < chain[-1][0]:
        chain = chain[::-1]
    return np.asarray(chain)


if __name__ == "__main__":
    for nm in ["circle", "tri_long_0_dns", "square_45_degree"]:
        sdf = make_sdf(nm); ch = upper_chain(nm)
        sb = sdf(ch[:, 0], ch[:, 1])
        print(f"{nm:18s} chain {ch.shape[0]} pts, x[{ch[:,0].min():.2f},{ch[:,0].max():.2f}]; "
              f"|sdf(boundary)|max={np.abs(sb).max():.1e}; sdf(5,0)={float(np.ravel(sdf(5.0,0.0))[0]):.2f}")
