"""Analytic, differentiable signed-distance functions (SDFs) and boundary-point
samplers for the 9 bluff bodies, defined in the PIV data frame (D=1).

CONVENTION (verified against the PIV masks, see shapes_sdf.md):
  * circle / ellipse / square are centroid-centered at the origin.
  * triangles have their DOWNSTREAM-most vertex/edge at x=0 (NOT centroid
    centered): apex upstream at 0 degree, apex downstream at 180 degree;
    base B=1 (cross-stream), streamwise length L=1 (short) / L=2 (long).

SDF sign convention: NEGATIVE inside the body, POSITIVE in the fluid, ~0 on the
wall.  Each `sdf(x, y)` takes scalar x, y and is autodiff-friendly (used inside
the Reynolds-stress-divergence jacrev), so it must use only smooth/`jnp` ops.

The `mask` in the PIV files is the field-of-view / occluded region, NOT the
body, and is deliberately not used here.
"""
import numpy as np
import jax.numpy as jnp

SQRT2_2 = float(np.sqrt(2.0) / 2.0)   # cos(45) = sin(45)

# ---- shape registry: name -> geometry descriptor -------------------------
# kind 'circle'   : params (r,)
# kind 'ellipse'  : params (ax, ay)
# kind 'polygon'  : params = list of vertices [(x,y), ...] (CCW or CW; sign handled)
SHAPES = {
    "circle":            dict(kind="circle", r=0.5),
    "ellipse_0_degree":  dict(kind="ellipse", ax=1.0, ay=0.5),
    "ellipse_90_degree": dict(kind="ellipse", ax=0.5, ay=1.0),
    "square_0_degree":   dict(kind="polygon",
                              verts=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]),
    "square_45_degree":  dict(kind="polygon",
                              verts=[(SQRT2_2, 0.0), (0.0, SQRT2_2),
                                     (-SQRT2_2, 0.0), (0.0, -SQRT2_2)]),
    # triangles: base B=1 (y in [-0.5,0.5]); downstream-most feature at x=0.
    "short_triangle_0_degree":   dict(kind="polygon",
                                      verts=[(-1.0, 0.0), (0.0, -0.5), (0.0, 0.5)]),
    "short_triangle_180_degree": dict(kind="polygon",
                                      verts=[(0.0, 0.0), (-1.0, -0.5), (-1.0, 0.5)]),
    "long_triangle_0_degree":    dict(kind="polygon",
                                      verts=[(-2.0, 0.0), (0.0, -0.5), (0.0, 0.5)]),
    "long_triangle_180_degree":  dict(kind="polygon",
                                      verts=[(0.0, 0.0), (-2.0, -0.5), (-2.0, 0.5)]),
    # NekRS DNS long-triangle (0 deg): measured placement = PIV one shifted +0.5 in x
    # (apex upstream at (-1.5,0), base face at x=0.5, corners (0.5,+-0.5)).
    "tri_long_0_dns":            dict(kind="polygon",
                                      verts=[(-1.5, 0.0), (0.5, -0.5), (0.5, 0.5)]),
    # NekRS DNS equilateral triangle (0 deg): base 1 at x=0.5 (corners (0.5,+-0.5)),
    # apex upstream at (0.5 - sqrt(3)/2, 0) = (-0.366, 0).
    "tri_equi_0_dns":            dict(kind="polygon",
                                      verts=[(-0.36603, 0.0), (0.5, -0.5), (0.5, 0.5)]),
    # DNS-placed short-triangle 0deg (apex upstream) + 180deg family (apex downstream).
    # Vertices verified against the DNS body hole (URANS/analysis/sdf_gate.py).
    "tri_short_0_dns":           dict(kind="polygon",
                                      verts=[(-0.5, 0.0), (0.5, -0.5), (0.5, 0.5)]),
    "tri_short_180_dns":         dict(kind="polygon",
                                      verts=[(0.5, 0.0), (-0.5, 0.5), (-0.5, -0.5)]),
    "tri_long_180_dns":          dict(kind="polygon",
                                      verts=[(1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5)]),
}


# ======================================================================
#  Differentiable scalar SDFs  sdf(x, y) -> scalar  (negative inside)
# ======================================================================
def _sdf_circle(r):
    def sdf(x, y):
        return jnp.sqrt(x * x + y * y + 1e-30) - r
    return sdf


def _sdf_ellipse(ax, ay):
    """First-order implicit->distance estimate, smooth and sign-correct:
       g = |(x/ax, y/ay)|;  sdf ~ (g - 1) / |grad g|.  Accurate near the wall
       (where the feature matters); degrades only deep inside the body."""
    def sdf(x, y):
        gx, gy = x / ax, y / ay
        g = jnp.sqrt(gx * gx + gy * gy + 1e-30)
        # |grad g| = |(x/ax^2, y/ay^2)| / g
        gnx, gny = x / (ax * ax), y / (ay * ay)
        grad = jnp.sqrt(gnx * gnx + gny * gny + 1e-30)
        return (g - 1.0) * g / grad
    return sdf


def _sdf_polygon(verts):
    """Exact polygon SDF (Inigo Quilez), negative inside. Works for convex or
    concave polygons regardless of vertex winding."""
    V = jnp.asarray(verts, dtype=jnp.float64)
    n = V.shape[0]

    def sdf(x, y):
        p = jnp.array([x, y])
        d = jnp.dot(p - V[0], p - V[0])      # squared distance, init
        s = 1.0
        for i in range(n):
            j = (i - 1) % n
            e = V[j] - V[i]
            w = p - V[i]
            t = jnp.clip(jnp.dot(w, e) / (jnp.dot(e, e) + 1e-30), 0.0, 1.0)
            b = w - e * t
            d = jnp.minimum(d, jnp.dot(b, b))
            # winding-number sign test (three conditions)
            c1 = p[1] >= V[i, 1]
            c2 = p[1] < V[j, 1]
            c3 = (e[0] * w[1] > e[1] * w[0])
            allc = c1 & c2 & c3
            nonec = (~c1) & (~c2) & (~c3)
            s = jnp.where(allc | nonec, -s, s)
        return s * jnp.sqrt(d + 1e-30)
    return sdf


def make_sdf(name):
    """Return a scalar, autodiff-friendly sdf(x, y) for a shape name."""
    g = SHAPES[name]
    if g["kind"] == "circle":
        return _sdf_circle(g["r"])
    if g["kind"] == "ellipse":
        return _sdf_ellipse(g["ax"], g["ay"])
    if g["kind"] == "polygon":
        return _sdf_polygon(g["verts"])
    raise ValueError(g["kind"])


# ======================================================================
#  Boundary-point samplers (no-slip wall points), numpy -> (n, 2)
# ======================================================================
def boundary_points(name, n, rng):
    """Sample ~n points on the body boundary (for the no-slip BC)."""
    g = SHAPES[name]
    if g["kind"] == "circle":
        th = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return np.stack([g["r"] * np.cos(th), g["r"] * np.sin(th)], axis=1)
    if g["kind"] == "ellipse":
        th = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return np.stack([g["ax"] * np.cos(th), g["ay"] * np.sin(th)], axis=1)
    if g["kind"] == "polygon":
        V = np.asarray(g["verts"], dtype=np.float64)
        m = V.shape[0]
        # edge lengths -> sample proportionally
        nxt = np.roll(V, -1, axis=0)
        seglen = np.linalg.norm(nxt - V, axis=1)
        prob = seglen / seglen.sum()
        which = rng.choice(m, size=n, p=prob)
        t = rng.uniform(0.0, 1.0, size=n)
        pts = V[which] + (nxt[which] - V[which]) * t[:, None]
        return pts
    raise ValueError(g["kind"])


# ======================================================================
#  Small self-test
# ======================================================================
if __name__ == "__main__":
    import jax
    rng = np.random.default_rng(0)
    for name in SHAPES:
        g = SHAPES[name]
        sdf = make_sdf(name)
        vsdf = jax.vmap(sdf)
        bp = boundary_points(name, 2000, rng)
        # SDF on boundary points should be ~0
        sb = np.asarray(vsdf(jnp.asarray(bp[:, 0]), jnp.asarray(bp[:, 1])))
        # robust interior point: centroid (circle/ellipse -> origin)
        if g["kind"] == "polygon":
            cx, cy = np.asarray(g["verts"], float).mean(axis=0)
        else:
            cx, cy = 0.0, 0.0
        s_in = float(sdf(jnp.array(cx), jnp.array(cy)))
        s_far = float(sdf(jnp.array(9.0), jnp.array(0.0)))
        print(f"{name:28s} |sdf(boundary)|max={np.abs(sb).max():.2e}  "
              f"sdf(centroid)={s_in:+.3f}  sdf(far)={s_far:+.3f}")
