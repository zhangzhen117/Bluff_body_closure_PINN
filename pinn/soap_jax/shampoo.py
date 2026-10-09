"""A compact full-matrix Shampoo optimizer (optax GradientTransformation).

Used as a second-order *baseline* against SOAP in the cylinder-PINN ablation.
SOAP runs Adam in the eigenbasis of the Shampoo preconditioner; classic Shampoo
instead preconditions the raw gradient directly by the inverse (2k)-th root of the
per-axis Gram matrices.  For a weight G with shape (m, n):

    L = EMA(G Gᵀ)  (m×m),   R = EMA(Gᵀ G)  (n×n)
    update = L^{-1/4} · G · R^{-1/4}            (exponent -1/(2·ndim))

with Heavy-ball / Adam-style first-moment momentum on the preconditioned update.
1-D params (biases) use a single-axis preconditioner with exponent -1/2.  The net
here is small (≤80×80 matrices) so we eigendecompose every `precondition_frequency`
steps.  Reuses the Preconditioner pytree + EMA update from soap.py.
"""
from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from optax import GradientTransformation, Updates

from .soap import (Preconditioner, init_conditioner, update_preconditioner,
                   _is_preconditioner)


class ShampooState(NamedTuple):
    count: jax.Array
    m: Updates           # Adam first moment of the raw gradient (for direction + graft)
    v: Updates           # Adam second moment of the raw gradient (for the graft norm)
    H: Updates           # pytree of Preconditioner (per-axis Gram EMAs)
    P: Updates           # pytree of Preconditioner (cached inverse-root matrices)


def _inv_root(m, exponent, eps):
    """Symmetric PSD matrix raised to `exponent` (< 0) with eigenvalue damping."""
    if m is None:
        return None
    msym = 0.5 * (m + m.T)
    w, v = jnp.linalg.eigh(msym)
    w = jnp.maximum(w, 0.0)
    d = jnp.power(w + eps, exponent)
    return (v * d) @ v.T


def _precondition(grad, P):
    """Apply the cached inverse-root matrices along each axis of grad."""
    g = grad
    n = g.ndim
    for axis, mat in enumerate(P.matrices):
        if mat is None:
            continue
        # contract grad's `axis` with mat, result axis goes last -> move it back
        g = jnp.tensordot(g, mat, axes=((axis,), (0,)))
        g = jnp.moveaxis(g, -1, axis)
    return g


def scale_by_shampoo(b1=0.9, b2=0.99, beta=0.95, eps=1e-8, graft_eps=1e-12,
                     precondition_frequency=10, max_precond_dim=10000):
    """Full-matrix Shampoo with Adam-norm grafting: the *direction* comes from the
    Shampoo preconditioner, the per-tensor *magnitude* from the Adam update.  This
    fixes bare Shampoo's ill-calibrated step size and matches how production Shampoo
    (Google distributed_shampoo) is run."""
    def init_fn(params):
        m = otu.tree_zeros_like(params)
        v = otu.tree_zeros_like(params)
        H = jtu.tree_map(lambda p: init_conditioner(p, max_precond_dim, True, jnp.float32), params)
        P = jtu.tree_map(lambda p: init_conditioner(p, max_precond_dim, True, jnp.float32), params)
        return ShampooState(count=jnp.zeros([], jnp.int32), m=m, v=v, H=H, P=P)

    def update_fn(updates, state, params=None):
        del params
        count = optax.safe_int32_increment(state.count)
        # Adam moments of the raw gradient (direction source + graft magnitude)
        m = otu.tree_update_moment(updates, state.m, b1, 1)
        v = otu.tree_update_moment_per_elem_norm(updates, state.v, b2, 2)
        m_hat = otu.tree_bias_correction(m, b1, count)
        v_hat = otu.tree_bias_correction(v, b2, count)
        adam_upd = jtu.tree_map(lambda mh, vh: mh / (jnp.sqrt(vh) + eps), m_hat, v_hat)

        # EMA of per-axis Gram matrices (from the raw gradient)
        new_H = jtu.tree_map(
            lambda g, h: update_preconditioner(g, h, beta),
            updates, state.H, is_leaf=_is_preconditioner)

        # refresh inverse-root matrices every `precondition_frequency` steps
        def refresh():
            return jtu.tree_map(
                lambda g, h: Preconditioner([
                    _inv_root(mm, -1.0 / (2 * max(g.ndim, 1)), eps) for mm in h.matrices]),
                updates, new_H, is_leaf=_is_preconditioner)
        new_P = jax.lax.cond((count - 1) % precondition_frequency == 0,
                             refresh, lambda: state.P)

        # Shampoo direction = preconditioned (momentum) gradient
        shampoo_dir = jtu.tree_map(
            lambda mh, p: _precondition(mh, p), m_hat, new_P, is_leaf=_is_preconditioner)

        # graft: keep Shampoo's direction, scale to the Adam update's per-tensor norm
        grafted = jtu.tree_map(
            lambda s, a: s * (jnp.linalg.norm(a) / (jnp.linalg.norm(s) + graft_eps)),
            shampoo_dir, adam_upd)

        return grafted, ShampooState(count=count, m=m, v=v, H=new_H, P=new_P)

    return GradientTransformation(init_fn, update_fn)


def shampoo(learning_rate=3e-3, b1=0.9, b2=0.99, beta=0.95, eps=1e-8,
            precondition_frequency=10, max_precond_dim=10000):
    return optax.chain(
        scale_by_shampoo(b1=b1, b2=b2, beta=beta, eps=eps,
                         precondition_frequency=precondition_frequency,
                         max_precond_dim=max_precond_dim),
        optax.scale_by_learning_rate(learning_rate),
    )
