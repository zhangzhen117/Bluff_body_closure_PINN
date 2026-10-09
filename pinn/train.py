"""Phase A: jointly train per-shape mean-field nets + the shared closure net.

  params = {"closure": cl_params, "mf": {shape: mf_params}}

SOAP optimizer + pseudo-time stepping + grad-norm dynamic loss weighting, all
ported from the cylinder PINN (cyl_pinn_pt.py) to the multi-shape steady setting.

Usage:
  python train.py --tag cyl --split cylinder --steps 40000
  python train.py --tag canon --split canonical --steps 80000
"""
import os
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad, lax, random
from jax.tree_util import tree_leaves
import optax

from soap_jax import soap
import config
import geometry as geo
import data as dataio
import model as M
import rans
import sampling
import common

HERE = config.HERE


def get_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cyl")
    p.add_argument("--split", default="cylinder")
    p.add_argument("--loso_shape", default=None)
    p.add_argument("--dns_csv", default=None,
                   help="if set, the single shape's data comes from this DNS CSV")
    p.add_argument("--use_dns", action="store_true",
                   help="multi-shape: each shape's data comes from its config.DNS_LINKS entry")
    p.add_argument("--init_closure", default=None,
                   help="tag of a pretrained closure to initialize from (transfer learning)")
    p.add_argument("--train_last_only", action="store_true",
                   help="freeze all closure layers except the output Dense (last-layer fine-tune)")
    for k, v in config.DEFAULTS.items():
        argtype = type(v)
        p.add_argument(f"--{k}", type=argtype, default=v)
    return p.parse_args()


def main():
    args = get_cfg()
    cfg = config.base_cfg(**{k: getattr(args, k) for k in config.DEFAULTS})
    train_shapes, _ = config.split_lists(args.split, args.loso_shape)
    print(f"[{args.tag}] devices: {jax.devices()} | split={args.split} "
          f"| train_shapes={train_shapes}", flush=True)

    MF = M.make_meanfield_arch(cfg)
    CL = M.make_closure_arch(cfg)
    SDFS = {s: geo.make_sdf(s) for s in train_shapes}
    KEYS = rans.train_keys(cfg)

    key = random.PRNGKey(cfg.seed)
    key, kc = random.split(key)
    cl_params = CL.init(kc, jnp.ones(M.n_closure_features(cfg)))
    if args.init_closure:   # transfer learning: load a pretrained closure
        ick = os.path.join(config.CKPT_DIR, f"{args.init_closure}_params.npz")
        cl_params = common.load_params(ick, {"closure": cl_params})["closure"]
        print(f"[{args.tag}] initialized closure from {ick}", flush=True)
    params = {"closure": cl_params, "mf": {}}
    for s in train_shapes:
        key, ks = random.split(key)
        params["mf"][s] = MF.init(ks, jnp.ones(2))
    nparam = sum(x.size for x in tree_leaves(params))

    # transfer learning: optionally train ONLY the closure's output Dense (+ all mf)
    train_mask = None
    if args.train_last_only:
        clp = params["closure"]["params"]
        cl_out = 2 if getattr(cfg, "use_keq", False) else 3
        out_name = next(n for n, lyr in clp.items()
                        if "kernel" in lyr and lyr["kernel"].shape[-1] == cl_out)
        def _mask(path, _leaf):
            keys = [getattr(p, "key", None) for p in path]
            if keys and keys[0] == "mf":
                return True                      # all mean-field params trainable
            return out_name in keys              # closure: only the output Dense
        train_mask = jax.tree_util.tree_map_with_path(_mask, params)
        ntrain = sum(x.size for x, m in zip(tree_leaves(params), tree_leaves(train_mask)) if m)
        print(f"[{args.tag}] train_last_only: trainable params {ntrain}/{nparam} "
              f"(closure output = {out_name})", flush=True)
    print(f"[{args.tag}] total params: {nparam}", flush=True)

    # data + collocation pools per shape
    shapes = {}
    for s in train_shapes:
        if args.dns_csv:                       # single-shape DNS benchmark
            shapes[s] = dataio.load_dns(args.dns_csv, name=s)
            src = args.dns_csv
        elif args.use_dns:                     # multi-shape: per-shape DNS via link map
            shapes[s] = dataio.load_dns(config.dns_path(s), name=s)
            src = config.dns_path(s)
        else:
            shapes[s] = dataio.load_shape(s); src = "PIV"
        print(f"[{args.tag}] {s}: {shapes[s].data_xy.shape[0]} pts ({src})", flush=True)
    pools = {s: sampling.precompute_pools(s, cfg, seed=cfg.seed) for s in train_shapes}
    print(f"[{args.tag}] coll pool sizes: "
          f"{ {s: int(pools[s]['coll_pool'].shape[0]) for s in train_shapes} }", flush=True)

    # ---- loss / weighting ----
    def shape_loss_dict(params, s, batch, dat, sol_prev, theta):
        return rans.loss_terms(MF, CL, cfg, SDFS[s], params["closure"], params["mf"][s],
                               batch, dat, sol_prev, theta, with_data=True)

    TAU_KEYS = {"tau_uu", "tau_uv", "tau_vv", "tau_zz"}
    PDE_KEYS_SET = set(rans.PDE_KEYS)
    K_KEYS_SET = set(rans.K_ALL_KEYS)      # non-local transported-k terms (use_keq)

    def kscale(k):
        if k in TAU_KEYS:
            return cfg.tau_weight
        if k in K_KEYS_SET:
            return cfg.k_weight            # rk / data_k / k-BCs (transported-k closure)
        if k in PDE_KEYS_SET:
            return cfg.pde_weight          # 0 -> pure-supervised (no physics residual)
        return 1.0

    def total_loss(params, batches, datas, sol_prevs, thetas, weights):
        total = 0.0
        aux = {}
        for s in train_shapes:
            d = shape_loss_dict(params, s, batches[s], datas[s], sol_prevs[s], thetas[s])
            total = total + sum(weights[s][k] * kscale(k) * d[k] for k in KEYS)
            if cfg.lip_weight > 0.0:
                total = total + cfg.lip_weight * d["lip"]   # unified closure-Lipschitz penalty (M1-M4)
            if getattr(cfg, "force_sup", 0.0) > 0.0:
                total = total + cfg.force_sup * d.get("data_F", 0.0)   # direct DNS-force supervision
            aux[s] = d
        return total, aux

    def grad_norm_weights_shape(params, s, batch, dat, sol_prev, theta):
        def single(key):
            g = grad(lambda p: shape_loss_dict(p, s, batch, dat, sol_prev, theta)[key])(params)
            return jnp.sqrt(sum(jnp.sum(l ** 2) for l in tree_leaves(g)))
        norms = {k: single(k) for k in KEYS}
        mean = jnp.mean(jnp.stack([norms[k] for k in KEYS]))
        return {k: mean / (norms[k] + 1e-5 * mean) for k in KEYS}

    gnorm_jit = jit(grad_norm_weights_shape, static_argnums=(1,))

    # ---- optimizer ----
    lr = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup,
        transition_steps=cfg.decay_steps, decay_rate=cfg.decay_rate, staircase=False)
    if getattr(cfg, "opt", "soap") == "adam":           # ablation: Adam instead of SOAP
        tx = optax.adam(learning_rate=lr, b1=0.95, b2=0.95, eps=1e-8)
    else:
        tx = soap(learning_rate=lr, b1=0.95, b2=0.95, eps=1e-8, weight_decay=0.0,
                  precondition_frequency=cfg.precond_freq, max_precond_dim=10000)
    if train_mask is not None:
        tx = optax.masked(tx, train_mask)   # freeze non-trainable leaves
    opt_state = tx.init(params)

    # ---- state: weights, theta, prev params ----
    weights = {s: {k: jnp.array(1.0) for k in KEYS} for s in train_shapes}
    thetas = {s: jnp.ones(3) for s in train_shapes}
    prev_params = params

    # baseline residual per shape for the cosine-shrink schedule
    losses0 = {}
    for s in train_shapes:
        key, kb = random.split(key)
        b = sampling.sample_step(kb, pools[s], cfg)
        losses0[s] = rans.residual_mse_per_comp(MF, CL, cfg, SDFS[s],
                                                params["closure"], params["mf"][s], b["coll"])

    def sol_prev_of(prev_params, batches):
        sp = {}
        for s in train_shapes:
            coll = batches[s]["coll"]
            u, v, p = vmap(lambda x, y: M.mf_fwd(MF, prev_params["mf"][s], x, y))(
                coll[:, 0], coll[:, 1])
            sp[s] = lax.stop_gradient(jnp.stack([u, v, p], axis=1))
        return sp

    @jit
    def step(params, prev_params, opt_state, batches, datas, thetas, weights):
        sol_prevs = sol_prev_of(prev_params, batches)
        (loss, aux), g = value_and_grad(total_loss, has_aux=True)(
            params, batches, datas, sol_prevs, thetas, weights)
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    # per-shape jitted pseudo-time-weight fns (close over the static MF/CL/cfg/sdf)
    def make_pt_fn(sdf_s):
        @jit
        def f(cl_params, mf_params, prev_mf_params, coll, losses0):
            return rans.pseudo_time_weights(MF, CL, cfg, sdf_s, cl_params, mf_params,
                                            prev_mf_params, coll, losses0)
        return f
    pt_fns = {s: make_pt_fn(SDFS[s]) for s in train_shapes}

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(config.CKPT_DIR, f"{args.tag}_params.npz")
    t0 = time.time()
    for it in range(cfg.steps):
        batches, datas = {}, {}
        for s in train_shapes:
            key, kb, kd = random.split(key, 3)
            batches[s] = sampling.sample_step(kb, pools[s], cfg)
            datas[s] = sampling.sample_data(kd, shapes[s], cfg)

        # grad-norm balancing (per shape)
        if not getattr(cfg, "no_gradnorm", 0) and it % cfg.w_update_every == 0:   # ablation: uniform weights
            sp = sol_prev_of(params, batches)
            for s in train_shapes:
                nw = gnorm_jit(params, s, batches[s], datas[s], sp[s], thetas[s])
                weights[s] = jax.tree_util.tree_map(
                    lambda o, n: cfg.w_momentum * o + (1 - cfg.w_momentum) * n,
                    weights[s], nw)

        # adaptive pseudo-time weights
        if not getattr(cfg, "no_pt", 0) and it % cfg.pt_update_every == 0 and it > 0:   # ablation: constant θ=1
            for s in train_shapes:
                nt = pt_fns[s](params["closure"], params["mf"][s],
                               prev_params["mf"][s], batches[s]["coll"], losses0[s])
                thetas[s] = cfg.pt_momentum * thetas[s] + (1 - cfg.pt_momentum) * nt

        new_params, opt_state, loss, aux = step(
            params, prev_params, opt_state, batches, datas, thetas, weights)
        prev_params = params
        params = new_params

        if it % 1000 == 0 or it == cfg.steps - 1:
            s0 = train_shapes[0]
            d = {k: float(aux[s0][k]) for k in KEYS}
            pde = d["ru"] + d["rv"] + d["rc"]
            lip = float(aux[s0].get("lip", 0.0))
            ktxt = ""
            if getattr(cfg, "use_keq", False):
                ktxt = (f" | rk {d['rk']:.2e} datak {d['data_k']:.2e} "
                        f"kwall {d['k_wall']+d['k_in']:.2e}")
            # force model has no stress terms; report the freestream-force anchor instead
            mtxt = (f"Ffree {d['F_free']:.2e}" if getattr(cfg, "use_force", False)
                    else f"tau {d['tau_uu']+d['tau_uv']+d['tau_vv']:.2e}")
            print(f"step {it:6d} | loss {float(loss):.3e} | [{s0}] pde {pde:.2e} "
                  f"data {d['data_u']+d['data_v']:.2e} {mtxt} "
                  f"wall {d['u_wall']+d['v_wall']:.2e} lip {lip:.2e}{ktxt} | theta {np.array(thetas[s0]).round(2)} "
                  f"| {time.time()-t0:.0f}s", flush=True)
        if cfg.err_every > 0 and it % cfg.err_every == 0 and it > 0:
            parts = []
            for s in train_shapes:
                e = common.field_rel_l2(MF, params["mf"][s], shapes[s])
                nw = common.near_wall_rel_l2(MF, params["mf"][s], shapes[s], SDFS[s], band=0.3)
                parts.append(f"{s[:14]} {e:.3f}/{nw:.3f}")
            print(f"  [relL2 all/near-wall @ {it}] " + " | ".join(parts), flush=True)
        if it % 20000 == 0 and it > 0:
            common.save_params(ckpt, params)

    common.save_params(ckpt, params)
    print(f"[{args.tag}] saved {ckpt}", flush=True)
    # training-shape fit (sanity) + near-wall fit vs benchmark
    for s in train_shapes:
        e = common.field_rel_l2(MF, params["mf"][s], shapes[s])
        nw = common.near_wall_rel_l2(MF, params["mf"][s], shapes[s], SDFS[s], band=0.3)
        print(f"[{args.tag}] TRAIN field rel-L2 {s}: all={e:.4f}  near-wall(<0.3)={nw:.4f}", flush=True)


if __name__ == "__main__":
    main()
