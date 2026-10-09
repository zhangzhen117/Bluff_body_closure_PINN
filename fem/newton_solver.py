"""Matrix-free, right-preconditioned Newton-GMRES for the deployed weak residual.

The block preconditioner factors frozen Oseen/Stokes (including pressure) and
k-transport matrices with MUMPS. Finite differences include all closure/filter
response. Boundary unknowns are eliminated. No k clipping is applied to Newton
steps: negative k is reported separately from equation convergence. The k block is
optional, allowing the same solver to deploy M1.
"""
import json
import os
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc


def solve(monitor, history, a, ak, bcs, bcs_k, nu_art, output_dir, tag):
    nu_art.value = 0.0
    m = monitor
    m._pack()
    has_k = m.kcheck is not None
    checks = [m.momentum] + ([m.kcheck] if has_k else [])
    free = [np.flatnonzero(c.free) for c in checks]
    n0 = len(free[0])
    scale = np.concatenate([np.sqrt(c.diagonal[ids] * c.volume)
                            for c, ids in zip(checks, free)])
    # Residual norm uses the same mass-diagonal weighting as the stopping test.
    # No division by the tiny initial k residual in the line-search merit.
    def pack():
        m._pack()
        return np.concatenate([m.state.x.array[free[0]]] + ([m.k.x.array[free[1]]] if has_k else []))

    def unpack(x):
        m.state.x.array[free[0]] = x[:n0]
        m.velocity.x.array[:] = m.state.x.array[m.umap]
        m.pressure.x.array[:] = m.state.x.array[m.pmap]
        if has_k:
            m.k.x.array[free[1]] = x[n0:]
        for f in ([m.state, m.velocity, m.pressure] + ([m.k] if has_k else [])):
            f.x.scatter_forward()

    evaluations = 0
    def residual(x):
        nonlocal evaluations
        evaluations += 1
        unpack(x)
        m.update_closure(relax=1.0)
        arrays = []
        for c, ids in zip(checks, free):
            if hasattr(c, "raw_vector"):
                arrays.append(c.raw_vector()[ids])
                continue
            v = assemble_vector(c.form)
            v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            arrays.append(v.array[ids].copy())
            v.destroy()
        return np.concatenate(arrays) / scale

    def factor(form, bc):
        A = assemble_matrix(fem.form(form), bcs=bc)
        A.assemble()
        ksp = PETSc.KSP().create(A.comm)
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setUp()
        return A, ksp, A.createVecRight(), A.createVecLeft()

    x = pack()
    r = residual(x)
    records = []
    reason = 'iteration_limit'
    maxiter = int(os.environ.get('NEWTON_MAX', '60'))
    fdrel = float(os.environ.get('NEWTON_FD', '1e-7'))
    central = os.environ.get('NEWTON_CENTRAL', '0') == '1'
    for it in range(maxiter):
        # Refresh at the base state before forming the physics preconditioner.
        r = residual(x)
        factors = [factor(a, bcs)]
        if has_k:
            factors.append(m.kcheck.factor() if hasattr(m.kcheck, "factor") else factor(ak, bcs_k))
        def precondition(v):
            rhs = v * scale
            out = []
            for (A, ksp, sol, b), ids, part in zip(factors, free, (rhs[:n0], rhs[n0:])):
                b.array[:] = 0.0
                b.array[ids] = part
                ksp.solve(b, sol)
                if ksp.getConvergedReason() <= 0:
                    raise RuntimeError('block factor solve failed')
                out.append(sol.array[ids].copy())
            return np.concatenate(out)
        nv = 0
        def jvp(v):
            nonlocal nv
            nv += 1
            d = precondition(v)
            dn = np.linalg.norm(d)
            if dn == 0:
                return np.zeros_like(v)
            h = fdrel * (1 + np.linalg.norm(x)) / dn
            if central:
                return (residual(x+h*d)-residual(x-h*d))/(2*h)
            return (residual(x + h*d) - r) / h
        operator = LinearOperator((x.size, x.size), matvec=jvp, dtype=float)
        linear_trace = []
        z, info = gmres(operator, -r, rtol=1e-3, atol=0.0,
                        restart=80, maxiter=8, callback=linear_trace.append,
                        callback_type='pr_norm')
        dx = precondition(z)
        linerr = float(np.linalg.norm(jvp(z) + r) / np.linalg.norm(r))
        for A, ksp, sol, b in factors:
            sol.destroy(); b.destroy(); ksp.destroy(); A.destroy()
        norm0 = np.linalg.norm(r)
        accepted = False
        step = 1.0
        for ls in range(16):
            trial = x + step*dx
            rt = residual(trial)
            if np.all(np.isfinite(rt)) and np.linalg.norm(rt) <= (1 - 1e-4*step)*norm0:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            residual(x)
            reason = 'line_search_failed'
            print(f'NEWTON {it}: line search failed; GMRES info={info} relative error={linerr:.3e}', flush=True)
            break
        unpack(x)
        old_u = m.velocity.x.array.copy()
        old_k = m.k.x.array.copy() if has_k else None
        x = trial
        r = residual(x)
        du = np.linalg.norm(m.velocity.x.array-old_u)/(np.linalg.norm(m.velocity.x.array)+1e-30)
        dk = float(np.linalg.norm(m.k.x.array-old_k)/(np.linalg.norm(m.k.x.array)+1e-30)) if has_k else None
        values, div = m.evaluate()
        converged = history.record(len(history.rows), values, du, dk, div, 0.0)
        kmin = float(m.k.x.array.min()) if has_k else None
        neg = int(np.sum(m.k.x.array < -1e-10)) if has_k else 0
        row = dict(iteration=it, residuals=values, step=step, du=float(du), dk=dk,
                   gmres_info=int(info), gmres_iterations=len(linear_trace),
                   linear_relative_error=linerr, k_min=kmin, k_negative_dofs=neg,
                   evaluations=evaluations, equation_converged=converged)
        records.append(row)
        print('NEWTON '+json.dumps(row), flush=True)
        if converged:
            reason = 'converged' if neg == 0 else 'root_with_negative_k'
            break
    residual(x)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, tag+'_dofs.npz'), velocity=m.velocity.x.array,
             pressure=m.pressure.x.array, **({"k":m.k.x.array} if has_k else {}))
    result = dict(reason=reason, records=records, evaluations=evaluations,
                  initial=history.initial, fd_relative_step=fdrel, fd_scheme='central' if central else 'forward',
                  admissible=bool(np.min(m.k.x.array) >= -1e-10) if has_k else True)
    with open(os.path.join(output_dir, tag+'_newton.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('NEWTON_RESULT '+json.dumps({k:v for k,v in result.items() if k not in ('records',)}), flush=True)
