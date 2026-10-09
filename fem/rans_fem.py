# Revised deployment: strain-based physical filter; coupled nonlinear residual stopping.
"""Steady incompressible RANS in dolfinx with the frozen neural closure.

Taylor-Hood (P2/P1), half-domain body-fitted mesh. Reynolds stress
tau = k*I - 2*nu_t*S + g2*(S*Om-Om*S): nu_t implicit (turbulent viscosity),
(k*I + g2*T2) explicit source. Picard iteration; closure coefficients updated
each step from grad(u_n) via the numpy port. Saves U,V,p and the Reynolds-stress
tensor (uu,uv,vv) on the common grid.

Default input smoothing: variable-coefficient Helmholtz filter with physical
length delta/D=Cf/(1+sqrt(2 S:S) D/U_inf), Cf=0.2 (--filter_cf)
+ SUPG (--supg). Input-smoothing removes the cell-scale numerical striations from
the closure INPUT -> clean off-body stress, better Reynolds-stress accuracy, and
it breaks the closure-jitter feedback so the Picard iteration converges; SUPG then
tightens the convection operator the rest of the way. This pairs with the
Lipschitz-penalty-trained closure (train.py --lip_weight).

The closure is wall-damped by the SDF (--wall_d0): turbulence vanishes at the
wall and the NN is least reliable / noisiest there. The pointwise NN stress is
denoised for output with a monotone Helmholtz filter (--out_filt).

Usage:
  python rans_fem.py --shape tri_long_0_dns --ckpt is_loc_trl               # locked config
  python rans_fem.py --shape tri_long_0_dns --ckpt is_loc_trl --filter_mode none  # disable smoothing
"""
import os, sys, argparse
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix.ufl
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
import mesh2d, geom_np, closure_np
from deployment_checks import InputGradientFilter, NonlinearMonitor, ConvergenceHistory, resolve_filter_options

NU = 1.0e-4


def run(shape, ckpt, laminar=False, picard=500, alpha=0.4, tol=1e-4, wall_d0=0.20,
        nu_floor=0.0, filt=0.0, out_filt=0.03, diag=False, npow=25, supg=True, dc=False, nu_phys=NU,
        hwall=0.02, hfar=0.8, wake_h=0.08, anderson=False, nut_max=0.0, gate_w0=0.0, train_wall_d0=0.0,
        evonly=False, no_g2=False, insmooth=None, outtag="", filter_mode=None, filter_cf=0.2, residual_rtol=1e-6,
        residual_atol=1e-8, output_dir=None):
    filter_mode, insmooth = resolve_filter_options(filter_mode, insmooth)
    if picard < 1 or not 0 < alpha <= 1:
        raise ValueError("picard must be positive and alpha must lie in (0,1]")
    if not outtag:
        outtag = f"_{filter_mode}{filter_cf:g}_residual" if filter_mode == "strain" else f"_{filter_mode}_residual"
    msh, ct, ft = mesh2d.build_mesh(shape, hwall=hwall, hfar=hfar, wake_h=wake_h)
    if msh.comm.size != 1:
        raise ValueError("The existing common-grid exporter requires one MPI rank; use threaded Slurm tasks")
    sdf = geom_np.make_sdf(shape)
    W = closure_np.load_closure(os.path.join(closure_np.config.CKPT_DIR, f"{ckpt}_params.npz"))

    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(2,))
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    ME = fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
    V0, _ = ME.sub(0).collapse()
    Q0 = fem.functionspace(msh, ("DG", 0))
    Qt = fem.functionspace(msh, ("DG", 0, (2, 2)))

    # --- BCs ---
    def vdofs(space, sub, tags):
        facets = np.concatenate([ft.find(t) for t in tags])
        return fem.locate_dofs_topological((sub, space), 1, facets) if space is not None \
            else fem.locate_dofs_topological(sub, 1, facets)
    u_in = fem.Function(V0); u_in.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    u_w = fem.Function(V0); u_w.x.array[:] = 0.0
    bc_in = fem.dirichletbc(u_in, vdofs(V0, ME.sub(0), [mesh2d.TAGS["inlet"]]), ME.sub(0))
    bc_wall = fem.dirichletbc(u_w, vdofs(V0, ME.sub(0), [mesh2d.TAGS["wall"]]), ME.sub(0))
    V0y, _ = ME.sub(0).sub(1).collapse()
    zy = fem.Function(V0y); zy.x.array[:] = 0.0
    bc_sym = fem.dirichletbc(zy, vdofs(V0y, ME.sub(0).sub(1),
                             [mesh2d.TAGS["symmetry"], mesh2d.TAGS["top"]]), ME.sub(0).sub(1))
    bcs = [bc_in, bc_wall, bc_sym]

    # --- coefficient Functions (updated each Picard step) ---
    # The NN closure is evaluated at DG0 cell centroids (local + stable). NB: an
    # L2 gradient recovery / CG1 projection (the textbook denoiser) instead feeds
    # Gibbs overshoot from the sharp corner and the nu_t cap into the NN and blows
    # up Picard -- it needs consistent (SUPG) stabilization to use. The SDF wall-
    # damping phi(d) is what tames the near-wall noise here: it zeroes the closure
    # in the thin band where gradients are largest / least reliable, which also
    # keeps nu_t off its cap.
    u_n = fem.Function(V0)
    u_n.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))  # freestream init
    nut = fem.Function(Q0); te = fem.Function(Qt)             # DG0 closure, used in form (stable)
    coords = Q0.tabulate_dof_coordinates()[:, :2]
    dvals = sdf(coords[:, 0], coords[:, 1])
    phi = closure_np.wall_damp(dvals, wall_d0)                # SDF wall-damping of the closure
    gspaces = [fem.Function(Q0) for _ in range(4)]            # ux,uy,vx,vy at centroids
    gexpr = [fem.Expression(u_n[i // 2].dx(i % 2), Q0.element.interpolation_points()) for i in range(4)]

    # Physical strain-dependent length: delta/D = Cf/(1 + sqrt(2 S:S) D/U_inf).
    # D=U_inf=1 in this solver. --filter_mode cell retains the old c*h comparison.
    gradient_filter = InputGradientFilter(msh, Q0, u_n, mode=filter_mode,
                                          coefficient=filter_cf, cells=insmooth)
    grad_arrays = gradient_filter.arrays

    NUT_CAP = 0.5; RELAX = 0.3

    # Optional Helmholtz / differential filter of the closure source:
    #   (1 - filt^2 * Laplacian) f_bar = f   solved on CG1.
    # Unlike an L2 projection this is monotone (the filt^2*grad term is
    # dissipative), so it denoises the near-body NN output without the Gibbs
    # overshoot that blows up Picard. nu_t is re-clipped after filtering.
    use_filt = filt > 0.0
    if use_filt:
        S1 = fem.functionspace(msh, ("Lagrange", 1))
        G1 = fem.functionspace(msh, ("Lagrange", 1, (2, 2)))
        nut_f = fem.Function(S1); te_f = fem.Function(G1)
        d2 = fem.Constant(msh, PETSc.ScalarType(filt ** 2))
        _sa, _sb = ufl.TrialFunction(S1), ufl.TestFunction(S1)
        _hn = LinearProblem(ufl.inner(_sa, _sb) * ufl.dx + d2 * ufl.inner(ufl.grad(_sa), ufl.grad(_sb)) * ufl.dx,
                            ufl.inner(nut, _sb) * ufl.dx, u=nut_f,
                            petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
        _ta, _tb = ufl.TrialFunction(G1), ufl.TestFunction(G1)
        _ht = LinearProblem(ufl.inner(_ta, _tb) * ufl.dx + d2 * ufl.inner(ufl.grad(_ta), ufl.grad(_tb)) * ufl.dx,
                            ufl.inner(te, _tb) * ufl.dx, u=te_f,
                            petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
        nut_use, te_use = nut_f, te_f
    else:
        nut_use, te_use = nut, te

    def filter_source():
        if not use_filt:
            return
        _hn.solve(); _ht.solve()
        nut_f.x.array[:] = np.clip(nut_f.x.array, 0.0, NUT_CAP)   # keep realizable after filtering

    # cfg carrying the deployment-consistent processing flags (must match training)
    cfg_exp = closure_np.config.base_cfg(nut_max=nut_max, gate_w0=gate_w0, wall_d0=train_wall_d0)

    def closure_fields(ux, uy, vx, vy):
        k, nt, g2 = closure_np.coeffs(W, ux, uy, vx, vy, dvals, cfg=cfg_exp)
        if nut_max <= 0:
            nt = np.clip(nt, 0.0, NUT_CAP)                    # external cap (old checkpoints)
        if train_wall_d0 <= 0:
            k, nt, g2 = phi * k, phi * nt, phi * g2           # external wall-damp (old checkpoints)
        if evonly:                                            # pure eddy viscosity: tau = -2 nu_t S
            a = 0.5 * (ux - vy); b = 0.5 * (uy + vx)          # (drops k*I and g2*T2 source terms;
            return nt, -2.0 * nt * a, -2.0 * nt * b, 2.0 * nt * a  #  te then cancels to 0 -> implicit only)
        if no_g2:                                             # k + nu_t model: drop only the g2 (S*Om-Om*S) term
            g2 = np.zeros_like(g2)
        tuu, tuv, tvv = closure_np.build_tau(k, nt, g2, ux, uy, vx, vy)  # realizable
        return nt, tuu, tuv, tvv

    def update_closure(relax=RELAX):
        ux, uy, vx, vy = grad_arrays()                           # (smoothed if --insmooth)
        if laminar:
            nut.x.array[:] = 0.0; te.x.array[:] = 0.0; return
        nt, tuu, tuv, tvv = closure_fields(ux, uy, vx, vy)
        a = 0.5 * (ux - vy); b = 0.5 * (uy + vx)
        # explicit source = damped tau + 2*nut*S(u_n) (implicit -2 nut S(u) cancels at convergence)
        te11 = tuu + 2 * nt * a; te12 = tuv + 2 * nt * b; te22 = tvv - 2 * nt * a
        nut.x.array[:] = (1 - relax) * nut.x.array + relax * nt   # under-relax (relax=1 -> fresh)
        arr = te.x.array.reshape(-1, 4)
        new = np.stack([te11, te12, te12, te22], 1)
        arr[:] = (1 - relax) * arr + relax * new
        filter_source()                                          # Helmholtz-smooth nut_use/te_use

    # previous-iterate pressure for the discontinuity-capturing residual
    Qp, _ = ME.sub(1).collapse()
    p_n = fem.Function(Qp)

    # --- forms ---
    (u, p) = ufl.TrialFunctions(ME); (w, q) = ufl.TestFunctions(ME)
    nu = fem.Constant(msh, PETSc.ScalarType(nu_phys)); gd = fem.Constant(msh, PETSc.ScalarType(0.2))
    nu_art = fem.Constant(msh, PETSc.ScalarType(0.05))   # startup continuation, ramped to nu_floor
    sym = lambda z: ufl.sym(ufl.grad(z))
    a_form = (ufl.inner(ufl.dot(ufl.grad(u), u_n), w)
              + (nu + nu_art) * ufl.inner(ufl.grad(u), ufl.grad(w))
              + 2.0 * nut_use * ufl.inner(sym(u), ufl.grad(w))
              - p * ufl.div(w) + q * ufl.div(u)
              + gd * ufl.div(u) * ufl.div(w)) * ufl.dx
    L_form = ufl.inner(te_use, ufl.grad(w)) * ufl.dx

    # Consistent SUPG stabilization (opt-in). The Galerkin P2/P1 solve is
    # convection-dominated off-body (cell-Peclet >> 1), producing dispersive
    # vorticity ripples that the closure faithfully turns into spurious stress.
    # SUPG adds streamline diffusion weighted by the STRONG momentum residual Rm
    # (-> 0 for the exact smooth field, so no over-diffusion bias), which removes
    # the oscillatory modes from the discrete fixed point and lets nu_art -> 0.
    # The DG0 closure source div(te_use) and the grad(nut_use) terms vanish in the
    # element interior, so Rm uses 2*nut_use*div(sym u) for the eddy part only.
    if supg:
        h = ufl.CellDiameter(msh)
        unorm = ufl.sqrt(ufl.dot(u_n, u_n) + 1e-8)
        nu_eff = nu + nu_art + nut_use
        tau_supg = 1.0 / ufl.sqrt((2.0 * unorm / h) ** 2 + 9.0 * (4.0 * nu_eff / h ** 2) ** 2)
        Rm = (ufl.dot(ufl.grad(u), u_n) + ufl.grad(p)
              - (nu + nu_art) * ufl.div(ufl.grad(u))
              - 2.0 * nut_use * ufl.div(sym(u)))
        a_form = a_form + tau_supg * ufl.inner(Rm, ufl.dot(ufl.grad(w), u_n)) * ufl.dx

    # Discontinuity-capturing (crosswind) stabilization (opt-in). SUPG only damps
    # STREAMLINE oscillations; the off-body striations radiate obliquely (crosswind),
    # so they need isotropic dissipation -- but applied selectively. nu_dc is gated
    # by the strong residual R_n(u_n,p_n): it is large only where the discrete
    # solution is unresolved/oscillating and -> 0 where it is smooth, so it removes
    # the crosswind ripples WITHOUT globally smearing the resolved shear layers.
    # Capped at the first-order upwind scale h*|u|/2 for robustness.
    if dc:
        h = ufl.CellDiameter(msh)
        unorm = ufl.sqrt(ufl.dot(u_n, u_n) + 1e-8)
        R_n = (ufl.dot(ufl.grad(u_n), u_n) + ufl.grad(p_n)
               - nu * ufl.div(ufl.grad(u_n))
               - 2.0 * nut_use * ufl.div(sym(u_n)) - ufl.div(te_use))
        gnorm = ufl.sqrt(ufl.inner(ufl.grad(u_n), ufl.grad(u_n)) + 1e-8)
        nu_dc = ufl.min_value(0.5 * h * ufl.sqrt(ufl.inner(R_n, R_n) + 1e-12) / gnorm,
                              0.5 * h * unorm)
        a_form = a_form + nu_dc * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx

    prob = LinearProblem(a_form, L_form, bcs=bcs,
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                        "pc_factor_mat_solver_type": "mumps"})

    monitor = NonlinearMonitor(ME, u_n, p_n, a_form, L_form, bcs, nu_art,
        update_closure, [nut, te, nut_use, te_use])
    initial_residuals, _ = monitor.evaluate()
    history = ConvergenceHistory(initial_residuals, residual_rtol, residual_atol, tol)
    print(f"  filter={filter_mode} Cf={filter_cf:g}; nonlinear discrete residual initial={initial_residuals}", flush=True)

    du_prev = None; rho_hist = []   # contraction ratio rho_k = dU_k / dU_{k-1}
    aa_X, aa_F = [], []             # Anderson-acceleration history (iterates, residuals)
    m_aa = 5
    for it in range(picard):
        nu_art.value = max(nu_floor, 0.05 * (1.0 - it / 50.0))   # ramp to a small floor, then hold
        update_closure()
        wh = prob.solve()
        unew = wh.sub(0).collapse()
        f = unew.x.array - u_n.x.array                          # fixed-point residual g(u_n)
        du = np.linalg.norm(f) / (np.linalg.norm(unew.x.array) + 1e-30)
        rho = (du / du_prev) if (du_prev and du_prev > 0) else np.nan
        rho_hist.append(rho); du_prev = du
        # Anderson acceleration converges the coupled closure-solver fixed point
        # (kills the rho_k~1 closure-jitter stall); else plain under-relaxation.
        if anderson:
            aa_X.append(u_n.x.array.copy()); aa_F.append(f.copy())
            if len(aa_F) > m_aa + 1:
                aa_X.pop(0); aa_F.pop(0)
            if len(aa_F) >= 2:
                dF = np.diff(np.array(aa_F), axis=0).T          # (N, k)
                dX = np.diff(np.array(aa_X), axis=0).T
                gamma, *_ = np.linalg.lstsq(dF, aa_F[-1], rcond=None)
                u_n.x.array[:] = aa_X[-1] + alpha * aa_F[-1] - (dX + alpha * dF) @ gamma
            else:
                u_n.x.array[:] = u_n.x.array + alpha * f
        else:
            u_n.x.array[:] = (1 - alpha) * u_n.x.array + alpha * unew.x.array
        pnew = wh.sub(1).collapse()
        p_n.x.array[:] = (1 - alpha) * p_n.x.array + alpha * pnew.x.array
        u_n.x.scatter_forward(); p_n.x.scatter_forward()
        residuals, div_rms = monitor.evaluate()
        converged = history.record(it, residuals, du, None, div_rms, float(nu_art.value))
        if it < 5 or it % 5 == 0 or converged:
            equations = " ".join(f"{key}={val:.3e}" for key, val in residuals.items())
            print(f"  picard {it:3d} | dU={du:.3e} | {equations} | div_rms={div_rms:.3e} "
                  f"| nu_art={float(nu_art.value):.3e}", flush=True)
        if converged:
            print(f"  converged: all equation residuals and updates passed at {it}", flush=True)
            break
        if not all(np.isfinite(v) for v in residuals.values()):
            print("  stopped: nonfinite governing-equation residual", flush=True)
            break
    if not history.converged:
        print(f"  NOT CONVERGED after {len(history.rows)} iterations; residuals={residuals}", flush=True)

    # --- sample velocity on common grid (upper half) + mirror ---
    xs = np.linspace(-3, 8, 360); ys = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(xs, ys)
    qpts = np.c_[XX.ravel(), np.abs(YY.ravel()), np.zeros(XX.size)]
    tree = geometry.bb_tree(msh, 2)
    cc = geometry.compute_collisions_points(tree, qpts)
    cells = geometry.compute_colliding_cells(msh, cc, qpts)
    cell_idx = []
    valid = np.zeros(qpts.shape[0], bool)
    for i in range(qpts.shape[0]):
        lc = cells.links(i)
        if len(lc) > 0:
            cell_idx.append(lc[0]); valid[i] = True
        else:
            cell_idx.append(0)
    ci = np.array(cell_idx, dtype=np.int32)
    sgn = np.sign(YY.ravel())   # y->-y mirror: u,p,uu,vv even ; v,uv odd

    def grid_eval(f, ncomp):
        out = np.asarray(f.eval(qpts, ci)).reshape(qpts.shape[0], ncomp)
        out[~valid] = np.nan
        return out

    uev = grid_eval(u_n, 2)
    U = uev[:, 0].reshape(XX.shape); Vv = (uev[:, 1] * sgn).reshape(XX.shape)

    # pressure from the final mixed solution (gauge fixed later, relative field)
    ph = p_n  # pressure belonging to the monitored coupled state
    P = grid_eval(ph, 1)[:, 0].reshape(XX.shape)

    # Reynolds stress from the final (damped) closure on the converged u_n (DG0 cells)
    ux, uy, vx, vy = grad_arrays()                               # (smoothed if --insmooth)
    if laminar:
        tuu = tuv = tvv = np.zeros_like(ux)
    else:
        _, tuu, tuv, tvv = closure_fields(ux, uy, vx, vy)
    # The mean flow (U,V,p) is left unfiltered for accuracy; the pointwise NN
    # stress is the noisy field, so it is Helmholtz-smoothed (out_filt) only for
    # output -- a monotone, realizability-preserving denoise of the visualization.
    ow = max(filt, out_filt)
    if ow > 0.0 and not laminar:
        Go = fem.functionspace(msh, ("Lagrange", 1, (2, 2)))
        tau_dg0 = fem.Function(Qt)
        tau_dg0.x.array.reshape(-1, 4)[:] = np.stack([tuu, tuv, tuv, tvv], 1)
        tau_f = fem.Function(Go)
        _oa, _ob = ufl.TrialFunction(Go), ufl.TestFunction(Go)
        ow2 = fem.Constant(msh, PETSc.ScalarType(ow ** 2))
        LinearProblem(ufl.inner(_oa, _ob) * ufl.dx + ow2 * ufl.inner(ufl.grad(_oa), ufl.grad(_ob)) * ufl.dx,
                      ufl.inner(tau_dg0, _ob) * ufl.dx, u=tau_f,
                      petsc_options={"ksp_type": "cg", "pc_type": "jacobi"}).solve()
        tg = grid_eval(tau_f, 4)
        TUU = tg[:, 0].reshape(XX.shape); TUV = (tg[:, 1] * sgn).reshape(XX.shape); TVV = tg[:, 3].reshape(XX.shape)
    else:
        fT = fem.Function(Q0)

        def grid_cell(arr):
            fT.x.array[:] = arr
            return grid_eval(fT, 1)[:, 0]
        TUU = grid_cell(tuu).reshape(XX.shape)
        TUV = (grid_cell(tuv) * sgn).reshape(XX.shape)
        TVV = grid_cell(tvv).reshape(XX.shape)

    out = output_dir or os.path.join(closure_np.config.RESULTS_DIR, "fem")
    os.makedirs(out, exist_ok=True)
    nutag = "" if abs(nu_phys - NU) < 1e-12 else f"_nu{nu_phys:g}"
    tag = f"{shape}_{'laminar' if laminar else ckpt}{nutag}{outtag}"   # outtag: ablation variants w/o clobbering baseline
    metadata = dict(shape=shape, checkpoint=ckpt, filter_mode=filter_mode, filter_cf=filter_cf,
        insmooth=insmooth, hwall=hwall, wake_h=wake_h, hfar=hfar,
        cells=int(msh.topology.index_map(2).size_global), alpha=alpha,
        source_filter=filt, output_filter=out_filt, supg=supg, dc=dc,
        nu=nu_phys, nu_floor=nu_floor, max_iterations=picard,
        residual_definition="free-row stabilized weak equations; mass-diagonal RMS; fresh closure; no startup viscosity")
    history.save(os.path.join(out, f"{tag}_convergence.json"), metadata)
    FILTER_WIDTH = grid_eval(gradient_filter.length, 1)[:, 0].reshape(XX.shape)
    np.savez(os.path.join(out, f"{tag}.npz"), X=XX, Y=YY, U=U, V=Vv,
             converged=history.converged, filter_width=FILTER_WIDTH, filter_cf=filter_cf, filter_mode=filter_mode,
             speed=np.sqrt(U**2 + Vv**2), p=P, uu=TUU, uv=TUV, vv=TVV)
    print(f"saved {out}/{tag}.npz  (valid {int(valid.sum())}/{qpts.shape[0]})")

    # --- conditioning diagnostics (opt-in) -------------------------------------
    # rho(J_G): spectral radius of the UNDAMPED one-step coupled map
    #   T0(u) = solve( closure_fresh(u) )   (alpha=1, RELAX=1, nu_art=0 at the end)
    # measured by matrix-free power iteration at the converged fixed point u*.
    # rho>1 => the raw Picard+closure map diverges and only relaxation holds it
    # together; the Lipschitz penalty is expected to lower rho. Also dumps the
    # leading mode and the closure gain map ||d(k,nu_t,g2)/d(grad u)||_F.
    if diag and not laminar:
        u_star = u_n.x.array.copy()
        eps = 1e-6 * (np.linalg.norm(u_star) + 1e-30)

        def T0(disp):
            # raw one-step image: solve( fresh closure( u_star + disp ) ), velocity dofs
            u_n.x.array[:] = u_star + disp
            update_closure(relax=1.0)            # fresh closure, no under-relaxation
            img = prob.solve().sub(0).collapse().x.array.copy()
            u_n.x.array[:] = u_star
            return img

        # centered directional derivative about u_star (which need NOT be a fixed
        # point: the solve stalls at the jitter floor, so subtract the un-perturbed
        # image g0 rather than u_star itself).
        g0 = T0(np.zeros_like(u_star))

        def Jdir(v):
            return (T0(eps * v) - g0) / eps

        rng = np.random.default_rng(0)
        v = rng.standard_normal(u_star.size); v /= (np.linalg.norm(v) + 1e-30)
        rho_jg = np.nan
        for _ in range(npow):
            jv = Jdir(v); nrm = np.linalg.norm(jv)
            rho_jg = nrm                          # v is unit-norm each step
            v = jv / (nrm + 1e-30)
        update_closure(relax=1.0)                 # leave closure at the converged state
        rho_arr = np.array(rho_hist, float)
        tail = rho_arr[~np.isnan(rho_arr)]
        rho_tail = float(np.median(tail[-20:])) if tail.size else np.nan
        print(f"  [diag] rho(J_G) power[{npow}]={rho_jg:.4f} | observed rho_k tail-median={rho_tail:.4f}")

        # closure gain map ||d(k,nu_t,g2)/d(grad u)||_F at centroids (finite diff)
        for gf, ge in zip(gspaces, gexpr):
            gf.interpolate(ge)
        gu = [g.x.array.copy() for g in gspaces]           # ux,uy,vx,vy
        h = 1e-4; gain2 = np.zeros(gu[0].shape)
        for j in range(4):
            pp = [c.copy() for c in gu]; pp[j] += h
            mm = [c.copy() for c in gu]; mm[j] -= h
            cp = np.stack(closure_np.coeffs(W, *pp, dvals), 1)   # (Ncell,3) k,nut,g2
            cm = np.stack(closure_np.coeffs(W, *mm, dvals), 1)
            gain2 += np.sum(((cp - cm) / (2 * h)) ** 2, axis=1)
        gain = np.sqrt(gain2) * phi                          # solver sees phi*coeffs

        gf_cell = fem.Function(Q0); gf_cell.x.array[:] = gain
        GAIN = grid_eval(gf_cell, 1)[:, 0].reshape(XX.shape)
        mode = fem.Function(V0); mode.x.array[:] = v
        mev = grid_eval(mode, 2)
        MODE = np.sqrt(mev[:, 0] ** 2 + mev[:, 1] ** 2).reshape(XX.shape)

        np.savez(os.path.join(out, f"{tag}_diag.npz"), X=XX, Y=YY,
                 rho_k=rho_arr, rho_jg=rho_jg, rho_tail=rho_tail,
                 gain=GAIN, mode=MODE, gain_max=float(np.nanmax(GAIN)),
                 gain_mean=float(np.nanmean(GAIN)))
        print(f"  [diag] saved {out}/{tag}_diag.npz  "
              f"gain max={np.nanmax(GAIN):.3e} mean={np.nanmean(GAIN):.3e}")

    return history.converged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="tri_long_0_dns")
    ap.add_argument("--ckpt", default="tb_tri")
    ap.add_argument("--laminar", action="store_true")
    ap.add_argument("--wall_d0", type=float, default=0.20, help="near-wall closure damping length")
    ap.add_argument("--nu_floor", type=float, default=0.0, help="residual continuation viscosity (stabilization); >0 for hard cases")
    ap.add_argument("--filt", type=float, default=0.0, help="Helmholtz filter width on the SOLVE source (0=off; tightens convergence but smears the mean flow)")
    ap.add_argument("--out_filt", type=float, default=0.03, help="Helmholtz filter width on the OUTPUT stress only (visualization denoise)")
    ap.add_argument("--diag", action="store_true", help="measure conditioning: rho(J_G) power iteration + gain map + leading mode")
    ap.add_argument("--npow", type=int, default=25, help="power-iteration steps for rho(J_G)")
    ap.add_argument("--supg", action=argparse.BooleanOptionalAction, default=True, help="SUPG streamline stabilization (default on; --no-supg to disable)")
    ap.add_argument("--dc", action="store_true", help="discontinuity-capturing (crosswind) stabilization; pair with --supg")
    ap.add_argument("--nu", type=float, default=NU, help="physical kinematic viscosity (default 1e-4 = Re 1e4; use 1e-3 for Re 1e3)")
    ap.add_argument("--hwall", type=float, default=0.02, help="mesh size at the wall")
    ap.add_argument("--hfar", type=float, default=0.8, help="mesh size in the far/mid field (lower = refine the striation zone)")
    ap.add_argument("--wake_h", type=float, default=0.08, help="mesh size in the wake box")
    ap.add_argument("--anderson", action="store_true", help="Anderson-accelerate the Picard fixed point (kills the closure-jitter stall)")
    ap.add_argument("--nut_max", type=float, default=0.0, help="bounded nu_t cap baked into the closure (match training)")
    ap.add_argument("--gate_w0", type=float, default=0.0, help="turbulence gate scale baked into the closure (match training)")
    ap.add_argument("--train_wall_d0", type=float, default=0.0, help="wall-damping baked into the closure during training (match training)")
    ap.add_argument("--evonly", action="store_true", help="pure eddy-viscosity closure: tau = -2 nu_t S (drop k*I and g2 source terms)")
    ap.add_argument("--no_g2", action="store_true", help="k + nu_t model: keep k*I and -2 nu_t S, drop the g2 (S*Om-Om*S) term")
    ap.add_argument("--insmooth", type=float, default=None, help="legacy cell-width multiplier; supplying it selects cell mode unless --filter_mode is given")
    ap.add_argument("--outtag", default="", help="suffix on the output npz filename (ablation variants; '' = baseline)")
    ap.add_argument("--filter_mode", choices=("strain", "cell", "none"), default=None,
                    help="input filter: physical strain length (default), legacy c*h, or disabled")
    ap.add_argument("--filter_cf", type=float, default=0.2, help="Cf in delta/D=Cf/(1+s*D/U_inf)")
    ap.add_argument("--picard", type=int, default=500, help="maximum nonlinear iterations")
    ap.add_argument("--tol", type=float, default=1e-4, help="secondary relative update tolerance for velocity and k")
    ap.add_argument("--residual_rtol", type=float, default=1e-6, help="residual tolerance relative to the fixed initial value of each equation block")
    ap.add_argument("--residual_atol", type=float, default=1e-8, help="absolute mass-diagonal-scaled residual tolerance and essential-BC tolerance")
    ap.add_argument("--output_dir", default=os.path.join(closure_np.config.RESULTS_DIR, "fem"), help="directory for the deployed fields and convergence histories")
    a = ap.parse_args()
    success = run(a.shape, a.ckpt, a.laminar, picard=a.picard, tol=a.tol,
        filter_mode=a.filter_mode, filter_cf=a.filter_cf, residual_rtol=a.residual_rtol,
        residual_atol=a.residual_atol, output_dir=a.output_dir, wall_d0=a.wall_d0, nu_floor=a.nu_floor, filt=a.filt, out_filt=a.out_filt,
        diag=a.diag, npow=a.npow, supg=a.supg, dc=a.dc, nu_phys=a.nu,
        hwall=a.hwall, hfar=a.hfar, wake_h=a.wake_h, anderson=a.anderson,
        nut_max=a.nut_max, gate_w0=a.gate_w0, train_wall_d0=a.train_wall_d0, evonly=a.evonly, no_g2=a.no_g2,
        insmooth=a.insmooth, outtag=a.outtag)

    sys.exit(0 if success else 2)  # never report an unconverged deployment as successful
