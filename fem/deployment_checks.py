"""Mesh-independent input filtering and nonlinear discrete-equation checks.

Lengths and velocities are nondimensionalized by D and U_inf (both equal to 1).
Residuals are assembled from the deployed weak equations, including stabilization.
They are not strong-form PDE errors or estimates of discretization error.
"""
import json
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector


def resolve_filter_options(mode=None, cells=None):
    """Default to physical smoothing; explicit historical --insmooth selects c*h."""
    if mode is None:
        mode = "cell" if cells is not None else "strain"
    if mode != "cell" and cells is not None:
        raise ValueError("--insmooth applies only to cell mode; use --filter_mode none to disable")
    return mode, 1.0 if cells is None else cells


def strain_filter_length(ux, uy, vx, vy, coefficient=0.2):
    """delta/D = Cf/(1 + sqrt(2 S:S) D/U_inf), using the full symmetric gradient."""
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("filter coefficient must be finite and nonnegative")
    strain = np.sqrt(2.0 * (ux**2 + vy**2) + (uy + vx)**2)
    return coefficient / (1.0 + strain)


class InputGradientFilter:
    """Componentwise variable-coefficient Helmholtz filter, natural no-flux BCs.

    The strain sensor is evaluated from the current unfiltered centroid gradients.
    Its length is held fixed while the four components are filtered. No mesh floor,
    strain prefilter, or learned parameter is introduced. 'cell' reproduces c*h.
    """
    def __init__(self, mesh, dg0, velocity, mode="strain", coefficient=0.2, cells=1.0):
        if mode not in ("strain", "cell", "none"):
            raise ValueError("unknown input filter mode")
        if not all(np.isfinite(x) and x >= 0 for x in (coefficient, cells)):
            raise ValueError("filter parameters must be finite and nonnegative")
        self.mode, self.coefficient, self.cells = mode, coefficient, cells
        self.enabled = (mode == "strain" and coefficient > 0) or (mode == "cell" and cells > 0)
        self.raw = [fem.Function(dg0) for _ in range(4)]
        self.expressions = [fem.Expression(velocity[i // 2].dx(i % 2),
                            dg0.element.interpolation_points()) for i in range(4)]
        self.length = fem.Function(dg0)
        self.mesh = mesh
        if self.enabled:
            cg1 = fem.functionspace(mesh, ("Lagrange", 1))
            self.source = fem.Function(dg0)
            self.filtered = fem.Function(cg1)
            self.samples = [fem.Function(dg0) for _ in range(4)]
            self.sample_expr = fem.Expression(self.filtered, dg0.element.interpolation_points())
            if mode == "cell":
                self.length.interpolate(fem.Expression(cells * ufl.CellDiameter(mesh),
                                        dg0.element.interpolation_points()))
            trial, test = ufl.TrialFunction(cg1), ufl.TestFunction(cg1)
            a = (trial * test + self.length**2 * ufl.inner(ufl.grad(trial), ufl.grad(test))) * ufl.dx
            self.problem = LinearProblem(a, self.source * test * ufl.dx, u=self.filtered,
                petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-10,
                               "ksp_atol": 1e-12, "ksp_error_if_not_converged": True})

    def arrays(self):
        for field, expr in zip(self.raw, self.expressions):
            field.interpolate(expr)
            field.x.scatter_forward()
        values = [g.x.array for g in self.raw]
        if not self.enabled:
            self.length.x.array[:] = 0
            return values
        if self.mode == "strain":
            self.length.x.array[:] = strain_filter_length(*values, self.coefficient)
            self.length.x.scatter_forward()
        for src, dst in zip(self.raw, self.samples):
            self.source.x.array[:] = src.x.array
            self.problem.solve()
            dst.interpolate(self.sample_expr)
            dst.x.scatter_forward()
        return [g.x.array for g in self.samples]


class WeakResidual:
    """Free-row weak residual, scaled by the diagonal of the consistent mass matrix.

    norm_b = sqrt(sum_{i in free block b} R_i^2/M_ii / volume).
    This avoids an unscaled coefficient-vector norm and does not discard the
    interelement contributions of DG0 stresses. Essential-row reactions are
    excluded; essential boundary errors are checked separately.
    """
    def __init__(self, space, state, residual, mass, bcs, blocks):
        self.space, self.state, self.bcs = space, state, bcs
        self.comm = space.mesh.comm
        self.form = fem.form(residual)
        self.nowned = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
        self.blocks = {key: np.asarray(idx)[np.asarray(idx) < self.nowned]
                       for key, idx in blocks.items()}
        mat = assemble_matrix(fem.form(mass)); mat.assemble()
        diagonal = mat.getDiagonal()
        self.diagonal = diagonal.array.copy()
        diagonal.destroy(); mat.destroy()
        self.free = np.ones(self.nowned, dtype=bool)
        for bc in bcs:
            ids, n = bc.dof_indices()
            self.free[ids[:n]] = False
        vol = fem.assemble_scalar(fem.form(fem.Constant(space.mesh, PETSc.ScalarType(1)) * ufl.dx))
        self.volume = self.comm.allreduce(vol, op=MPI.SUM)
        if np.any(self.diagonal <= 0):
            raise ValueError("Residual mass scaling must have positive diagonal")

    def evaluate(self):
        self.state.x.scatter_forward()
        vec = assemble_vector(self.form)
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        r = vec.array.copy(); vec.destroy()
        values = {}
        for name, ids in self.blocks.items():
            ids = ids[self.free[ids]]
            square = np.sum(np.abs(r[ids])**2 / self.diagonal[ids])
            values[name] = float(np.sqrt(self.comm.allreduce(square, op=MPI.SUM) / self.volume))
        prescribed = self.state.x.array.copy()
        for bc in self.bcs:
            bc.set(prescribed)
        error = np.abs((prescribed - self.state.x.array)[:self.nowned])
        values["bc"] = float(self.comm.allreduce(float(error.max(initial=0)), op=MPI.MAX))
        return values


class NonlinearMonitor:
    """Evaluate the coupled state with a fresh closure, then restore solver coefficients."""
    def __init__(self, mixed_space, velocity, pressure, a, L, bcs, artificial_viscosity,
                 update_closure, coefficient_fields, k=None, ak=None, Lk=None, bcs_k=None):
        self.velocity, self.pressure, self.k = velocity, pressure, k
        self.state = fem.Function(mixed_space)
        _, self.umap = mixed_space.sub(0).collapse()
        _, self.pmap = mixed_space.sub(1).collapse()
        self.update_closure = update_closure
        self.fields = list({id(f): f for f in coefficient_fields}.values())
        self.comm = mixed_space.mesh.comm
        u, p = ufl.TrialFunctions(mixed_space)
        w, q = ufl.TestFunctions(mixed_space)
        residual = ufl.action(a, self.state) - L
        residual = ufl.replace(residual, {artificial_viscosity: ufl.as_ufl(0.0)})
        self.momentum = WeakResidual(mixed_space, self.state, residual,
            (ufl.inner(u, w) + p*q) * ufl.dx, bcs,
            {"momentum_x": mixed_space.sub(0).sub(0).collapse()[1], "momentum_y": mixed_space.sub(0).sub(1).collapse()[1], "continuity": self.pmap})
        self.kcheck = None
        if k is not None:
            space = k.function_space
            kt, wk = ufl.TrialFunction(space), ufl.TestFunction(space)
            n = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
            self.kcheck = WeakResidual(space, k, ufl.action(ak, k) - Lk,
                kt*wk*ufl.dx, bcs_k, {"k": np.arange(n)})
        self.div_form = fem.form(ufl.div(velocity)**2 * ufl.dx)
        # Begin on the essential-boundary constraint manifold; relaxation then preserves it.
        self._pack()
        for bc in bcs:
            bc.set(self.state.x.array)
        self.state.x.scatter_forward()
        velocity.x.array[:] = self.state.x.array[self.umap]
        pressure.x.array[:] = self.state.x.array[self.pmap]
        if k is not None:
            for bc in bcs_k:
                bc.set(k.x.array)
            k.x.scatter_forward()

    def _pack(self):
        self.velocity.x.scatter_forward(); self.pressure.x.scatter_forward()
        self.state.x.array[self.umap] = self.velocity.x.array
        self.state.x.array[self.pmap] = self.pressure.x.array

    def evaluate(self):
        self._pack()
        if self.k is not None:
            self.k.x.scatter_forward()
        old = [f.x.array.copy() for f in self.fields]
        try:
            # Never evaluate only the last frozen linear Picard subproblem.
            self.update_closure(relax=1.0)
            for field in self.fields:
                field.x.scatter_forward()
            values = self.momentum.evaluate()
            if self.kcheck is not None:
                kr = self.kcheck.evaluate()
                values["k"] = kr["k"]
                values["bc"] = max(values["bc"], kr["bc"])
            div2 = self.comm.allreduce(fem.assemble_scalar(self.div_form), op=MPI.SUM)
            divergence_rms = float(np.sqrt(max(div2, 0.0) / self.momentum.volume))
            return values, divergence_rms
        finally:
            for field, arr in zip(self.fields, old):
                field.x.array[:] = arr
                field.x.scatter_forward()


class ConvergenceHistory:
    """Fixed initial references; every equation block must pass atol+rtol*R0.

    A small velocity update alone never satisfies this criterion. Pointwise
    divergence RMS is diagnostic, since weak incompressibility does not make a
    Taylor-Hood velocity pointwise divergence-free on a finite mesh.
    """
    def __init__(self, initial, rtol=1e-6, atol=1e-8, update_tol=1e-4):
        if not all(np.isfinite(x) and x > 0 for x in (rtol, atol, update_tol)):
            raise ValueError("convergence tolerances must be finite and positive")
        if not all(np.isfinite(x) for x in initial.values()):
            raise ValueError("nonfinite initial residual")
        self.initial = dict(initial)
        self.rtol, self.atol, self.update_tol = rtol, atol, update_tol
        self.rows = []
        self.converged = False

    def record(self, iteration, residuals, du, dk, divergence_rms, nu_art):
        row = dict(iteration=int(iteration), **residuals, du=float(du),
                   divergence_rms=float(divergence_rms), nu_art=float(nu_art))
        if dk is not None:
            row["dk"] = float(dk)
        ok = True
        for name, value in residuals.items():
            threshold = self.atol if name == "bc" else self.rtol * self.initial[name]
            row[name + "_threshold"] = threshold
            row[name + "_ratio"] = float(value / threshold)
            ok &= np.isfinite(value) and value <= threshold
        ok &= np.isfinite(du) and du <= self.update_tol
        if dk is not None:
            ok &= np.isfinite(dk) and dk <= self.update_tol
        ok &= np.isfinite(divergence_rms) and np.isfinite(nu_art) and abs(nu_art) < 1e-15
        self.converged = bool(ok)
        row["converged"] = self.converged
        self.rows.append(row)
        return self.converged

    def save(self, path, settings):
        # JSON null represents nonfinite diagnostic values; never emit invalid JSON NaN.
        def clean(x):
            if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list): return [clean(v) for v in x]
            if isinstance(x, float) and not np.isfinite(x): return None
            return x
        with open(path, "w") as f:
            json.dump(clean(dict(settings=settings, initial_residuals=self.initial,
                residual_rtol=self.rtol, residual_atol=self.atol, update_tol=self.update_tol,
                equation_threshold="rtol * initial; atol applies only to boundary error", converged=self.converged, iterations=len(self.rows), history=self.rows)),
                f, indent=2, allow_nan=False)
            f.write("\n")
