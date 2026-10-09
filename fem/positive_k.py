"""Monotone P1 k transport: graph diffusion, lumped destruction, positive load.

No SUPG source weighting and no post-solve clipping. Single MPI rank, matching
existing deployment. This is the low-order algebraic-diffusion scheme, not a
high-order flux-limited scheme. The nonlinear residual includes graph diffusion.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
import ufl


def graph_operator(K, reaction):
    """Residual convention A k=b: enforce A_ij<=0 for i!=j.

    d_ij=-max(0,K_ij,K_ji); d_ii=-sum_{j!=i}d_ij.
    Lump nonnegative reaction onto the diagonal. Transport row sums are zero
    by partition of unity; reconstructing the diagonal removes roundoff drift.
    """
    off=K.copy().tocsr(); off.setdiag(0); off.eliminate_zeros()
    c=off.maximum(off.T).maximum(0).tocsr(); c.eliminate_zeros()
    D=-c+sparse.diags(np.asarray(c.sum(axis=1)).ravel())
    lowoff=off-c
    low=lowoff+sparse.diags(-np.asarray(lowoff.sum(axis=1)).ravel()+reaction)
    return low.tocsr(),D.tocsr()


class PositiveK:
    def __init__(self, check, velocity, diffusivity, production, reaction):
        self.__dict__.update(check.__dict__)
        self.original_check=check
        self.solution=fem.Function(self.space)
        trial,test=ufl.TrialFunction(self.space),ufl.TestFunction(self.space)
        self.transport=fem.form((ufl.dot(ufl.grad(trial),velocity)*test+
                                 diffusivity*ufl.inner(ufl.grad(trial),ufl.grad(test)))*ufl.dx)
        self.reaction_form=fem.form(reaction*test*ufl.dx)
        self.load_form=fem.form(production*test*ufl.dx)
        self.ids=np.flatnonzero(self.free)
        if self.comm.size != 1:
            raise ValueError('PositiveK currently requires one MPI rank')

    def vector(self, form):
        v=assemble_vector(form)
        v.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE)
        a=v.array.copy(); v.destroy()
        return a

    def assemble(self):
        mat=assemble_matrix(self.transport); mat.assemble()
        ai,aj,av=mat.getValuesCSR()
        K=sparse.csr_matrix((av.copy(),aj.copy(),ai.copy()),shape=mat.getSize())
        mat.destroy()
        reaction=self.vector(self.reaction_form)
        b=self.vector(self.load_form)
        if reaction.min() < -1e-13 or b.min() < -1e-13:
            raise ValueError('Nonnegative reaction and production required')
        A,D=graph_operator(K,reaction)
        self.last_matrix,self.last_load,self.last_diffusion=A,b,D
        return A,b

    def raw_vector(self):
        A,b=self.assemble()
        return A@self.state.x.array-b

    def evaluate(self):
        r=self.raw_vector()
        prescribed=self.state.x.array.copy()
        for bc in self.bcs: bc.set(prescribed)
        return {'k':float(np.sqrt(np.sum(r[self.ids]**2/self.diagonal[self.ids])/self.volume)),
                'bc':float(np.max(np.abs(prescribed-self.state.x.array)))}

    def solve(self):
        A,b=self.assemble()
        prescribed=np.zeros(A.shape[0])
        for bc in self.bcs: bc.set(prescribed)
        fixed=np.flatnonzero(~self.free)
        rhs=b[self.ids]-A[self.ids][:,fixed]@prescribed[fixed]
        self.solution.x.array[:]=prescribed
        self.solution.x.array[self.ids]=splu(A[self.ids][:,self.ids].tocsc()).solve(rhs)
        self.solution.x.scatter_forward()
        if self.solution.x.array.min() < -1e-11:
            raise RuntimeError('Monotone linear k solve produced a significant negative value')
        return self.solution

    def factor(self):
        A,_=self.assemble()
        # Homogeneous correction boundary conditions, zero rows and columns.
        F=sparse.diags(self.free.astype(float))
        A=(F@A@F+sparse.diags((~self.free).astype(float))).tocsr()
        mat=PETSc.Mat().createAIJ(size=A.shape,csr=(A.indptr.astype(PETSc.IntType),
            A.indices.astype(PETSc.IntType),A.data),comm=self.comm)
        mat.assemble()
        ksp=PETSc.KSP().create(self.comm); ksp.setOperators(mat)
        ksp.setType('preonly'); ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType('mumps'); ksp.setUp()
        return mat,ksp,mat.createVecRight(),mat.createVecLeft()

    def diagnostics(self):
        A,b=self.assemble()
        off=A.copy(); off.setdiag(0); off.eliminate_zeros()
        D=self.last_diffusion
        k=self.state.x.array
        return dict(k_min=float(k.min()),k_max=float(k.max()),negative_dofs=int(np.sum(k < -1e-11)),
                    offdiagonal_max=float(off.data.max(initial=0)),
                    minimum_row_sum=float(np.asarray(A.sum(1)).min()),
                    minimum_production_load=float(b.min()),
                    graph_diffusion_energy=float(k@(D@k)),
                    graph_diffusion_diagonal_max=float(D.diagonal().max()),
                    k_residual=self.evaluate()['k'])
