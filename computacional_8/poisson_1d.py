"""Resolve -u'' = pi² sin(pi x) em (0, 1), com u(0) = u(1) = 0.

Execute em um ambiente com Firedrake: python poisson_1d.py
A solução exata é u(x) = sin(pi x).
"""

from firedrake import (
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    UnitIntervalMesh,
    dx,
    errornorm,
    grad,
    inner,
    pi,
    sin,
    solve,
)
from firedrake.petsc import PETSc


def main():
    # Malha uniforme com 40 elementos no intervalo [0, 1].
    mesh = UnitIntervalMesh(40)
    V = FunctionSpace(mesh, "CG", 1)  # Elementos contínuos lineares (P1).

    x = SpatialCoordinate(mesh)[0]
    u_exata = sin(pi * x)
    f = pi**2 * sin(pi * x)

    u = TrialFunction(V)
    v = TestFunction(V)

    # Forma fraca: integral(u' v') = integral(f v).
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    uh = Function(V, name="u_h")
    solve(
        a == L,
        uh,
        bcs=bc,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )

    PETSc.Sys.Print(f"Erro L2: {errornorm(u_exata, uh, norm_type='L2'):.6e}")
    PETSc.Sys.Print(f"Erro H1: {errornorm(u_exata, uh, norm_type='H1'):.6e}")


if __name__ == "__main__":
    main()
