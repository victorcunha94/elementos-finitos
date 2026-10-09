import firedrake as fd
from firedrake.output import VTKFile
from firedrake import sin, pi, cos


N_vetor = [32, 64, 128, 256, 512]

for N in N_vetor:
    # Finite element mesh
    Nx, Ny = N, N
    Lx, Ly = 1.0, 1.0
    msh = fd.RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=False)

    # Space of functions
    Vd = fd.FunctionSpace(msh, "CG", degree=1)

    # Test and Trial functions
    u  = fd.TrialFunction(Vd)
    v  = fd.TestFunction(Vd)


    ## Solução Exata
    ue     = sin(2 * pi * x[0]) * sin(2 * pi * x[1])
    mue = 1 + cos(2 * pi * x[0])*(2 * pi * x[1])
    u_boundary = sin(2 * pi * x[0]) * sin(2 * pi * x[1])

    muexata = 1 + cos(2 * pi * x[0])*(2 * pi * x[1])

    # Boundary conditions
    bc = fd.DirichletBC(Vd, u_boundary, "on_boundary")

    # Source term
    f = -fd.div(mu(x)*fd.grad(uex(x)))

    x       = fd.SpatialCoordinate(msh)
    
    
    # Bilinear form (lhs) and linear form (rhs)
    a  = fd.inner(mu * fd.grad(u), fd.grad(v)) * fd.dx
    L  = fd.inner(f, v) * fd.dx

    # Solve the problem
    uh = fd.Function(Vd)
    mu = fd.Function(Vd)

    opts={"ksp_type": "preonly", "pc_type": "lu"}
    fd.solve(a==L, uh, bcs=[bc], solver_parameters=opts)


    error_L2 = fd.assemble(((uh - ue)**2) * fd.dx)
    errorH1 = fd.assemble((((uh - ue)**2) + fd.dot(fd.grad(uh - ue),fd.grad(uh - ue))) * fd.dx)

    uexata   = fd.Function(Vh, name='uexata')
    uexata.interpolate(ue)

    erro = fd.Function(Vh, name='erro')
    erro.interpolate((uh - uexata))

    print("    |-L2error=", np.sqrt(error_L2))
    print("    |-H1error=", np.sqrt(errorH1))


    def SolvePoisson(N=16, degree=1, flag_quads):
        msh = fd.RectangleMesh(N, N, 1.0, 1.0,
        quadrilateral=flag_quads)
        V = fd.FunctionSpace(msh, "CG", degree)
        .
        .
        .
        eh = uh - uex(x)
        
        .
        .
        return EL2, EH1, EInt


    # Visualize in paraview
    ud.rename = "mySolution"
    VTKFile("SolPoisson.pvd").write(ud)

plt.figure()

plt.loglog(h_plot, erro_T0, 'o-', label='Inclusão 0')
plt.loglog(h_plot, erro_T1, 's-', label='Inclusão 1')
plt.loglog(h_plot, erro_T2, 'd-', label='Inclusão 2')

plt.xlabel('Tamanho de malha solicitado ao Gmsh')
plt.ylabel('Erro absoluto da temperatura média')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig('converg.png', dpi=300)
plt.show()


