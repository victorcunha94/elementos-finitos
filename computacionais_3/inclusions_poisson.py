import firedrake as fd
from firedrake.output import VTKFile 
from firedrake import interpolate 
import matplotlib.pyplot as plt

import gmsh
import numpy as np

# User defined data
bottom_wall = 0
right_wall  = 1
top_wall    = 2
left_wall   = 3

inclusion_marker = 3
background_marker = 2

ninclusions = 3
Lx = 2.0
Ly = 2.0
R1 = 0.25
R2 = 0.15
R3 = 0.25


hvet  = []

medias_T0 = []
medias_T1 = []
medias_T2 = []


h_mesh = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]

for h in h_mesh:
    hvet.append(h)
    #--------------------------------------------------------------------
    #--- Preprocess: Mesh generation, boundary and region identification

    def GenerateMesh():

        gmsh.initialize()

        mass1 = np.pi*R1**2
        mass2 = np.pi*R2**2
        mass3 = np.pi*R3**2
        mass_inc = mass1 + mass2 + mass3

        proc = 0
        if proc == 0:
            # We create one rectangle and the circular inclusion
            background = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
            inclusion1 = gmsh.model.occ.addDisk(0.5, 1.0, 0, R1, R1)
            inclusion2 = gmsh.model.occ.addDisk(1.0, 1.5, 0, R2, R2)
            inclusion3 = gmsh.model.occ.addDisk(1.5, 1.0, 0, R3, R3)
            gmsh.model.occ.synchronize()
            all_inclusions = [(2, inclusion1)]
            all_inclusions.extend([(2, inclusion2)])
            all_inclusions.extend([(2, inclusion3)])
            whole_domain = gmsh.model.occ.fragment([(2, background)], all_inclusions)
            gmsh.model.occ.synchronize()

            background_surfaces = []
            other_surfaces = []
            for domain in whole_domain[0]:
                com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
                mass = gmsh.model.occ.getMass(domain[0], domain[1])
                #print(mass, com)
                # Identify the square by its mass
                if np.isclose(mass, (Lx*Ly - mass_inc)):
                    gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=background_marker)
                    background_surfaces.append(domain)
                # Identify the inner circle by its center of mass
                elif np.isclose(np.linalg.norm(com), np.sqrt((0.5)**2 + (1.0)**2)):
                    gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker)
                    other_surfaces.append(domain)
                elif np.isclose(np.linalg.norm(com), np.sqrt((1.0)**2 + (1.5)**2)) and com[1] > 1.0:
                    gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+1)
                    other_surfaces.append(domain)
                elif np.isclose(np.linalg.norm(com), np.sqrt((1.5)**2 + (1.0)**2)):
                    gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+2)
                    other_surfaces.append(domain)
        
            # Tag the left and right boundaries
            left = []
            right = []
            for line in gmsh.model.getEntities(dim=1):
                com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
                if np.isclose(com[0], 0.0):
                    #print('L', line)
                    left.append(line[1])
                if np.isclose(com[0], Lx):
                    #print('R', line)
                    right.append(line[1])
            gmsh.model.addPhysicalGroup(1, left, left_wall)
            gmsh.model.addPhysicalGroup(1, right, right_wall)

        if(False):
            gmsh.model.mesh.field.add("Distance", 1)
            edges = gmsh.model.getBoundary(other_surfaces, oriented=False)
            gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            r = 0.04
            gmsh.model.mesh.field.setNumber(2, "LcMin", r / 2)
            gmsh.model.mesh.field.setNumber(2, "LcMax", 2 * r)
            gmsh.model.mesh.field.setNumber(2, "DistMin", 1 * r)
            gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * r)
            gmsh.model.mesh.field.setAsBackgroundMesh(2)
            # Generate mesh
            gmsh.option.setNumber("Mesh.Algorithm", 2)
            gmsh.model.mesh.generate(2) 
        else:
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
            gmsh.model.mesh.generate(2)
            
        gmsh.write("mesh.msh")
        gmsh.finalize()

        mesh = fd.Mesh("mesh.msh")

        return mesh
    #--------------------------------------------------------------------

    mesh = GenerateMesh()

    #Espaço de funções para a difusividade thermal 
    Qh = fd.FunctionSpace(mesh, "DG", degree=0) 

    # Problema data
    muA = 5.0
    muB = 500.0
    f   = 100.0
    uleft  = 0.0
    uright = 0.0
    exact_problem = False # True if muA=muB and f = 0


    print(f"muA = {muA} muB = {muB}")
    print(f"uleft = {uleft} uright = {uright}")
    print(f"f = {f}")

    '''
    The following lines define a piecewise constant function, being muA on the background and muB on the inclusions
    It is a projection of a piecewise constant function onto the space of elementwise constant functions.
    This is done for plotting purposes and also to use later on in the variational formulation of Poisson's problem
    Projections will be discussed later on in the course
    '''

    mu = fd.TrialFunction(Qh)
    vp = fd.TestFunction(Qh)
    ap = mu*vp*fd.dx
    Lp = fd.Constant(muA)*vp*fd.dx(background_marker)
    for k in range(ninclusions): # Add the contribuition of each inclusion
        Lp += fd.Constant(muB)*vp*fd.dx(inclusion_marker+k)

    muh = fd.Function(Qh, name='mu')
    opts={"ksp_type": "preonly", "pc_type": "lu"}
    fd.solve(ap==Lp, muh, bcs=[], solver_parameters=opts)
    VTKFile("mu.pvd").write(muh)

    # Poisson problem

    Vh = fd.FunctionSpace(mesh, "CG", degree=1)

    # Boundary conditions - Set Dirichlet values
    bcs = [fd.DirichletBC(Vh, fd.Constant(uleft), left_wall), 
           fd.DirichletBC(Vh, fd.Constant(uright), right_wall)]

    # Variational formulation
    u, v = fd.TrialFunction(Vh), fd.TestFunction(Vh)

    a = fd.inner(muh*fd.grad(u), fd.grad(v)) * fd.dx
    x = fd.SpatialCoordinate(mesh)
    source = fd.assemble(interpolate(fd.Constant(f), Qh)) # We interpolate the constant function onto the space of elementwise constants
    L = source * v * fd.dx


    # Solve the problem
    uh = fd.Function(Vh, name='uh')

    opts={"ksp_type": "preonly", "pc_type": "lu"}
    fd.solve(a==L, uh, bcs=bcs, solver_parameters=opts)
    VTKFile("uh.pvd").write(uh)


    if(exact_problem):
        print('Error norms since analytical solution is available (muA=muB and f=0)')
        #ue = uleft - (uleft - uright)*x[0]/Lx
        ue = 50*x[0]*(Lx - x[0])
        gradue  = fd.as_vector([-(uleft - uright)/Lx, 0.0])
        error_L2 = fd.assemble(((uh - ue)**2) * fd.dx)
        errorH1 = fd.assemble((((uh - ue)**2) + fd.dot(fd.grad(uh - ue),fd.grad(uh - ue))) * fd.dx)

        uexata   = fd.Function(Vh, name='uexata')
        uexata.interpolate(ue)

        erro = fd.Function(Vh, name='erro')
        erro.interpolate((uh - uexata))

        print("    |-L2error=", np.sqrt(error_L2))
        print("    |-H1error=", np.sqrt(errorH1))

        VTKFile("erro.pvd").write(erro)
        VTKFile("uexata.pvd").write(uexata)

    #VTKFile("uh.pvd").write(uh)
    VTKFile(f"uh_h_{h:.5f}.pvd").write(uh)

    print('Energy balance')
    n = fd.FacetNormal(mesh)


    #Qleft mede o calor entrando pela esquerda
    Qleft  = fd.assemble((fd.inner(muh*fd.grad(uh),n) * fd.ds(left_wall)))
    #Qrigth mede o calor saindo pela direita
    Qright = -fd.assemble((fd.inner(muh*fd.grad(uh),n) * fd.ds(right_wall)))
    gen    = fd.assemble(source * fd.dx)
    print('Qin=', Qleft, '\nQout=', Qright, '\nPower=', gen)

    saida_total = -Qleft + Qright

    print("Saída total =", saida_total)
    print("Resíduo do balanço =", saida_total - gen)

    if f == 0.0 and uleft != uright
        # Compute effective mu
        Q  = abs(Qright)
        u_dif = abs(uleft - uright)

        mu_eff = (Q/Ly) / (u_dif/Lx)
        print(f" mu_eff = {mu_eff}")
    


    print('Mean temperature at inclusions:')
    one = fd.assemble(interpolate(fd.Constant(1.0), Qh))
    Tincav = []
    for k in range(ninclusions):
        area = fd.assemble((one * fd.dx(inclusion_marker+k)))
        Tinc = fd.assemble(uh * fd.dx(inclusion_marker+k))

        Tincav.append(Tinc/area)

        print(f'T_%d= %f' %(k, Tincav[k]))

    medias_T0.append(Tincav[0])
    medias_T1.append(Tincav[1])
    medias_T2.append(Tincav[2])

# A última execução corresponde à malha mais fina.
T0_ref = medias_T0[-1]
T1_ref = medias_T1[-1]
T2_ref = medias_T2[-1]

erro_T0 = []
erro_T1 = []
erro_T2 = []

# Excluímos a própria malha de referência.
for j in range(len(hvet) - 1):
    erro_T0.append(abs(medias_T0[j] - T0_ref))
    erro_T1.append(abs(medias_T1[j] - T1_ref))
    erro_T2.append(abs(medias_T2[j] - T2_ref))

    print(
        f"h = {hvet[j]:.5f}: "
        f"E0 = {erro_T0[j]:.6e}, "
        f"E1 = {erro_T1[j]:.6e}, "
        f"E2 = {erro_T2[j]:.6e}"
)


h_plot = hvet[:-1]

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


