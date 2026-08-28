import os
os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd
from firedrake.output import VTKFile
import gmsh
import matplotlib.pyplot as plt

# Finite element mesh
Nx, Ny = 16, 16
Lx, Ly = 1.0, 1.0
#msh = fd.RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
msh = fd.Mesh('mesh.msh')

# Space of functions
Vd = fd.FunctionSpace(msh, "CG", degree=1)

Vstress = fd.FunctionSpace(msh, "DG", degree=0)

# Test and Trial functions
u  = fd.TrialFunction(Vd)
v  = fd.TestFunction(Vd)

# Boundary conditions
u_boundary = fd.Constant(0.0)
bc = fd.DirichletBC(Vd, u_boundary, "on_boundary")
# Exemplo de Dirichlet zero nas paredes e valor 1 no obstáculo
bc_paredes = fd.DirichletBC(Vd, fd.Constant(0.0), 10)
bc_obstaculo = fd.DirichletBC(Vd, fd.Constant(1.0), 11)

# Junta as condições de contorno para o solver
bcs = [bc_paredes, bc_obstaculo]

# Source term
f  = fd.Constant(1.0)

x  = fd.SpatialCoordinate(msh) #acess the physical spatial coordinates from the mesh
#mu = fd.Constant(1.0) # this could be a function of x
mu = 0.01 + fd.exp(-100 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

# Bilinear form (lhs) and linear form (rhs)
a  = fd.inner(mu * fd.grad(u), fd.grad(v)) * fd.dx
L  = fd.inner(f, v) * fd.dx

# Solve the problem
ud = fd.Function(Vd)
opts={"ksp_type": "preonly", "pc_type": "lu"}
fd.solve(a==L, ud, bcs=[bc], solver_parameters=opts)

# Shear Stress and Boundary Integration
tau = mu * fd.grad(ud)
n   = fd.FacetNormal(msh)
Wss = fd.inner(tau, n) * fd.ds
print(fd.assemble(Wss))


# Visualize in paraview
ud.rename = "mySolution"
VTKFile("SolPoisson.pvd").write(ud)


