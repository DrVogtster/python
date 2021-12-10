from dolfin import *
from dolfin_adjoint import *
import moola
n = 10
mesh = RectangleMesh(Point(-1,-1),Point(1,1), n, n)

V = FunctionSpace(mesh, "CG", 1)
u, d = Function(V), Function(V)
u_v, d_v = TestFunction(V), TestFunction(V)
S0 = Constant(1)

bc = DirichletBC(V, 1, "on_boundary")

u_v, u_d = project(u, V, bcs=bc), project(d, V, bcs=bc)
J = assemble((inner(grad(u_v), grad(u_v)) +inner(grad(d), grad(d))\
        - u_v*S0)*dx)
control = [Control(u), Control(d)]
J_hat = ReducedFunctional(J, control)
#m_opt = minimize(J_hat, method = "L-BFGS-B", options = {"gtol": 1e-9})
J_hat(m_opt)

problem = MoolaOptimizationProblem(J_hat)

u_moola = moola.DolfinPrimalVector(u)
d_moola = moola.DolfinPrimalVector(d)
m_moola = moola.DolfinPrimalVectorSet([u_moola, d_moola])
solver = moola.BFGS(problem, m_moola, options={'jtol': 0,
                                                'gtol': 1e-9,
                                                'Hinit': "default",
                                                'maxiter': 100,
                                                'mem_lim': 10})

sol = solver.solve()
m_opt = sol['control'].data
