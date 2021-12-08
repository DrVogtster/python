from dolfin import *
from dolfin_adjoint import *
import moola
import pylab as plt


def main_func(ncells):
    print("hi")
    mesh = IntervalMesh(ncells, 0, 2*pi)

    Welm = MixedElement([FiniteElement('Lagrange', interval, 5),
                         FiniteElement('Lagrange', interval, 5)])
    W = FunctionSpace(mesh, Welm)

    bcsys = [DirichletBC(W.sub(0), Constant(0.0), 'near(x[0], 0)'),
             DirichletBC(W.sub(1), Constant(1.0), 'near(x[0], 0)')]

    up = Function(W)
    u, p = split(up)
    v, q = split(TestFunction(W))
    source = Expression('2.0*sin(x[0])', degree=2)

    weak_form = u.dx(0)*v*dx - p*v*dx + p.dx(0)*q*dx - u*q*dx  + source*q*dx
    Jac = derivative(weak_form, up, TrialFunction(W))

    solve(weak_form == 0, up, J=Jac, bcs=bcsys)

    u, p = split(up)
        
    x, = SpatialCoordinate(mesh)

    print('u error' +str(sqrt(abs(assemble(inner(u - sin(x), u - sin(x))*dx)))))
    print('p error' + str(sqrt(abs(assemble(inner(p - cos(x), p - cos(x))*dx)))))


def main_func_opt(ncells):
    print("hi")
    mesh = IntervalMesh(ncells, 0, 2*pi)

    Welm = MixedElement([FiniteElement('Lagrange', interval, 5),
                         FiniteElement('Lagrange', interval, 5)])
    W = FunctionSpace(mesh, Welm)

    bcsys = [DirichletBC(W.sub(0), Constant(0.0), 'near(x[0], 0)'),
             DirichletBC(W.sub(1), Constant(1.0), 'near(x[0], 0)')]
    W_c = FunctionSpace(mesh, "DG", 0)
    up = Function(W)
    u, p = split(up)
    v, q = split(TestFunction(W))
    source = Expression('2.0*sin(x[0])', degree=2)
    f = interpolate(Expression("2.0*cos(x[0])", name='Control', degree=1), W_c)
    weak_form = u.dx(0)*v*dx - p*v*dx + p.dx(0)*q*dx - u*q*dx  + f*q*dx
    Jac = derivative(weak_form, up, TrialFunction(W))

    solve(weak_form == 0, up, J=Jac, bcs=bcsys)

    J = assemble((0.5 * inner(source - f, source - f)) * dx)
    control = Control(f)
    u, p = split(up)
    rf = ReducedFunctional(J, control)
    problem = MoolaOptimizationProblem(rf)
    f_moola = moola.DolfinPrimalVector(f)
    solver = moola.BFGS(problem, f_moola, options={'jtol': 0,
                                               'gtol': 1e-9,
                                               'Hinit': "default",
                                               'maxiter': 100,
                                               'mem_lim': 10})
    sol = solver.solve()
    f_opt = sol['control'].data
    x, = SpatialCoordinate(mesh)
    f.assign(f_opt)
    plot(f)
    plt.savefig("sol.pdf")
    print('u error' +str(sqrt(abs(assemble(inner(u - sin(x), u - sin(x))*dx)))))
    print('p error' + str(sqrt(abs(assemble(inner(p - cos(x), p - cos(x))*dx)))))
    #print("control error" + assemble((0.5 * inner(source - f_opt, source - f_opt)) * dx))



main_func_opt(10000)


#map(main_func, [10**i for i in range(2, 5)])
