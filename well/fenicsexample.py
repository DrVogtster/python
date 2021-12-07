from dolfin import *


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

for i in range(2,5):
    main_func(10**i)


#map(main_func, [10**i for i in range(2, 5)])
