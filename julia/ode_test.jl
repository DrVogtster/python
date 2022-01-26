#=
ode_test:
- Julia version: 
- Author: mathb
- Date: 2022-01-25
=#



using Juniper
using Ipopt
using JuMP
using LinearAlgebra # for the dot product
using Base.Threads
using Cbc
using DifferentialEquations
using Trapz
#using Alpine

function test(x)
f(u,p,t) = u*(1-p)
u0 = 1
tspan = (0.0,1.0)
p=x
prob = ODEProblem(f,u0,tspan,p)
sol = DifferentialEquations.solve(prob)
u_sol = sol.u
t = sol.t

exp_t = exp.(t)
println(exp_t)
I=trapz(t,(u_sol - exp_t).^2)

return I



end


optimizer = Juniper.Optimizer
nl_solver= optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
mip_solver = optimizer_with_attributes(Cbc.Optimizer, "logLevel" => 0, "threads"=>nthreads())
m = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver, "mip_solver"=>mip_solver))



v = [10,20,12,23,42]
w = [12,45,12,22,21]
#@variable(m, x[1:5], Bin)


# model = Model()
@variable(m, x,Bin)

@NLobjective(m, Min, test(x))
# f(x...) = test(w,x)
# #register(m, :f, 5, f; autodiff = true)
# register(m, Symbol("g1"), 5, f; autodiff = true)
# register(m, Symbol("g2"), 5, f; autodiff = true)
# #@NLobjective(model, Min, f(x...))
# t(x) = test(w,x)
#
# #@NLconstraint(m, sum(w[i]*x[i]^2 for i=1:5) <=45)
# @NLconstraint(m, g1(x...) <=45)
# @NLconstraint(m, g2(x...) <=20)

optimize!(m)

# retrieve the objective value, corresponding x values and the status
println(JuMP.value.(x))
println(JuMP.objective_value(m))
println(JuMP.termination_status(m))
