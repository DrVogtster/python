#=
ode_test:
- Julia version: 
- Author: mathb
- Date: 2022-01-25
=#

using Cubature

using Juniper
using Ipopt
using JuMP
using LinearAlgebra # for the dot product
using Base.Threads
using Cbc
using DifferentialEquations
using Trapz
#using Alpine


function diff_solver(in,f,tspan,u0,a,b,x::T) where {T <: Real}
    out = Vector{T}(undef, length(in))
    #out = zeros(length(in))
    for i=1:length(out)
        r = in[i]
        g =(u,p,t)-> f(u,p,t,r)
        prob = ODEProblem(g,u0,tspan,x)
        sol = DifferentialEquations.solve(prob)
        u_sol = sol.u
        t = sol.t
        out[i] = u_sol[end]
        
    end
    #println(out)
    return out

end

function test(x)
f(u,p,t,r) = u +r -p
u0 = 1

tspan = (0.0,1.0)
p=x
#prob = ODEProblem(f,u0,tspan,p)
a=-1.0
b=1.0
in = collect(LinRange(a,b,1000))

#println(in)
u_sol=diff_solver(in,f,tspan,u0,a,b,x)

exp_t = zeros(length(u_sol))
for i=1:length(exp_t)
    exp_t[i] = exp(tspan[2])
end
#println(exp_t)
I_ex=trapz(in,(1.0/(b-a)).*(u_sol - exp_t).^2)
I_x2 =trapz(in,(1.0/(b-a)).*(u_sol - exp_t).^4)
#sol = DifferentialEquations.solve(prob)



println("Taking step")
println("Expected  $I_ex")
println("Variance $(I_x2 - I_ex^2)")
println(typeof(I_ex))
#println((I_ex,I_x2,x,I_ex + I_x2 - I_ex^2))
#return 5.0

return I_ex + I_x2 - I_ex^2



end


optimizer = Juniper.Optimizer
nl_solver= optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
mip_solver = optimizer_with_attributes(Cbc.Optimizer, "logLevel" => 0, "threads"=>nthreads())
m = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver, "mip_solver"=>mip_solver))

#@variable(m, x[1:5], Bin)


# model = Model()
# @variable(m, x,Bin)
@variable(m, x,start=5)

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
