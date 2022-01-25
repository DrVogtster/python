#=
mip:
- Julia version: 
- Author: mathb
- Date: 2022-01-24
=#
using Juniper
using Ipopt
using JuMP
using LinearAlgebra # for the dot product
using Base.Threads
using Cbc
#using Alpine

function test(w,x)

return sum(w[i]*x[i]^2 for i=1:5)
end


optimizer = Juniper.Optimizer
nl_solver= optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
mip_solver = optimizer_with_attributes(Cbc.Optimizer, "logLevel" => 0, "threads"=>nthreads())
m = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver, "mip_solver"=>mip_solver))



v = [10,20,12,23,42]
w = [12,45,12,22,21]
#@variable(m, x[1:5], Bin)


# model = Model()
@variable(m, x[1:5],Bin)
@objective(m, Max, dot(v,x))
f(x...) = test(w,x)
#register(m, :f, 5, f; autodiff = true)
register(m, Symbol("g1"), 5, f; autodiff = true)
register(m, Symbol("g2"), 5, f; autodiff = true)
#@NLobjective(model, Min, f(x...))
t(x) = test(w,x)

#@NLconstraint(m, sum(w[i]*x[i]^2 for i=1:5) <=45)
@NLconstraint(m, g1(x...) <=45)
@NLconstraint(m, g2(x...) <=20)

optimize!(m)

# retrieve the objective value, corresponding x values and the status
println(JuMP.value.(x))
println(JuMP.objective_value(m))
println(JuMP.termination_status(m))


# const alpine = optimizer_with_attributes(Alpine.Optimizer, 
#                                         #  "minlp_solver" => minlp_solver,
#                                         "nlp_solver" => nlp_solver,  
#                                         "mip_solver" => mip_solver,
#                                         "presolve_bt" => true,
#                                         "presolve_bt_max_iter" => 5,
#                                         "disc_ratio" => 10)

# # Try different integer values ( >=4 ) for `disc_ratio` to have better Alpine run times 
# # Choose `presolve_bt` to `false` if you prefer the OBBT presolve to be turned off. 

# m = nlp3(solver=alpine)
# v = [10,20,12,23,42]
# w = [12,45,12,22,21]
# #@variable(m, x[1:5], Bin)


# # model = Model()
# @variable(m, x[1:5],Bin)
# @objective(m, Max, dot(v,x))
# f(x...) = test(w,x)
# #register(m, :f, 5, f; autodiff = true)
# register(m, Symbol("g1"), 5, f; autodiff = true)
# register(m, Symbol("g2"), 5, f; autodiff = true)
# #@NLobjective(model, Min, f(x...))
# t(x) = test(w,x)

# #@NLconstraint(m, sum(w[i]*x[i]^2 for i=1:5) <=45)
# @NLconstraint(m, g1(x...) <=45)
# @NLconstraint(m, g2(x...) <=20)

# optimize!(m)

# # retrieve the objective value, corresponding x values and the status
# println(JuMP.value.(x))
# println(JuMP.objective_value(m))
# println(JuMP.termination_status(m))
# JuMP.optimize!(m)


