#=
sys_ex:
- Julia version: 
- Author: mathb
- Date: 2022-01-28
=#
using Juniper
using Ipopt
using JuMP
using LinearAlgebra # for the dot product
using Base.Threads
using Cbc
using DifferentialEquations
using Trapz
using LinearAlgebra

f(u,p,t) = p*u


# for i=1:100000
# size=i
# u0 = 1.0*Matrix(I, size, size)
#
# tspan = (0.0,10.0)
# p=(1/10.0)*Matrix(I, size, size)
#
# prob = @time ODEProblem(f,u0,tspan,p)
# println("size:$size")
# #u_sol = sol.u
# # println(u_sol[end])
# end


size=10000
u0 = 1.0*Matrix(I, size, size)

tspan = (0.0,200.0)
p=(1/10.0)*Matrix(I, size, size)

prob = @time ODEProblem(f,u0,tspan,p)
println("size:$size")
#u_sol = sol.u
# println(u_sol[end])
