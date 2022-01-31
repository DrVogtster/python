#=
adj:
- Julia version: 
- Author: mathb
- Date: 2022-01-29
=#

using DiffEqFlux
 using Flux
 using OrdinaryDiffEq
 using DiffEqSensitivity
 using ForwardDiff,Calculus,Tracker
 using ReverseDiff
print("starting")
function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + u[1]*u[2]
end

p = [1.5,1.0,3.0]
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
sol =solve(prob,Vern9(),abstol=1e-10,reltol=1e-10)

g(u,p,t) = (sum(u).^2) ./ 2
function dg(out,u,p,t)
  out[1]= u[1] + u[2]
  out[2]= u[1] + u[2]
end
# function dg(out,u,p,t)
#   out[1]= u[1]
#   out[2]=  u[2]
# end

ts = 0:0.5:10
function G(p)
  tmp_prob = remake(prob,u0=eltype(p).(prob.u0),p=p,
                    tspan=eltype(p).(prob.tspan))
  sol = solve(tmp_prob,Tsit5(),abstol=1e-14,reltol=1e-14)
  res,err = quadgk((t)-> (sum(sol(t)).^2)./2,0.0,10.0,atol=1e-14,rtol=1e-10)
  res
end
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0,1.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0,1.0])


adj_prob = ODEAdjointProblem(sol,QuadratureAdjoint(abstol=1e-14,reltol=1e-14),g,nothing,dg)
adj_sol = solve(adj_prob,Tsit5(),abstol=1e-14,reltol=1e-10)
integrand = AdjointSensitivityIntegrand(sol,adj_sol,QuadratureAdjoint(abstol=1e-14,reltol=1e-14))
res,err = quadgk(integrand,0.0,10.0,atol=1e-14,rtol=1e-10)


#
# res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-8,
#                                  reltol=1e-8,iabstol=1e-8,ireltol=1e-8)
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])
# res4 = Tracker.gradient(G,[1.5,1.0,3.0])
# res5 = ReverseDiff.gradient(G,[1.5,1.0,3.0])
println(res)
println(res2)
println(res3)
# println(res4)
# println(res5)