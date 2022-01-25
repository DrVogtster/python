#=
packages:
- Julia version: 
- Author: mathb
- Date: 2022-01-24
=#


using Pkg

dependencies = ["Juniper","Ipopt","JuMP", "Cbc","Alpine"]

Pkg.add(dependencies)