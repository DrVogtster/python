using LinearAlgebra

function H_gate()
    H = zeros((2,2))
    H[1,1] = 1
    H[1,2] = 1
    H[2,1] = 1
    H[2,2] = -1
    H = (1.0/math.sqrt(2))*H
    return H
end
function Z_theta(theta)
    Z = zeros(ComplexF64, (2,2))
    itheta = 1im*theta
    Z[1,1]  =1
    Z[2,2] = exp(itheta)
    return Z
end
function P_gate()
    return Z_theta(math.pi/2.0)
end
function T_gate()
    return Z_theta(math.pi/4.0)
end
function C_gate()
    C = zeros((4,4))
    C[1,1]=1
    C[2,2]=1
    C[4,3]=1
    C[3,4]=1
    return C
end


println(Z_theta(-pi/2.0))