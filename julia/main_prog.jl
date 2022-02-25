

using DiffEqFlux
using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using ForwardDiff,Calculus,Tracker
using ReverseDiff
using LinearAlgebra

include("kron_lib.jl")
include("gates.jl")


diconedtothreed=Dict()
dicthreedtooned=Dict()

function build_dics(Nc,Nt,Np)
    # global diconedtothreed
    # global dicthreedtooned
    # diconedtothreed={}
    # dicthreedtooned={}
    counter=1
    for i=1:Nc
        for j=1:Nt
            for k=1:Np
                diconedtothreed[(counter,)] = (i,j,k)
                dicthreedtooned[(i,j,k)] = counter
                counter+=1

            end
        end
    end
end

function fid_func(U,V, basis_list)


size = length(basis_list)
    U_hat = zeros(ComplexF64, (size,size))
    #V=np.kron(Z_theta(-math.pi),Z_theta(-math.pi/2.0))
    #V=np.kron(Z_theta(math.pi),np.eye(2))
    #V=np.kron(Z_theta(math.pi),Z_theta(math.pi/2.0))
    #V = Z_theta(math.pi/2)
    bad_states = []
    #print(np.array_str(U, precision=2, suppress_small=True))
    #quit()
    for i=1:size
        for j=1:size
            # print(basis_list[i])
            # print(basis_list[j])
            U_hat[i, j] = dot(basis_list[i], U*basis_list[j])
            # print(U_hat[i,j])
            #U_hat[i,j] = (basis_list[i].T.dot(U)*basis_list[j].T).sum(axis=1)
            if (V[i, j] == 0.0)
                bad_states.append((basis_list[i], basis_list[j]))
            end
        end
    end


    # U_hat[2:4,2:4]= -U_hat[2:4,2:4]
    U_hat = U_hat*(kron(Z_theta(pi),Matrix{Float64}(I, 2, 2)  ))
    # print("UHAT")
    # print(np.array_str(U_hat, precision=2, suppress_small=True))

    M = V'*U_hat

    TrM = tr(M)
    # print(TrM)
    # TrMr = TrM.real
    # TrMi = TrM.imag
    #mod_squared = TrMr ** 2 + TrMi ** 2

    fid = ((1.0) / (size * (size + 1))) * (tr(M*M') + abs(TrM)^2)
    # print(fid)
    leak = 0
    # for k in range(0,len(bad_states)):
    #     leak = leak + np.abs(np.transpose(bad_states[k][0]*U*bad_states[k][1]))**2
    # print(fid)
    # quit()
    return real(fid) + real(leak)



end
function creat_matrix_from_vector(vector, p, m, n):
    matrix_list = []
    for i in range(1, p):
        v_i = vector[i * m * n:(i + 1) * m * n]
        v_i_mat = reshape(v_i, (m, n))
        append!(matrix_list,[v_i_mat])
    return matrix_list

    end
    function test_fong(number_e, gate, T, dt, amp_list, plank):
    #f trust_region(number_e, gate, T, dt, amp_list, plank):
    global accum_list
    global mydic
    global diconedtothreed
    global dicthreedtooned
    print(dicthreedtooned)
    mydic = {}
    Nt = convert(Int64,(T / dt))
    Np = length(amp_list)
    Nc = number_e - 1

    sigma_x, sigma_y, sigma_z = generate_sigmas_xyz()
    real_list = []
    imag_list = []
    time_list = []
    H_list=[]
    for i=1:Nt-1
        append!(time_list,[[i * dt, (i + 1) * dt]])
    end
    my_order = [(3,2),(2,1),(1,4),(4,5),(5,6)]
    #my_order = [(1,2),(2,3),(3,4),(4,5),(5,6)]
    #my_order = [(1,2),(2,3)]
    # for i in range(1, number_e):
    #     s_iip1 = s_one(i, i + 1, sigma_x, sigma_y, sigma_z, 1,number_e)
    #     H_list.append(s_iip1)
    for k=1:length(my_order)
        ele = my_order[k]
        s_iip1 = s_one(ele[1], ele[2], sigma_x, sigma_y, sigma_z, 1,number_e)
        append!(H_list,[s_iip1])
    end
 
    neighbor_list = gen_right_neigh(Nc)
    # print(neighbor_list)
    # quit()
    x = fong_gen_single()
    
    arr = [None] * number_e

    acum_list=generateAllBinaryStrings(int(number_e/3), arr, 0)
    gamma = 1
    sigma = 0
    e0, e1 = dfs_0_1(gamma, sigma, x)
    basis_list = ket_gen_dfs(e0, e1, int(number_e/3), accum_list)
    #print(basis_list)
    #print(fid_leak_obj(gate, gate, basis_list))
    # e0 = [1, 0]
    # e1 = [0,1]

    # H = math.pi*H_list[0]
    # matprint(H)
    # sol = scipy.linalg.expm(-1j*H)
    # e100 = kron(e1,kron(e0,e0))
    # e010 = kron(e0,kron(e1,e0))
    # print(e100)
    # print(e010)
    # matprint(sol)
    # print(matprint(sol.conjugate()@sol))
    # print("-")
    # print(e100.shape)
  
    # res1 = sol@e100
    # res2 = sol@e010
    # print("---------------")
    # print(res1)
    # print(res2)
    # print("---------------")

    # tr = trust_region_problem(np.random.randint(2, size=Nc*Nt*Np).tolist(),.75,Nc*Nt*Np,fid_grad_routine_tr)
    # samples=20
    # sol_list=[]
    # best_sol_obj=None
    # best_sol_v=None
    # # pool = mp.Pool(mp.cpu_count())

    # # pool_list= [pool.apply_async(tr_helper, args=(i,))
    # #           for i in range(0,samples)]
    # ig = []
    # for k in range(0,Nt*Np*Nc):
    #     ig.append(0.0)
    # v_list =creat_matrix_from_vector(ig,Nc,Nt,Np)

    # #amp_list = [p1,p2,.5,-.5,1,p1,p2,-p1,1-p2,.25,-.25]
    # #v_list[i][k,random.randint(0, len(amp_list)-1)] = random.randint(0, 1) 
    
    # #[p1,p2,.5,-.5,1,-p1,1-p2,.25,-.25]
    # #[p1,p2,.5,3.0/2.0,1,-p1,1-p2]
 
    
    # # [p1,p2,.5,3.0/2.0,1,-p1,1-p2]
    # shift=0


    # # v_list[0][1,4]=1.0

    # # v_list[1][0,4]=1.0
    # # v_list[1][2,4]=1.0

    # # v_list[1][0+shift,4]=1.0
    # # v_list[3][0+shift,2]=1.0

    # # v_list[0][17,4]=1.0
    # # v_list[1][16,4]=1.0
    # # v_list[1][18,4]=1.0
    # #-----
    # # v_list[0][1,4]=1.0

    # # v_list[1][0,4]=1.0
    # # v_list[1][2,4]=1.0
    # v_list[1][4+shift,4]=1.0
    # v_list[1][6+shift,3]=1.0
    # v_list[1][8+shift,4]=1.0

    # v_list[2][1+shift,2]=1.0
    # v_list[2][3+shift,3]=1.0
    # v_list[2][5+shift,3]=1.0
    # v_list[2][7+shift,3]=1.0
    # v_list[2][9+shift,3]=1.0
    # v_list[2][11+shift,2]=1.0

    # v_list[3][0+shift,0]=1.0
    # v_list[3][2+shift,4]=1.0
    # v_list[3][4+shift,3]=1.0
    # v_list[3][6+shift,2]=1.0
    # v_list[3][8+shift,3]=1.0
    # v_list[3][10+shift,4]=1.0
    # v_list[3][12+shift,5]=1.0
    
    # v_list[4][1+shift,1]=1.0
    # v_list[4][3+shift,3]=1.0
    # v_list[4][5+shift,4]=1.0
    # v_list[4][7+shift,4]=1.0
    # v_list[4][9+shift,3]=1.0
    # v_list[4][11+shift,6]=1.0

    # # v_list[0][17,4]=1.0
    # # v_list[1][16,4]=1.0
    # # v_list[1][18,4]=1.0


    # #----
    # # v_list[0][1,4]=1.0

    # # v_list[1][0,4]=1.0
    # # v_list[1][2,4]=1.0

    
    # # v_list[1][0+shift,2]=1.0
    
    
    # # v_list[0][17,4]=1.0
    # # v_list[1][16,4]=1.0
    # # v_list[1][18,4]=1.0




    # out = v_list[0].flatten()
    # for i in range(1,len(v_list)):
    #     out =np.concatenate((out, v_list[i].flatten()), axis=None)
    # ig = out.tolist()
    # v_mat=creat_matrix_from_vector(ig,Nc,Nt,Np)
    # #cur_out.append((v_mat,cur_sol_obj))
    # make_movie(v_mat,Nc,Nt,Np,1,str(i)+"Ne" + str(number_e))
    # print(Nc,Nt,Np)
    # (obj,grad) = fid_grad_routine_tr(ig,False)
    # print("fong obj theta" + str(obj))
    # #print(ig)
    
        ig = []
        for k=1:Nt*Np*Nc
            append!(ig,[0.0])
        end
        if(Nc>2):
            v_list =creat_matrix_from_vector(ig,Nc,Nt,Np)

            shift=1
            v_list[1+shift][4+shift,4+shift]=1.0
            v_list[1+shift][6+shift,3+shift]=1.0
            v_list[1+shift][8+shift,4+shift]=1.0

            v_list[2+shift][1+shift,2+shift]=1.0
            v_list[2+shift][3+shift,3+shift]=1.0
            v_list[2+shift][5+shift,3+shift]=1.0
            v_list[2+shift][7+shift,3+shift]=1.0
            v_list[2+shift][9+shift,3+shift]=1.0
            v_list[2+shift][11+shift,2+shift]=1.0

            v_list[3+shift][0+shift,0+shift]=1.0
            v_list[3+shift][2+shift,4+shift]=1.0
            v_list[3+shift][4+shift,3+shift]=1.0
            v_list[3+shift][6+shift,2+shift]=1.0
            v_list[3+shift][8+shift,3+shift]=1.0
            v_list[3+shift][10+shift,4+shift]=1.0
            v_list[3+shift][12+shift,5+shift]=1.0
            
            v_list[4+shift][1+shift,1+shift]=1.0
            v_list[4+shift][3+shift,3+shift]=1.0
            v_list[4+shift][5+shift,4+shift]=1.0
            v_list[4+shift][7+shift,4+shift]=1.0
            v_list[4+shift][9+shift,3+shift]=1.0
            v_list[4+shift][11+shift,5+shift]=1.0
            out = v_list[0].flatten()

            for i in range(2,length(v_list)):
                out =concatenate((out, v_list[i].flatten()), axis=None)

            end
        for i in range(0,keep_sol):
            cur_out = []
            cur_sol = sol_list[i][1]
            cur_sol_obj = sol_list[i][0]
            v_mat=creat_matrix_from_vector(cur_sol,Nc,Nt,Np)
            cur_out.append((v_mat,cur_sol_obj))
            make_movie(v_mat,Nc,Nt,Np,1,str(i)+"Ne" + str(number_e))
            os.rename("__temp__.mp4", str(i)+"Ne" + str(number_e) +".mp4")
            output.append(cur_out)
        dic["output"] = output
        pickle.dump( dic, open( str(number_e)+".pkl", "wb" ) )
        end
        end






end

function runner(rotation_list,G,T,dt)


end

function gen_right_neigh(Nc)

    buddy_list=[]
    for i=1:Nc-1
        append!(buddy_list,[[i,i+1]])
    end
    return buddy_list
end




T = 130
dt = 10
p1 = acos(-1.0/sqrt(3))/pi
p2 = asin(1.0/3)/pi
amp_list = [p1,p2,.5,3.0/2.0,1,2-p1,1-p2]
number_e=9
for i=1:length(amp_list)
    amp_list[i] = amp_list[i]*pi
end

number_e = 6
plank = 1.0
# #gate = gate_list[1]
# #gate = gate_list[2]
C=C_gate()

build_dics(number_e-1,convert(Int64,(T/dt)),length(amp_list))
#test_fong(number_e, C, T, dt, amp_list, plank)

