

 acum_list=[]
# Function to generate all binary strings
function printTheArray(arr, n)
    tem = []
    for i=1:n
        print(arr[i])
        append!(tem,[arr[i]])
    end
    append!(acum_list,[tem])
     
    

end
function get_acum()
return acum_list
end
function generateAllBinaryStrings(n, arr, i)

    if i-1 == n
        printTheArray(arr, n)
       #append!( acum_list,[arr])
        return
        end
     
    # First assign "0" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1)
 
    # And then assign "1" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1)
 
end


function split(word)
    return [char for char in word]
    end
function convert_str_to_list(mes)
        int_list=[]
        for i=1:length(mes)
           bit = mes[i]
           append!(int_list,[bit])
        end
    return int_list
end
function ret_vec(val,e0,e1)
    if(val==0)
        return e0
    else
        return e1
        end
        end

function ket_gen(order)
    e0=[1,0]
    e1=[0,1]
    print(order)
    k = len(order)-3
    temp =np.kron(ret_vec(order[end-1],e0,e1),ret_vec(order[end],e0,e1))
    while(k>=0)
        if(order[k]==0)
            temp = kron(e0,temp)
        else
            temp = kron(e1,temp)

        k=k-1
        end
    end
    return temp
    end

function ket_gen_dfs(e0,e1,nq,com_list)
    if(nq ==1)
        return [e0,e1]
    end
    

    out_list=[] 
    for p in com_list
        order = p
        temp =np.kron(ret_vec(order[0],e0,e1),ret_vec(order[1],e0,e1))
        k = 2
        while(k<len(order))
            if(order[k]==0)
                temp = np.kron(temp,e0)
            else
                temp = np.kron(temp,e1)
                end

            k=k+1
            end
        out_list.append(temp)
        end
    return out_list

end


function ket_gen(order)
    e0=np.asarray([1,0])
    e1=np.asarray([0,1])
    k = len(order)-3
    temp =np.kron(ret_vec(order[end-1],e0,e1),ret_vec(order[end-1],e0,e1))
    while(k>=0)
        if(order[k]==0)
            temp = kron(e0,temp)
        else
            temp = kron(e1,temp)
            end

        k=k-1  
          end
    return temp
end

function get_vec(string)
    con_s = convert_str_to_list(string)
    println(con_s)
    val = ket_gen(con_s)
    #print(val)
    return val
end
function fong_gen_single()
    f_mat = zeros((8,8))
    e_000 = get_vec("000")
    e_100 = get_vec("100")
    e_010 = get_vec("010")
    e_001 = get_vec("001")
    e_111 = get_vec("111")
    e_101 = get_vec("101")
    e_011 = get_vec("011")
    e_110 = get_vec("110")

    f_mat[:,1] = (1.0/ sqrt(2))*(e_010-e_100)
    f_mat[:,2] = (1.0/ sqrt(2))*(e_011-e_101)
    f_mat[:,3] = ( sqrt(2.0/3.0))*(e_001) -  sqrt(1.0/6.0)*(e_010) - sqrt(1.0/6.0)*(e_100)
    f_mat[:,4] = ( sqrt(1.0/6.0))*(e_011) + sqrt(1.0/6.0)*(e_101) - sqrt(2.0/3.0)*(e_110)
    f_mat[:,5] = e_000
    f_mat[:,6] = ( sqrt(1.0/3.0))*(e_001 +e_010+e_100)
    f_mat[:,7] = ( sqrt(1.0/3.0))*(e_011 +e_101+e_110)
    f_mat[:,8] = e_111
    # print("start")
    # print(f_mat[:,3])
    # print("done")
    # quit()
    return f_mat
    end
function dfs_0_1(gamma,sigma,f_mat)

    f1= f_mat[:,0]
    f2 = f_mat[:,1]
    f3 = f_mat[:,2]
    f4= f_mat[:,3]

    dfs_0 = (gamma*f1 + sigma*f2)
    dfs_1= (gamma*f3 + sigma*f4)
    return dfs_0,dfs_1
    end

#dont scale by .25
function s_one(m,n,sigma_x,sigma_y,sigma_z,k_min,k_max)
    #k_min=3
    dim=2
#     matprint(.25*(sigmas_n(m,sigma_x,k_min,k_max,dim)@sigmas_n(n,sigma_x,k_min,k_max,dim) + sigmas_n(m,sigma_y,k_min,k_max,dim)@sigmas_n(n,sigma_y,k_min,k_max,dim) + sigmas_n(m,sigma_z,k_min,k_max,dim)@sigmas_n(n,sigma_z,k_min,k_max,dim))
# )
    return .25*(sigmas_n(m,sigma_x,k_min,k_max,dim)*sigmas_n(n,sigma_x,k_min,k_max,dim) + sigmas_n(m,sigma_y,k_min,k_max,dim)*sigmas_n(n,sigma_y,k_min,k_max,dim) + sigmas_n(m,sigma_z,k_min,k_max,dim)*sigmas_n(n,sigma_z,k_min,k_max,dim))
end
function generate_sigmas_xyz()
    sigma_x = np.zeros((2,2), dtype=complex)
    sigma_x[1,2] =1
    sigma_x[2,1] =1
    
    sigma_y = np.zeros((2,2), dtype=complex)
    sigma_y[1,2]=-1j
    sigma_y[2,1]=1j

    sigma_z = np.zeros((2,2), dtype=complex)
    sigma_z[1,1]=1.0
    sigma_z[2,2] = -1.0
    return sigma_x,sigma_y,sigma_z

end



function sigmas_n(n,sigma,k_min,k_max,dim)
    eye = 1.0*Matrix(I, dim, dim)
    start=nothing

    if(k_min==n)
        start = kron(sigma,eye)
    elif(k_min+1==n)
        start = np.kron(eye, sigma)
    else
        start=np.kron(eye,eye)    

end
    k_min=k_min+2
    while(k_min<=k_max)
        if(k_min==n)
            start = np.kron(start,sigma)
        else
            start = np.kron(start,eye)
        end


        
        k_min=k_min+1
        end
    return start
    end


