import cplex
from cplex.exceptions import CplexError
import numpy as np
import math

class knapsack:
  def __init__(self,cur_v,mygrad,Nc,Nt,Np,amp_list,neighborlist,diconedtothreed,dicthreedtooned,tr_radius):
        self.cur_v=cur_v
        self.args=args
        self.Nc=Nc
        self.Nt = Nt
        self.amp_list=amp_list
    def solve_problem(self):
        num_vars = self.Nc*self.Nt*self.Np
        cur_v = self.cur_v

        try:
            my_prob = cplex.Cplex()
            prob =my_prob
            #prob.set_log_stream(None)
            #prob.set_error_stream(None)
            #prob.set_warning_stream(None)
            #prob.set_results_stream(None)
            
    

            my_obj = mygrad
            my_ctypeB = "B"
           
            
            my_ctype = my_ctypeB*len(v) 
            variable_list=[]
            numlist=[]
            for k in range(0,len(my_ctype)):
                variable_list.append(str(k))
                numlist.append(k)

            
            val2=self.cs

            #print(val)
            #print(len(self.g))
            #print(len(self.square_list))
            #rhs=[val]
            coeff = [0]*len(cur_v)

            for k in range(0,len(cur_v)):
                if(cur_v[k]==0):
                    coeff[k]= 1
                else:
                    coeff[k]=-1
            
            rhs = [tr_radius - cur_v.count(1)]

            my_sense="L"
            my_rownames = ["r1"]
            rows  =[[variable_list,coeff]]
            prob.linear_constraints.add(lin_expr=rows, senses=my_sense,rhs=rhs)
            
            


            prob.objective.set_sense(prob.objective.sense.minimize)
            #print(len(my_obj))
            #print(len(my_ctype))
            #print(len(variable_list))
            #print(my_obj)
            #print(my_ctype)
            #print(variable_list)
            prob.variables.add(obj=my_obj, types=my_ctype,names=variable_list)
           


            for i in range(0, Nt):
                cur_var_list=[]
                cur_num_var=[]
                for j in range(0,Nc):
                    neighbors = neighbor_list[j]
                    for n in neighbors:
                        for k in range(0,Np):
                            cur_var_list.append(variable_list[mydic[(i,n,k)]])
                            cur_num_var.append(numlist[mydic[(i,n,k)]])
                    prob.SOS.add(type="1", SOS=[cur_var_list, cur_num_var])

                


            #print("SOLVING A TRUST REGION SUB PROBLEM")
            my_prob.solve()
            #print("DONE SOLVING A TRUST REGION SUBPROBLEM")
            x = my_prob.solution.get_values()
          


            return x
        except CplexError as exc:
            print(exc)
            # mysum=0.0
            # for k in range(0,number_squares):
            #     sub_slice = square_list[k*nummats:k*nummats+nummats]

            #     for j in range(0,len(sub_slice)):
            #         mysum = mysum+sub_slice[j]*density_list[j]*dx**2
            # print(mysum)
            # print(self.cs)
            return None
        

class trust_region_problem:
    def __init__(self,v0,rho0,tr_radius0,obj_grad_func):
        self.v = v0
        self.rho =rho0
        self.tr_radius=tr_radius0
        self.obj_grad_func = obj_grad_func

    def execute_tr(self):
        v_cur = self.v
        rho = self.rho
        tr_radius = self.tr_radius
        obj_grad_func=self.obj_grad_func
        counter=0
        
        obj_cur,grad_cur = obj_grad_func(v_cur)
        while(tr_radius>=1):
            
            knap_problem = knapsack(len(v_cur),grad_cur,v_cur)
            new_v = knap_problem.solve_problem()
           
            obj_new,grad_new = obj_grad_func(new_v)
            diff =len(v_cur)*[0]
            my_sum=0.0
          
            for k in range(0,len(v_cur)):
                diff[k] = new_v[k]- v_cur[k]
                my_sum=my_sum+ diff[k]*grad_cur[k]
                #print(grad_cur)
            if(my_sum==0):
                print("Stationary point (local) found in " +str(counter) + " iterations")
                break
            rho_k = (obj_cur - obj_new)/(-my_sum)
            if(rho_k>rho):
                v_cur = new_v
                if(math.fabs(sum(diff))==tr_radius):
                    tr_radius=tr_radius*2
                obj_cur = obj_new
                grad_cur = grad_new
            elif(rho_k>0):
                v_cur = new_v
                obj_cur = obj_new
                grad_cur = grad_new

            else:
                tr_radius = int(math.floor(tr_radius/2))
            counter+=1
            print("-----------------")
            print("Current Objective: " +str(obj_cur))
            print("New Objective: " +str(obj_new))
            print("Current Radius" +str(tr_radius))

            print("-----------------")
        print("Trust-region terminated successfully")
        return(obj_cur,v_cur,np.linalg.norm(grad_cur,2))



def test_obj_grad_func(v):
    obj=0.0
    grad = len(v)*[0.0]
    for k in range(0,len(v)):
        obj = obj + (-1)**k*v[k]**2
        grad[k] = (-1)**k*2.0*v[k]
    return (obj,grad)


# size=100
# v = size*[1.0]
# # for k in range(0,len(v)):
# #     if(k%2==0):
# #         v[k]=1
# #     else:
# #         v[k]=0


# rho =.75
# tr_radius = size

# x = trust_region_problem(v,rho,tr_radius,test_obj_grad_func)
# sol = x.execute_tr()
# print(sol)



# x= knapsack(2000,2000*[1],2000*[1])
# v = x.solve_problem()
# print(v)
# print(len(v))
#test