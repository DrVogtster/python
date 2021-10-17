import cplex
from cplex.exceptions import CplexError
import numpy as np
import math

class knapsack:
    def __init__(self,N,g,square_list):
        self.N = N
        self.square_list= square_list
        self.g = g
    def solve_problem(self):
        try:
            my_prob = cplex.Cplex()
            prob =my_prob
            prob.set_log_stream(None)
            prob.set_error_stream(None)
            prob.set_warning_stream(None)
            prob.set_results_stream(None)
            my_obj = self.g
            my_ctype = "B"
            number_of_one = self.square_list.count(1.0)
            my_ctype = my_ctype*len(self.square_list)
            val = self.N  -number_of_one
            rhs=[val]
            my_sense="L"
            my_rownames = ["r1"]

            counter =0
            variable_list=[]
            coiff_list=[]
            for i in self.square_list:
                if i==0:
                    coiff_list.append(1.0)
                else:
                    coiff_list.append(-1.0)
                variable_list.append("w" + str(counter))
                counter+=1

            rows = [[variable_list, coiff_list]]


            rhs2=[-1]
            rhs3 = [self.N]

            pos_ones = self.N*[1]
            neg_ones = self.N*[-1]
            rows2 = [[variable_list, neg_ones]]
            rows3 = [[variable_list, pos_ones]]




            prob.objective.set_sense(prob.objective.sense.minimize)

            prob.variables.add(obj=my_obj, types=my_ctype,
                        names=variable_list)
            prob.linear_constraints.add(lin_expr=rows, senses=my_sense,
                                    rhs=rhs)

            prob.linear_constraints.add(lin_expr=rows2, senses=my_sense,
                                    rhs=rhs2)
            prob.linear_constraints.add(lin_expr=rows3, senses=my_sense,
                                    rhs=rhs3)

            my_prob.solve()
            x = my_prob.solution.get_values()
            return x
        except CplexError as exc:
            print(exc)
            return

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