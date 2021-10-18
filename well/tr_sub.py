import cplex
from cplex.exceptions import CplexError

# see problem at min 0^T w + 0^T v + e^Ty
class subproblem:
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
        
def lol_print(arr):
    for k in arr:
        print(k)         

