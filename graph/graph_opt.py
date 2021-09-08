

def sharp_optimization_and_get_buy_sell_prices(n,m,k):
    
    M = AbstractModel()
    M.n = RangeSet(1,n)
    M.m = RangeSet(1,m)
    M.k = RangeSet(1,k)

    inital_zero = [0]*len(self.portfolio.keys())
    inital_one = [1]*len(self.portfolio.keys())
    max_percent = 1.0/len(self.portfolio.keys())
    buffer_val=.1
    # for i in range(0,max_amount_of_stocks):
    #     inital_one[i] = max_percent

    init_vec0 = self.vec_func((inital_zero))
    init_vec1 = self.vec_func((inital_one))
    print(max_percent)
    #M.bi = Var(M.number_stocks, domain=Binary,initialize=init_vec0)
    #M.ni = Var(M.number_stocks, domain=NonNegativeReals,initialize=init_vec1)
    M.wi = Var(M.number_stocks, domain=NonNegativeReals,initialize=init_vec1)
    M.n_set = RangeSet(1, 1)
    M.p = Param(M.n_set, initialize={1: portfolio_allocation_amount})
    M.buffer = Param(M.n_set, initialize={1: buffer_val})

    M.mp = Param(M.n_set, initialize={1: max_percent})
    M.ns = Param(M.n_set, initialize={1: max_amount_of_stocks})
    buy_vec = self.vec_func(buy_list)
    M.buy_price = Param(M.number_stocks, initialize=buy_vec)
    n_max = round(portfolio_allocation_amount/min(buy_list))
    M.n_max = Param(M.n_set, initialize={1: n_max})
    print((np.amin(self.avg_mat),np.amax(self.avg_mat)))
    print((np.amin(self.cov_mat),np.amax(self.cov_mat)))
    print((np.amin(self.cor_mat),np.amax(self.cor_mat)))

    avg_vec = self.vec_func(self.avg_mat)
    cov_mat = self.mat_func(self.cov_mat)
    cor_mat = self.mat_func(self.cor_mat)
    buy_vec= self.vec_func(buy_list)


    M.expected_price = Param(M.number_stocks, initialize=avg_vec)

    M.cov_mat = Param(M.number_stocks, M.number_stocks, initialize=cov_mat)
    M.cor_mat = Param(M.number_stocks, M.number_stocks, initialize=cor_mat)
    M.i = RangeSet(len(self.portfolio.keys()))
    #M.buffer = Param(M.n_set, initialize={1: buffer_val})
    M.C1 = Constraint(M.i, rule=self.greater_percent)
    #M.C1 = Constraint(M.i, rule=self.build_sos1_constraint_one)
    #M.C11 = Constraint(M.i, rule=self.build_sos1_constraint_two)
    M.C2 = Constraint(rule=self.percentage_constraint)
    #M.C4 = Constraint(rule=self.satisfy_allocation)
    # M.C3 = Constraint(rule=self.allocation_constraint)
    M.obj = Objective(rule=self.sharp_obj, sense=minimize)



    instance = M.create_instance()

    #results = SolverFactory('mindtpy').solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)
    print("solving optimization problem")
    # results=SolverFactory('mindtpy').solve(instance,strategy='ECP',
    #                            time_limit=3600, mip_solver='cplex', nlp_solver='ipopt',tee=True)
    results=SolverFactory('ipopt').solve(instance,tee=True)
    #results.options['max_iter']= 10000 #number of iterations you wish
    instance.solutions.store_to(results)

    new_bi = []

    new_ni = []
    new_wi=[]
    for p in instance.number_stocks:
        # print(instance.v[p].value)
        # new_ni.append(instance.ni[p].value)
        # new_bi.append(instance.bi[p].value)
        new_wi.append(instance.wi[p].value)
    #print(new_bi)
    #print(new_ni)
    print(new_wi)
    res_dic={}
    counter=0
    real_num_list=[]
    for k in range(0,len(new_wi)):
            num = new_wi[k]*portfolio_allocation_amount/buy_list[k]
            real_num_list.append(num)

    print("integer piece")
    #real_num_list=[]
    V = AbstractModel()
    V.number_stocks = RangeSet(1, len(self.portfolio.keys()))
    V.buy_vector = Param(V.number_stocks, initialize=buy_vec)
    V.n_set = RangeSet(1, 1)
    V.p = Param(V.n_set, initialize={1: portfolio_allocation_amount})
    num_vec  =self.vec_func(real_num_list)
    V.buff_val = Param(V.n_set, initialize={1: buffer_val})

    round_val = []
    for k in range(0,len(real_num_list)):
        round_val.append(round(real_num_list[k]))
    print(round_val)
    round_vec = self.vec_func(round_val)
    V.n_real = Param(V.number_stocks, initialize=num_vec)
    V.ni = Var(V.number_stocks, domain=NonNegativeIntegers,initialize=round_vec)
    V.C1 = Constraint(rule=self.n_c1)
    V.obj = Objective(rule=self.obj_n, sense=minimize)

    instance = V.create_instance()
    results=SolverFactory('mindtpy').solve(instance,strategy='OA',
                                time_limit=3600, mip_solver='cplex', nlp_solver='ipopt',tee=True)



    instance.solutions.store_to(results)

    new_n = []
    for p in instance.number_stocks:
        # print(instance.v[p].value)
        # new_ni.append(instance.ni[p].value)
        # new_bi.append(instance.bi[p].value)
        new_n.append(instance.ni[p].value)


    counter=0
    for k in self.portfolio.keys():
        print("-----")
        print(new_wi[counter])
        print(counter)
        print(len(new_wi))
        print(len(buy_list))
        print("-----")
        if(new_wi[counter]!=1):
            #num = math.floor(((new_wi[counter]*portfolio_allocation_amount)/buy_list[counter]))
            num = new_n[counter]


            if(num!=0):
                res_dic[k] = {"name": str(k), "buy_price": str(buy_list[counter]), "sell_price": str(sell_list[counter]), "number_to_buy":str(num), "historical_expected_return": str(num*self.avg_mat[counter])}
        counter+=1
    return_data=0.0
    basic_return=0
    ml_buy=0.0
    percent_buy=0.0
    with open('portfolio_results.json', 'w') as fp:
        json.dump(res_dic, fp)
    str_file = "human_readable_results.txt"
    with open(str_file, 'w') as the_file:
        the_file.write('--------------------------------------------------------\n')
        the_file.write('Summary for Portfolio Allocation ' + '\n')
        sum =0.0
        total_return=0.0
        for k in res_dic.keys():
            the_file.write('Buy '+ str(res_dic[k]['number_to_buy']) + ' ' + str(res_dic[k]['name']) + ' for ' +str(res_dic[k]['buy_price']) + ' and sell at ' + str(res_dic[k]['sell_price']) + 'for an expected histroical return of ' + str(res_dic[k]['historical_expected_return']) +  '\n')
            sum += float(res_dic[k]['number_to_buy'])*float(res_dic[k]['buy_price'])
            total_return+= float(res_dic[k]['historical_expected_return'])
            return_data+= float(res_dic[k]['number_to_buy'])*(float(self.portfolio[str(res_dic[k]['name'])]["Close"][-1])-float(res_dic[k]['buy_price']))
            basic_return+= float(res_dic[k]['number_to_buy'])*(float(self.portfolio[str(res_dic[k]['name'])]["Close"][-1])-float(self.portfolio[str(res_dic[k]['name'])]["Open"][-21]))
            ml_buy+=float(res_dic[k]['number_to_buy'])*(float(res_dic[k]['sell_price'])-float(res_dic[k]['buy_price']))
        the_file.write('total investment in portfolio: '  + str(sum) + 'expected return on historical expectations: ' + str(total_return) +'\n')
        the_file.write('Actual return in portfolio (if buying prices are met with ML 3 weeks ago): '  + str(return_data) + '\n')
        the_file.write('Actual return in portfolio (if buying with market current price 3 weeks ago): '  + str(basic_return) + '\n')
        the_file.write('Actual return in portfolio (if selling early with ML): '  + str(ml_buy) + '\n')
        the_file.write('--------------------------------------------------------\n')
    the_file.close()
