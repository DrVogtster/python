class aug_int:
    def __init__(self,v0,lamb0,eta_star,omega_star,projector_func,obj_grad_func,eq_func_list,max_iter):
        pass
        self.mu= [10]
        self.omega =[1.0/self.mu[0]]
        self.eta=[(1.0/self.mu[0])**(.1)]
        self.max_iter = max_iter
        self.v0 =v0
        self.lamb0 = lamb0
        self.eta_star=eta_star
        self.omega_star=omega_star

    def execute():
