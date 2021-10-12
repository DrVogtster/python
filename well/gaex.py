import pygad
num=5
def fitness_function(solution, solution_idx):
    #global num
    return -sum(solution)

# ga_instance = pygad.GA(num_generations=10000,
#                        num_parents_mating=2,
#                        sol_per_pop=10,
#                        num_genes=4,
#                        fitness_func=fitness_function,gene_space=[0, 1],save_best_solutions=True)


ga_instance = pygad.GA(num_generations=10,
                       num_parents_mating=2,
                       sol_per_pop=3,
                       num_genes=4,
                       fitness_func=fitness_function,

                       init_range_low=5,
                       init_range_high=10,
                       
                       mutation_type=None,

                       gene_type=int,save_best_solutions=True)

ga_instance.run()
# 
# print(ga_instance.best_solution_generation)
a=ga_instance.best_solutions
b=ga_instance.best_solutions_fitness
my_list=[]
for i in range(0,len(a)):
    #print(b[i],a[i])
    my_list.append((b[i],a[i]))

my_list.sort(key=lambda tup: tup[0]) 
print(my_list)
