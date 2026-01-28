
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from typing import List
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "Computer Modern Roman",
#})

# Definitions
# cost_func_a             anonymous function for evaluating fitness (PI_a in Homework)
# cost_func_b             anonymous function for evaluating fitness (PI_b in Homework)
# P,            scalar,   number of design strings to preserve and breed
# TOL_GA,       scalar,   cost function to threshold to stop evolution
# G,            scalar,   maximum number of generations
# S,            scalar,   total number of design strings per generation
# dv,           scalar,   number of design variables per string
# PI,           G x S,    cost of sth design in the gth generation
# lim,          dv x 2,   lower and upper limits for each design variable

# design_array, S x dv,   array of most recent design strings
# g,            scalar,   generation counter
# PI_min,       1 x g,    minimum cost across strings and generations
# PI_avg,       1 x g,    average cost across strings and generations

# Helper Functions
def cost_func_a(design_array: List[float]) -> np.ndarray:
    # Return a numpy array of the cost for each value in design_array
    return 0

def cost_func_b(design_array: List[float]) -> np.ndarray:
    # Return a numpy array of the cost for each value in design_array
    return 0

def sort(pi: np.ndarray) -> List[np.ndarray, np.ndarray]:
    # Return a list with an array of the sorted costs and an array of the index order
    return [] 

# Freebie
def reorder(design_array: List[float], ind: np.ndarray) -> List[float]:
    temp = np.zeros((S,dv))
    for i in range(0, len(ind)):
        temp[i,:] = design_array[ind[i][0]]
    design_array = temp
    return design_array

# Fill in the Givens
P = 
TOL_GA = 
G = 
S = 
lim = [-20,20]
dv = 

domain_range = lim[1]-lim[0]
domain_min = lim[0]

# Initialize
PI = np.ones((G, S))
design_array = domain_range*np.random.rand(S, dv)+domain_min
g = 0
PI_min = np.zeros(G)
PI_avg = np.zeros(G)
MIN = 1000

# First generation
pi = cost_func_b(design_array)   # evaluate the fitness of each genetic string
[new_pi, ind] = sort(pi) # order in terms of decreasing "cost"

PI[0, :] = new_pi.reshape(1,S) # log the initial population "costs"

PI_min[0] = np.min(new_pi)
PI_avg[0] = np.mean(new_pi)
MIN = np.min(new_pi)

design_array = reorder(design_array, ind)

# All later generations
while (MIN > TOL_GA) and (g < G):
     
    # Mating 
    parents = design_array[0:P,:]
    children = np.zeros((P, dv))
    for p in list(range(0,P,2)): # p = 0, 2, 4, 6,...      
        if P % 2:
            print('P is odd. Choose an even number of parents.')
            break
        phi1 = np.random.rand()
        phi2 = 
        children[p,:]   = 
        children[p+1,:] = 
        
    # Update design_array (with parents)
    new_strings = 
    design_array = np.vstack((parents, children, new_strings)) # concatenate vertically

    # Update design_array (no parents)
    #new_strings = np.random.rand(S-P, dv)
    #design_array = np.vstack((children, new_strings)) # concatenate vertically

    # Evaluate fitness of new population
    pi =         
    [new_pi, ind] = sort(pi) 
    
    PI[g, :] = new_pi.reshape(1,S)        
    
    PI_min[g] = 
    PI_avg[g] = 
    if PI_min[g] < MIN:
        MIN = 
            
    design_array = reorder(design_array, ind)
    print(', '.join(('g = %s' % g, 'MIN = %s' % MIN)))
    g = g + 1

# Plotting
fig, ax = plt.subplots()
ax.semilogy(np.arange(0,g), PI_min[0:g])
ax.semilogy(np.arange(0,g), PI_avg[0:g])
plt.xlabel('Generation Number',  fontsize=20)
plt.ylabel('Cost', fontsize=20)
#title_str = '\n'.join(('Results of Genetic Algorithm with', 'Parents included in Subsequent Generations'))
title_str = '\n'.join(('Results of Genetic Algorithm without', 'Parents included in Subsequent Generations'))
plt.title(title_str, fontsize=20)
plt.legend(['Min Cost', 'Avg Cost'])
plt.show()
