
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from typing import List
import os

os.makedirs("figures", exist_ok=True) # Create directory for figures if it doesn't exist
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
    x = design_array[:,0] # turn 2D array into 1D array [[x1], [x2], ...] -> [x1, x2, ...]
    return x**2 # apply to each element

def cost_func_b(design_array: List[float]) -> np.ndarray:
    # Return a numpy array of the cost for each value in design_array
    x = design_array[:,0]
    return (x + (np.pi/2 * np.sin(x)))**2 # applies to each element


def sort(pi: np.ndarray):
    # Return a list with an array of the sorted costs and an array of the index order
    pi = np.asarray(pi).reshape(-1)
    ind  = np.argsort(pi).reshape(-1,1)  # get indices that would sort pi [[3], [0], ...] (low-> high)
    new_pi = pi[ind[:,0]]               # reorder pi using indices [pi[3], pi[0], ...] -> [0.5, 1.0, ...]
                                        # ind[:,0] -> [3, 1, ...] then apply pi for value
    return new_pi, ind

# Freebie
def reorder(design_array, ind):
    temp = np.zeros_like(design_array) # same shape as input
    for i in range(len(ind)):
        temp[i,:] = design_array[ind[i,0], :]
    return temp


# Fill in the Givens
P = 12
TOL_GA = 1e-6
G = 100
S = 50
K = 12
lim = np.array([[-20,20]])
dv = 1

domain_range = lim[:, 1] - lim[:, 0]
domain_min = lim[:, 0]

# Initialize
PI = np.ones((G, S))
design_array = domain_range*np.random.rand(S, dv) + domain_min 
g = 0
PI_min = np.zeros(G) # best cost in generation g
PI_avg = np.zeros(G) # average cost in generation g
MIN = 1000 # large initial value for minimum cost

# First generation
pi = cost_func_b(design_array)   # evaluate the fitness of each genetic string
[new_pi, ind] = sort(pi) # order in terms of decreasing "cost"

PI[0, :] = new_pi.reshape(1,S) # log the initial population "costs"

PI_min[0] = np.min(new_pi)
PI_avg[0] = np.mean(new_pi)
MIN = np.min(new_pi)

design_array = reorder(design_array, ind)

g = 1

# All later generations
while (MIN > TOL_GA) and (g < G):
     
    # Mating 
    parents = design_array[0:P,:]
    children = np.zeros((K, dv))

    if P % 2:
            print('P is odd. Choose an even number of parents.')
            break
    if K % 2:
            print('K is odd. Choose an even number of children.')
            break

    for c in range(0,K,2): # p = 0, 2, 4, 6,...  

        #which parents to mate? (p and p+1)
        pair = (c//2) % (P//2) # pair = 0, 0, 1, 1, 2, 2,... (wraps around after P/2 pairs)
        p = 2*pair # p = 0, 0, 2, 2, 4, 4,... (wraps around after P parents)

        phi1 = np.random.rand() # how much parent contributes
        phi2 = np.random.rand()
        children[c,:]   = phi1*parents[p,:] + (1-phi1)*parents[p+1,:] #mate parent 0 and 1
        children[c+1,:] = phi2*parents[p,:] + (1-phi2)*parents[p+1,:] # same parents different contribution
        
    # Update design_array (with parents)
    new_strings = domain_range*np.random.rand(S-P-K, dv) + domain_min
    design_array = np.vstack((parents, children, new_strings)) # concatenate vertically

    # Update design_array (no parents)
    #new_strings = np.random.rand(S-P, dv)
    #design_array = np.vstack((children, new_strings)) # concatenate vertically

    # Evaluate fitness of new population
    pi = cost_func_b(design_array)   # evaluate the fitness of each genetic string        
    [new_pi, ind] = sort(pi) 
    
    PI[g, :] = new_pi.reshape(1,S)        
    
    PI_min[g] = np.min(new_pi)
    PI_avg[g] = np.mean(new_pi)
    if PI_min[g] < MIN:
        MIN = PI_min[g]
            
    design_array = reorder(design_array, ind)
    print(', '.join(('g = %s' % g, 'MIN = %s' % MIN)))
    g = g + 1

# Plotting
fig, ax = plt.subplots()
ax.semilogy(np.arange(0,g), PI_min[0:g])
ax.semilogy(np.arange(0,g), PI_avg[0:g])
plt.xlabel('Generation Number',  fontsize=20)
plt.ylabel('Cost', fontsize=20)
title_str = '\n'.join(('Results of Genetic Algorithm with', 'Parents included in Subsequent Generations'))
#title_str = '\n'.join(('Results of Genetic Algorithm without', 'Parents included in Subsequent Generations'))
plt.title(title_str, fontsize=20)
plt.legend(['Min Cost', 'Avg Cost'])
plt.show()
