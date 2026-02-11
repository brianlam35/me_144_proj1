import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from typing import List, Callable, Tuple
import os

os.makedirs("figures", exist_ok=True) # Create directory for figures if it doesn't exist

# Helper Functions
def cost_func_a(design_array: List[float]) -> np.ndarray:
    # Return a numpy array of the cost for each value in design_array
    x = design_array[:,0] # turn 2D array into 1D array [[x1], [x2], ...] -> [x1, x2, ...]
    return x**2 # apply to each element

def cost_func_b(design_array: List[float]) -> np.ndarray:
    # Return a numpy array of the cost for each value in design_array
    x = design_array[:,0]
    return (x + (np.pi/2 * np.sin(x)))**2 


def sort(pi: np.ndarray):
    
    pi = np.asarray(pi).reshape(-1)
    ind  = np.argsort(pi).reshape(-1,1) 
    new_pi = pi[ind[:,0]]               
                                        
    return new_pi, ind

# Freebie
def reorder(design_array, ind):
    temp = np.zeros_like(design_array)
    for i in range(len(ind)):
        temp[i,:] = design_array[ind[i,0], :]
    return temp


# -----------------------------------------------------------------------------------

# Genetic Algorithm

def genetic_algorithm(
    cost_func: Callable[[np.ndarray], np.ndarray],
    S: int,
    P: int,
    K: int,
    TOL: float,
    G: int,
    dv: int,
    lim: np.ndarray,
    preserve_parents: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    

    # ---------------
    # Limit Checks
    # ---------------

    lim = np.asarray(lim, dtype=float)
    if lim.shape != (dv, 2):
        raise ValueError(f"lim must be shape (dv,2) = ({dv},2). Got {lim.shape}")
    
    if S <= 0 or P < 0 or K < 0 or G <= 0 or dv <= 0:
        raise ValueError("S, G, dv must be >0 and P, K must be >=0")
    
    if P > S:
        raise ValueError("P cannot be greater than S")
    if K > S:
        raise ValueError("K cannot be greater than S")
    if preserve_parents:
        if P + K > S:
            raise ValueError("P + K <= S")
    else:
        if K > S:
            raise ValueError("K <= S")
        
    if (P % 2) != 0:
        raise ValueError("P must be even")
    if (K % 2) != 0:
        raise ValueError("K must be even")
    


    # ---------------
    # Initialization
    # ---------------

    rng = np.random.default_rng()
    domain_min = lim[:, 0]
    domain_range = lim[:, 1] - lim[:, 0]

    # Allocate
    PI = np.zeros((G, S)) # cost of each string in each generation
    PI_min = np.zeros(G) # best cost in generation g
    PI_avg = np.zeros(G) # average cost in generation g

    design_array = domain_range * rng.random((S, dv)) + domain_min


    # -----------------
    # First Generation
    # -----------------
    pi = cost_func(design_array)
    new_pi, ind = sort(pi)
    design_array = reorder(design_array, ind)

    PI[0, :] = new_pi
    PI_min[0] = float(np.min(new_pi))
    PI_avg[0] = float(np.mean(new_pi))
    MIN = PI_min[0]

    g = 1
    while (MIN > TOL) and (g < G):
        # Mating
        parents = design_array[0:P,:]
        children = np.zeros((K, dv))

        for c in range(0, K, 2):
            pair = (c // 2) % (P // 2)
            p = 2 * pair

            phi1 = rng.random()
            phi2 = rng.random()

            #combination of neighbors
            child1 = phi1 * parents[p, :] + (1.0 - phi1) * parents[p + 1, :]
            child2 = phi2 * parents[p, :] + (1.0 - phi2) * parents[p + 1, :]

            children[c, :] = child1
            children[c + 1, :] = child2

        # Keep within bounds
        children = np.clip(children, lim[:, 0], lim[:, 1])


        # Update design_array
        if preserve_parents:
            n_new = S - P - K
            new_strings = domain_range * rng.random((n_new, dv)) + domain_min
            design_array = np.vstack((parents, children, new_strings))
        else:
            n_new = S - K
            new_strings = domain_range * rng.random((n_new, dv)) + domain_min
            design_array = np.vstack((children, new_strings))


        # Evaluate
        pi = cost_func(design_array)
        new_pi, ind = sort(pi)
        design_array = reorder(design_array, ind)

        PI[g, :] = new_pi
        PI_min[g] = float(np.min(new_pi))
        PI_avg[g] = float(np.mean(new_pi))

        if PI_min[g] < MIN:
            MIN = PI_min[g]

        g += 1
    
    Lambda = design_array  # most recent generation, sorted best->worst
    return PI, PI_min, PI_avg, Lambda



# -----------------------------------------------------------------------------------

# -----------------
# Main - Run Part (a) and (b)
# -----------------
if __name__ == "__main__":
    P = 12
    TOL_GA = 1e-6
    G = 100
    S = 50
    K = 12
    lim = np.array([[-20, 20]], dtype=float)
    dv = 1


    # ------------------
    # Part (a)
    # ------------------
    PI_a, PI_min_a, PI_avg_a, Lambda_a = genetic_algorithm(
        cost_func=cost_func_a,
        S=S, P=P, K=K, TOL=TOL_GA, G=G, dv=dv, lim=lim,
        preserve_parents=True,
    )

    valid_a = np.isfinite(PI_min_a)
    gens_a = np.where(valid_a)[0]

    fig, ax = plt.subplots()
    ax.semilogy(gens_a, PI_min_a[valid_a], label="Min Cost")
    ax.semilogy(gens_a, PI_avg_a[valid_a], label="Avg Cost")
    ax.set_xlabel("Generation Number", fontsize=14)
    ax.set_ylabel("Cost", fontsize=14)
    ax.set_title("Genetic Algorithm Debug on $\\Pi_a$", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/ga_debug_PIa.png", dpi=300, bbox_inches="tight")
    plt.show()


    # ------------------
    # Part (b)
    # ------------------
    PI_b, PI_min_b, PI_avg_b, Lambda_b = genetic_algorithm(
        cost_func=cost_func_b,
        S=S, P=P, K=K, TOL=TOL_GA, G=G, dv=dv, lim=lim,
        preserve_parents=True,
        verbose=True,
    )

    valid_b = np.isfinite(PI_min_b)
    gens_b = np.where(valid_b)[0]

    fig, ax = plt.subplots()
    ax.semilogy(gens_b, PI_min_b[valid_b], label="Min Cost")
    ax.semilogy(gens_b, PI_avg_b[valid_b], label="Avg Cost")
    ax.set_xlabel("Generation Number", fontsize=14)
    ax.set_ylabel("Cost", fontsize=14)
    ax.set_title("Genetic Algorithm on $\\Pi_b$", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/ga_results_PIb.png", dpi=300, bbox_inches="tight")
    plt.show()


