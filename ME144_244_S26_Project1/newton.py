import numpy as np
import matplotlib.pyplot as plt



# Part (a) - plot objection functions

x = np.linspace(-20, 20, 1000)

Pi_a = x**2
Pi_b = (x + ((np.pi / 2) * np.sin(x)))**2

plt.figure()
plt.plot(x, Pi_a, label=r'$\Pi_a(x) = x^2$')
plt.plot(x, Pi_b, label=r'$\Pi_b(x)$')
plt.legend()
plt.xlabel('x')
plt.ylabel('Cost')
plt.title('Objective Functions Comparison')
plt.savefig("figures/problem1a_objectives.png", dpi=300, bbox_inches="tight") #Save graph as png
plt.show()


# region Part B description
# Part (b) - Newton's Method Implementation

# x0 = starting position/initial guess
# f = function whose root we want to find
# df = (function) derivative of f : if steep --> take small steps, if shallow --> take big steps
# TOL = "close enough to zero" rule
# maxit = emergency stop -- after this many iterations if find nothing, stop


# NEWTON'S METHOD FOR OPTIMIZATION : Find the root of the derivative of the cost function
# so f is actually the derivative of the cost function, and df is the second derivative
# endregion

# region Algebraic Form
# Algebraic Form
# def myNewton(f, df, x0, TOL, maxit):

#     iterations = 0
#     x_curr = x0
#     hist = [x_curr]

#     while iterations < maxit:
#         f_curr = f(x_curr)
#         df_curr = df(x_curr)

#         if abs(f_curr) < TOL:
#             break

#         x_next = x_curr - (f_curr / df_curr)

#         hist.append(x_next)
#         x_curr = x_next

#         iterations += 1

#     sol = x_curr
#     its = iterations


#     return (sol, its, hist)
# endregion


# Part (b)

# Vectorized Form
def myNewton(f, df, x0, TOL, maxit):

    x = np.array([x0], dtype=float).reshape(-1, 1) # scalar to array form
    M = x.shape[0] # one dimension

    its = 0
    hist = [x.copy()]

    while its < maxit:
        f_curr = np.array(f(x), dtype=float).reshape(M,1)
        if np.linalg.norm(f_curr) < TOL:
            break

        H = np.array(df(x), dtype=float).reshape(M,M) # Hessian

        dx = np.linalg.solve(H, f_curr)

        x_next = x - dx
        hist.append(x_next.copy())
        x = x_next
        its += 1
    
    
    sol = x
    hist = np.hstack(hist) # convert list of arrays to single array

    return (sol, its, hist)




# region Explaining part c
# 1) 2 Cost functions
# 2) Get derivatives
# 3) Get double derivatives
# 4) Graph them
# endregion


# Part (c)

x_c = np.linspace(-20, 20, 2000)

# region Helpers for Pi_b
g = (x_c + ((np.pi / 2) * np.sin(x_c)))
gp = 1 + ((np.pi / 2) * np.cos(x_c))
gpp = -((np.pi / 2) * np.sin(x_c))
# endregion


# First derivatives
dPi_a = 2*x_c
dPi_b = 2*g*gp

# Second derivatives
ddPi_a = 2*np.ones_like(x_c) 
ddPi_b = 2*(gp**2 + (g * gpp))


# Graph First Derivatives
plt.figure()
plt.plot(x_c, dPi_a, label=r'$d\Pi_a(x)/dx$')
plt.plot(x_c, dPi_b, label=r'$d\Pi_b(x)/dx$')
plt.legend()
plt.xlabel('x')
plt.ylabel('Derivative of Cost')
plt.title('First Derivatives of Objective Functions')
plt.savefig("figures/problem1c_first_derivatives.png", dpi=300, bbox_inches="tight") #Save graph as png
plt.show()

# Graph Second Derivatives
plt.figure()
plt.plot(x_c, ddPi_a, label=r'$d^2\Pi_a(x)/dx^2$')
plt.plot(x_c, ddPi_b, label=r'$d^2\Pi_b(x)/dx^2$')
plt.legend()
plt.xlabel('x')
plt.ylabel('Second Derivative of Cost')
plt.title('Second Derivatives of Objective Functions')
plt.savefig("figures/problem1c_second_derivatives.png", dpi=300, bbox_inches="tight") #Save graph as png
plt.show()




# Part (d)

x0 = [2*10**k for k in (-1, 0, 1)] # [0.2, 2, 20]
TOL = 10**(-8)
maxit = 20

# region Cost Functions
def Pi_a_cost(x):
    return x**2

def Pi_b_cost(x):
    return (x + ((np.pi / 2) * np.sin(x)))**2
# endregion


# region First Derivatives

def dPi_a(x):
    return 2*x

def dPi_b(x):
    g  = x + (np.pi/2) * np.sin(x)
    gp = 1 + (np.pi/2) * np.cos(x)
    return 2 * g * gp
# endregion


# region Second Derivatives
def ddPi_a(x):
    return np.array([[2.0]])

def ddPi_b(x):
    g   = x + (np.pi/2) * np.sin(x)
    gp  = 1 + (np.pi/2) * np.cos(x)
    gpp = -(np.pi/2) * np.sin(x)
    return 2 * (gp**2 + g * gpp)
# endregion

# Graph Pi_a hist
plt.figure()
for x0_i in x0:
    sol, its, hist = myNewton(dPi_a, ddPi_a, x0_i, TOL, maxit)
    Pi_a_hist = Pi_a_cost(hist)

    iters = list(range(its + 1))
    yvals = [Pi_a_hist[0, i] for i in iters]

    plt.plot(iters, yvals, marker='o', label=f'x0 = {x0_i}')


plt.xlabel("Iterations")
plt.ylabel(r'$\Pi_a(x)$')
plt.title("Newton's Method to Minimize $\Pi_a(x)$")
plt.legend()
plt.savefig("figures/problem1d_Pi_a_convergence.png", dpi=300, bbox_inches="tight") #Save graph as png
plt.show()


#Graph Pi_b hist
plt.figure()
for x0_i in x0:
    sol, its, hist = myNewton(dPi_b, ddPi_b, x0_i, TOL, maxit)
    Pi_b_hist = Pi_b_cost(hist)

    iters = list(range(its + 1))
    yvals = [Pi_b_hist[0, i] for i in iters]

    plt.plot(iters, yvals, marker='o', label=f'x0 = {x0_i}')

plt.xlabel("Iterations")
plt.ylabel(r'$\Pi_b(x)$')
plt.title("Newton's Method to Minimize $\Pi_b(x)$")
plt.legend()
plt.savefig("figures/problem1d_Pi_b_convergence.png", dpi=300, bbox_inches="tight") #Save graph as png
plt.show()



