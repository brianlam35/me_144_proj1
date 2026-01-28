import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-20, 20, 1000)

Pi_a = x**2
Pi_b = (x + ((np.pi / 2) * np.sin(x)))**2

plt.plot(x, Pi_a, label=r'$\Pi_a(x) = x^2$')
plt.plot(x, Pi_b, label=r'$\Pi_b(x)$')
plt.legend()
plt.xlabel('x')
plt.ylabel('Cost')
plt.title('Objective Functions Comparison')
plt.savefig("figures/problem1a_objectives.png", dpi=300, bbox_inches="tight") #Save graph as png
plt.show()