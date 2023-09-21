import numpy as np
import matplotlib.pyplot as plt

# This file creats the contour plot of Figure 1 of the thess

# Define the Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Create a grid of points for x and y
X = np.linspace(-0.5, 0.5, 1000)
Y = np.linspace(-0.5, 0.5, 1000)
XX, YY = np.meshgrid(X, Y)
Z = rosenbrock(XX, YY)

# Create a contour plot
contour = plt.contour(XX, YY, Z, levels=np.logspace(-1,3,10), colors='black')
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.show()

