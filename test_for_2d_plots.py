import matplotlib.pyplot as plt
import numpy as np

# Set the center and radius of the circle
center = (0.5, 0.5)
radius = 0.3

# Generate points on the circumference of the circle
theta = np.linspace(0, 2*np.pi, 500)
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)

# Create the figure and axis objects
fig, ax = plt.subplots()

# Fill the circle
ax.fill(x, y, color='blue')

# Set the aspect ratio to 'equal' to ensure a circular shape
ax.set_aspect('equal')

# Set the limits of the plot
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Filled Circle')

# Display the plot
plt.show()