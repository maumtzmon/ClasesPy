# import matplotlib.pyplot as plt
# import numpy as np

# x=np.array(range(1000))
# y=np.array(range(1, 16000,16))

# plt.plot(x,y)

# plt.show()


import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
surf=ax.plot_surface(x, y, z)
fig.colorbar(surf, shrink=0.5)

plt.show()