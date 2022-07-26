import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftn


x_data=np.arange(-5,5,0.1)
y_data=np.arange(-5,5,0.1)

X, Y =np.meshgrid(x_data,y_data)

Z=np.sin(X) * np.cos(Y)

ax=plt.axes(projection="3d")
ax.plot_surface(X,Y,Z, cmap="plasma")
ax.set_title("custom plot")
ax.set_xlabel("values in X")
ax.set_ylabel("values in Y")
ax.set_zlabel("reults in Z")
plt.show()