#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:42:36 2018

@author: luiggi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')

#
# Generamos datos en 2D cambiantes con el tiempo
#
x = np.linspace(-3, 3, 91)
y = np.linspace(-3, 3, 91)
t = np.linspace(0, 25, 30)
xg, yg, tg = np.meshgrid(x, y, t)
A = np.sin(2 * np.pi * tg / tg.max())
F = A * (xg**2 + yg**2)
#F = A * np.sin(xg**2 + yg**2)
#F = A * np.sin(xg) * np.cos(yg)

#
# Figura y ejes
#
fig = plt.figure(figsize=(4,4))           # Figuras
ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3)) # Ejes
#
# Se dibuja el primer conjunto de datos usando contornos
#
contour_opts = {'levels': np.linspace(-1, 1, 20), 'cmap':'RdBu', 'lw': 2}
cax = ax.contour(x, y, F[..., 5], **contour_opts)
#cax = ax.contour(x, y, F[:,:, 5], **contour_opts)
#
# Función para cambiar los datos dependientes del tiempo
#
def animate(i):
    ax.collections = []
    ax.contour(x, y, F[..., i], **contour_opts)
#
# Usamos "FuncAnimation" para realizar la animación de los diferentes
# conjuntos de datos 
#
anim = FuncAnimation(fig,           # La figura
                     animate,       # la función que cambia los datos
                     interval=100,  # Intervalo entre cuadros en milisegundos
                     frames=30,     # Cuadros por segundo
                     repeat=True)   # Permite poner la animación en un ciclo
   
