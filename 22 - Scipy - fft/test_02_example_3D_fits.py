import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftn

from astropy.io.fits.hdu import image
from astropy.io import fits

#Electron Example
x_data=np.arange(1,701,1)  
y_data=np.arange(1,101,1)
path='/home/mauricio/datosFits/30JUN2022/'
file='proc_skp_mod9_wfr13_Vv2_cryoUnpluged_ltaPower2GND_335toGnd_mksUnplugged_all2C15_EXPOSURE0_NSAMP2_NROW100_NCOL700_img10.fits'

# clean example
# x_data=np.arange(1,801,1)  #need to be the same size of the image
# y_data=np.arange(1,51,1)
# path='/home/mauricio/datosFits/27JUN2022_2/'
# file='skp_mod9_wfr13_NSAMP1_NROW50_NCOL800_EXPOSURE0_img20.fz'

X, Y =np.meshgrid(x_data,y_data) #creates a grid with the axis on XY plain


hdul=fits.open(path+file)# fits file to analyze
Z_data=hdul[1].data #numpy array #choose the extention you want to use
header=hdul[1].header



ax=plt.axes(projection="3d")
ax.plot_surface(X,Y,Z_data, cmap="gray") #cmap='plasma' looks good
ax.set_title("custom plot")
ax.set_xlabel("NColumn in X")
ax.set_ylabel("NRows in Y")
ax.set_zlabel("ADUs in Z")
plt.show()


n=700                      #magnitud del arreglo
timestep=.0001               #intervalo en segundos de cada muestra

fft_Z=np.fft.fft(Z_data[10])           #Amplitud de la espiga
fftFreqZ=np.fft.fftfreq(n,d=timestep)  #frecuencia de la espiga


fig, (ax1,ax2)=plt.subplots(2,1)
ax1.plot(Z_data[2])
ax2.plot(fftFreqZ,(np.abs(fft_Z)))
plt.show()