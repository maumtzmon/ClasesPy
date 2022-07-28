from re import M
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftn

from astropy.io.fits.hdu import image
from astropy.io import fits


###
# Flags
###

plotOp = True

###
#  File sellection section
###

path='/home/oem/datosFits/27JUL2022/cryo_off_noLakeshore_newPowerSupply/'
file='skp_mod9_wfr13_Vvtest_vthswings2.5_sh2.5_oh-2.5_EXPOSURE0_NSAMP1_NROW700_NCOL800_img6.fz'

# path='/home/oem/datosFits/27JUN2022_2/'
# file='skp_mod9_wfr13_NSAMP1_NROW50_NCOL800_EXPOSURE0_img20.fz'

###
#  Extention section
###
ext=3 #0-3 fits; 1-4 fz
if file.endswith('.fz'):
    if ext==0:
        ext=1

###
# Adjust the shape automatically
###
hdul=fits.open(path+file)# fits file to analyze
Z_data=hdul[ext].data #numpy array #choose the extention you want to use
n_rows,n_cols=hdul[ext].shape
x_data=np.arange(0,n_cols,1)  
y_data=np.arange(0,n_rows,1)
X, Y =np.meshgrid(x_data,y_data) #creates a grid with the axis on XY plain


###
# Try to mask something
###
#
m_data=np.ma.masked_where(~((np.mean(Z_data*.95)<Z_data)&(Z_data<np.mean(Z_data*1.05))), Z_data) #To do


plt.figure()
plt.imshow(m_data.filled(fill_value=np.mean(Z_data)),cmap='gray')
plt.show()


###
# Plot 3d Image
###
if plotOp== True:
    ax=plt.axes(projection="3d")
    ax.plot_surface(X,Y,m_data.filled(fill_value=np.mean(Z_data)),cmap='gray') #cmap='plasma' looks good
    ax.set_title("custom plot")
    ax.set_xlabel("NColumn in X")
    ax.set_ylabel("NRows in Y")
    ax.set_zlabel("ADUs in Z")
    plt.show()



timestep_pix_col=.0001               #intervalo en segundos de cada pixel en X
timestep_pix_row=.1                  #intervalo en segundos de cada pixel en y


row=25                                  #which row Default value = 25
fft_Z_row=np.fft.fft(Z_data[row,:])           #Amplitud de la espiga
fftFreqZ_row=np.fft.fftfreq(n_cols,d=timestep_pix_col)  #frecuencia de la espiga

col=200                                 #which col 
fft_Z_col=np.fft.fft(Z_data[:,col])
fftFreqZ_col=np.fft.fftfreq(n_rows,d=timestep_pix_row)

fig, ax=plt.subplots(2,2)
# row analysis
ax[0,0].plot(Z_data[row])
ax[1,0].plot(np.abs(fftFreqZ_row),(np.abs(fft_Z_row))/n_cols)
# column analysis
# ax3.plot(Z_data[:,col])
# ax4.plot(fftFreqZ_col,(np.abs(fft_Z_col))/n_rows)
ax[0,1].plot(Z_data[:,col])
ax[1,1].plot(np.abs(fftFreqZ_col),(np.abs(fft_Z_col))/n_rows)
plt.show()


