from functions_py import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
import skimage.measure as sk
import ana_connie_lib as ana
#from astropy.io import fits
path="/home/oem/Software/cursoInstrumentacion_2022/ClasesPy/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP200_25.fits"
hdu=fits.open(path)
ANSAMP=int(hdu[3].header.cards['ANSAMP'][1])
data=hdu[3].data
# plt.imshow(data)
# plt.show()
AA=np.s_[:,:1070]
OS=np.s_[:,1070:]

threshold=np.median(data[OS])+12002
label,num_objects=ndimage.label(data>threshold,structure=[[1,1,1],[1,1,1],[1,1,1]])

mask_inv=np.invert(label==0)
mx=ma.array(data,mask=mask_inv)
#mx=ma.array(data/ANSAMP,mask=mask_inv)
plt.imshow(mx)
plt.show()
center=np.median(mx[AA])
histograma, bins_x,_=plt.hist((mx[AA].flatten()),bins=500)#,range=(center-5000,center+8000))
plt.yscale("log")
plt.show()

ana.Noise(data, active_mask, iMCM=1, nCCDs=16, dataOK=True, doPlot=True, pdfname='None.pdf')