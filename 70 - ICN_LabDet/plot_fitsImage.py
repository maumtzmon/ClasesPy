from functions_py import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma

path='/home/oem/datosFits/MicrochipTest_Marzo/datos/14JUN23/proc_skp_m-009_microchip_T_170__Vv82_NSAMP_1_NROW_700_NCOL_700_EXPOSURE_10800_NBINROW_1_NBINCOL_1_img_8.fits'
hdu_list = fits.open(path)
print(hdu_list.info())
print('----------------')
# hdu_list[0].header
plt.figure(figsize=(20,10))

active=np.s_[:640,9:537]
all=np.s_[:,:]

data2plot=hdu_list[0].data-np.median(hdu_list[0].data)
plt.subplot(2,2,1)
plt.imshow(data2plot[active],vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(1))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')
Ext_1=np.flip(data2plot[active], 0)



data2plot=hdu_list[1].data-np.median(hdu_list[0].data)
plt.subplot(2,2,3)
plt.imshow(data2plot[active],vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(2))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')
Ext_2=data2plot[active]


data2plot=hdu_list[2].data-np.median(hdu_list[0].data)
plt.subplot(2,2,4)
plt.imshow(data2plot[active],vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(3))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')
Ext_3=np.flip(data2plot[active], 0)



data2plot=hdu_list[3].data-np.median(hdu_list[0].data)
plt.subplot(2,2,2)
plt.imshow(data2plot[active],vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(4))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')
Ext_3=np.flip(data2plot[active], 1)

plt.show()

plt.figure(figsize=(20,10))

data2plot=hdu_list[0].data-np.median(hdu_list[0].data)
Ext_1=np.flip(data2plot[active], 0)
plt.subplot(2,2,1)
plt.imshow(Ext_1,vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(1))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')

data2plot=hdu_list[1].data-np.median(hdu_list[0].data)
Ext_2=data2plot[active]
plt.subplot(2,2,3)
plt.imshow(Ext_2,vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(2))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')

data2plot=hdu_list[2].data-np.median(hdu_list[0].data)
Ext_3=np.flip(data2plot[active], 1)
plt.subplot(2,2,4)
plt.imshow(Ext_3,vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(3))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')


data2plot=hdu_list[3].data-np.median(hdu_list[0].data)
Ext_4=np.flip(np.flip(data2plot[active], 1),0)
plt.subplot(2,2,2)
plt.imshow(Ext_4,vmin=-800,vmax=800, origin='lower')
#plt.imshow(np.flip(data2plot,1),vmin=-800,vmax=800, origin='lower')
plt.title('CHID '+str(4))
plt.ylabel('Y_pix')
plt.xlabel('X_pix')

plt.show()

plt.figure(figsize=(20,10))

data2plot=hdu_list[0].data-np.median(hdu_list[0].data)
Ext_1=np.flip(data2plot[active], 0)

data2plot=hdu_list[1].data-np.median(hdu_list[0].data)
Ext_2=data2plot[active]

leftHalf=np.concatenate((Ext_1,Ext_2),0)

#plt.subplot(2,2,2)
plt.imshow(leftHalf,vmin=-800,vmax=800, origin='lower')

plt.show()