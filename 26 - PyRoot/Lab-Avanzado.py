from ROOT  import TF1, TH1F, TCanvas#En el ICN no funciona

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

import sys

sys.path.insert(0, '/home/mauricio/Software/Serial_Register_Events_Detection')

from ReconLib import *

plt.rcParams.update({
    "image.origin": "lower",
    "image.aspect": 1,
    #"text.usetex": True,
    "grid.alpha": .5,
    "axes.linewidth":2,
    "lines.linewidth" : 1,
    "font.size":    15.0,
    "xaxis.labellocation": 'right',  # alignment of the xaxis label: {left, right, center}
    "yaxis.labellocation": 'top',  # alignment of the yaxis label: {bottom, top, center}
    "xtick.top":           True ,  # draw ticks on the top side
    "xtick.major.size":    8    ,# major tick size in points
    "xtick.minor.size":    4      ,# minor tick size in points
    "xtick.direction":     'in',
    "xtick.minor.visible": True,
    "ytick.right":           True ,  # draw ticks on the top side
    "ytick.major.size":    8    ,# major tick size in points
    "ytick.minor.size":    4      ,# minor tick size in points
    "ytick.direction":     'in',
    "ytick.minor.visible": True,
    "ytick.major.width":   2   , # major tick width in points
    "ytick.minor.width":   1 ,
    "xtick.major.width":   2   , # major tick width in points
    "xtick.minor.width":   1 ,
    "legend.framealpha": 0 ,
    "legend.loc": 'best',

})


path='/home/oem/datosFits/vFile_labAvanzado/proc_skp_m-009_microchip_vtested_T_160__seq__NSAMP_1_NROW_650_NCOL_700_EXPOSURE_600_NBINROW_1_NBINCOL_1_img_8.fits'
hdu_list = fits.open(path)
print(hdu_list.info())
print('----------------')
# hdu_list[0].header
plt.figure(figsize=(20,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(hdu_list[i].data-np.median(hdu_list[i].data),vmin=-800,vmax=800,cmap='gray')
    plt.title('CHID '+str(i))
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')
plt.show()

lo = -200
hi =700
## Definicion de la variable TFunction 
fgaus1 = TF1("fgauss1","[0]*exp(-0.5*((x-[1])/[2])^2)",lo,hi) # TF1("nombre", "funcion escrita como en root", min, max, #parametros)
#fgaus2 = TF1("fgauss2","[3]*exp(-0.5*((x-[0])/[1])^2)+[4]*exp(-0.5*((x-[0]-[2])/[1])^2)",-200,700,5) # TF1("nombre", "funcion escrita como en root", min, max, #parametros)


#serialRegisterEvents_n5, CCDMask, _, _=makeSerialRegisterEventAdvancedMask(hdu_list[0].data,n = 5,extend=True,overscan=550,frames=15)

#la_mascara=np.ma.masked_array(hdu_list[0].data, mask=serialRegisterEvents_n5)
ROI=hdu_list[0].data[:,550:]
plt.figure(figsize=(10,10))
plt.imshow(ROI)
flatROI=ROI.flatten()


h3=TH1F("histogram", "ROI in OHDU 0",1000, flatROI.min(),flatROI.max())

for pixel_value in flatROI:
    if not np.ma.is_masked(pixel_value):
        h3.Fill(pixel_value)
        #print(pixel_value)

#fgaus1.SetParameters(0,50,lo,hi) # Establecer parametros iniciales del fit, de manera visual es posible determinarlos como una primera aproximacion
#h3.Fit(fgaus1)

c3=TCanvas()

h3.Draw()
c3.Draw()
#fgaus2.Draw("same")

# print("chiSquare= "+str(fgaus2.GetChisquare()))
# print("NDegrees of Freedom: "+str(fgaus2.GetNDF()))

input()
