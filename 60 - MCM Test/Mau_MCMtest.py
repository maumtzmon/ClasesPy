import os
import time
import subprocess
import math
import fileinput
import sys
from astropy.io import fits 
import numpy as np
import matplotlib.pyplot as plt
import re
from dateutil.parser import parse
import datetime
from scipy.optimize import curve_fit
from scipy import ndimage


plt.rcParams.update({
    "image.origin": "lower",
    "image.aspect": 1,
    #"text.usetex": True,
    "grid.alpha": .5,
    }) 

#Functions definition==============================================================================================
def GetSingleCCDImage(hdul,LTA_channel,ColInit,NCOL,step,NrOfMCMs):
	#hdul: list of hdu of the muxed fit file
	#LTA_channel
	#ColInit: First column in the muxed image
	#NCOL: Number of columns in the image
	MuxedImage=hdul[LTA_channel].data
	step2=step*NrOfMCMs
	LastCol=ColInit+(int(NCOL)-1)*step2
	indexCol=list(range((ColInit-1),LastCol,step2))
	DeMuxedImage=MuxedImage[:, indexCol]
	return DeMuxedImage #return demuxed image


# ------------------------------------------------------------------------------
# 
def Noise(h, overscan_mask, iMCM, nCCDs, dataOK=True, doPlot=False, pdfname='noise'):
    noise = []
    plt.figure(figsize=(24,24))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("MCM {:d}".format(iMCM), fontsize=18)
    for i in range(nCCDs):
        #if dataOK[i]:
        if dataOK:
            plt.subplot(4,4,i+1)
            y,xb=np.histogram(h[i+1].data[overscan_mask].flatten(), bins=np.linspace(-250,250,200))
            x=(xb[1:]+xb[:-1])/2
            plt.plot(x,y,label='MCM {:d} – ohdu = {:d}'.format(iMCM,i+1))
            # gaussian fit
            try:
                popt,pcov=curve_fit(gaussian1,x,y,p0=[-10,100,10000])
                plt.plot(x,gaussian1(x,*popt),label="Gauss Fit $\sigma$: {:.3f} ADUs".format(popt[1]))
                noise.append(popt[1])
            except RuntimeError:
                print("Error - gain fit failed" + pdfname)
                noise.append(-1)
            plt.legend(fontsize=13)
            plt.xlabel("Charge [ADUs]",fontsize=12)
            plt.yscale("log")
            plt.ylabel("Entries",fontsize=12)
        else: noise.append(-1)
    # to save the plot
    #pdf_filename = f'noise_'+pdfname+'_{iMCM+1}.pdf'i
    if doPlot:
        #pdf_filename = f'noise_{pdfname}.pdf'
        #plt.savefig(pdf_filename, format='pdf')
        plt.show()
    elif doPlot == False:
        plt.close()
    else:
        plt.close()
    return noise
# ------------------------------------------------------------------------------

def Gain(h, active_mask, iMCM, nCCDs, dataOK, doPlot, pdfname):
    gain = []
    plt.figure(figsize=(24,24))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("MCM {:d}".format(iMCM), fontsize=18)
    for i in range(nCCDs):
        if dataOK:
            plt.subplot(4,4,i+1)
            y,xb=np.histogram(h[i+1].data[active_mask].flatten(), bins=np.linspace(-100,500,300))
            x=(xb[1:]+xb[:-1])/2
            plt.plot(x,y,label='MCM {:d} – ohdu = {:d}'.format(iMCM,i+1))
            # gaussian2 fit
            try:
                popt,pcov=curve_fit(gaussian2,x,y,p0=[-10,60,10000, 300, 1000])
                plt.plot(x,gaussian2(x,*popt),label="Gain: {:.3f} ADUs/e-".format(popt[3]))
                gain.append(popt[3])
            except RuntimeError:
                print("Error - gain fit failed" + pdfname)
                gain.append(-1)
            plt.legend(fontsize=13)
            plt.xlabel("Charge [ADUs]",fontsize=12)
            plt.yscale("log")
            plt.ylabel("Entries",fontsize=12)
        else: gain.append(-1)
    # to save the plot
    if doPlot:
        pdf_filename = f'gain_{pdfname}.pdf'
        plt.savefig(pdf_filename, format='pdf')
    plt.close()
    return gain
# ------------------------------------------------------------------------------

def Ser(h, active_mask, iMCM, nCCDs, dataOK, gain, doPlot, pdfname, itera=10, thr=5):
    ser = []
    plt.figure(figsize=(24,24))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("MCM {:d}".format(iMCM), fontsize=18)
    for i in range(nCCDs):
        if dataOK:
            data = h[i+1].data/gain[i]
            event_mask = data > thr
            event_halo_mask = ndimage.binary_dilation(
                              event_mask,
                              iterations = itera,
                              structure = ndimage.generate_binary_structure(rank=2, connectivity=2))
            dataMasked = np.where(event_halo_mask, np.nan, data )
            #mask = ndimage.binary_dilation(data>thr,iterations=itera,structure=[[1,1,1],[1,1,1],[1,1,1]])
            #dataMasked = data - 1000000*mask.astype(data.dtype)
            #dataMasked = np.ma.masked_less(dataMasked, -50000)
            plt.subplot(4,4,i+1)
            y, xb = np.histogram(dataMasked[active_mask].flatten(),range=[-0.5,2.5],bins=200)
            x = (xb[1:]+xb[:-1])/2
            plt.plot(x, y,label='MCM {:d} – ohdu = {:d}'.format(iMCM,i+1))
            try:
                popt, pcov = curve_fit(convolution, x, y, p0=[-0.4, 0.2, 1000, 0.1])
                plt.plot(x, convolution(x, *popt), label="Noise: {:.3f}  SER: {:.4f} ".format(abs(popt[1]),popt[3]),color='red')
                if popt[3]>0 and popt[3]<100:
                    ser.append(popt[3])
                else: ser.append(-1)
            except RuntimeError:
                print("Error - convolution fit failed " + pdfname)
                ser.append(-1)
            plt.xlabel("e-",fontsize=12)
            plt.ylabel("Entries",fontsize=12)
            plt.yscale("log")
            plt.legend(fontsize=13)
        else: ser.append(-1)
    # to save the ploti
    if doPlot:
        pdf_filename = f'ser_{pdfname}.pdf'
        plt.savefig(pdf_filename, format='pdf')
    plt.close()
    return ser
# -------------------------------------------------------------------------


#
#Channel mapping between MCM and Mux in the front-end==============================================================
#MCM=>FLEX=>idb=>Front end electronics
# 1  5  9 13
# 2  6 10 14 => to MCM flex
# 3  7 11 15 => to MCM flex
# 4  8 12 16
MappingToMux=[8,7,6,16,15,14,13,12,11,10,9,5,4,3,2,1] #SENSEICOPY Mapping from MCM positions to MUX inputs.CCD1=>Mux S8, CCD2=>Mux S7, CCD3=>Mux S6,...CCD16=>Mux S1 
#MappingToMux=[1,3,2,4,5,9,10,11,12,13,14,15,16,7,6,8];  #Old mapping with idb and 50 pins front end:
MappingToOHDUinfits=[1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16] #Mapping from Mux inputs to OHDU positions in fits file

hduUse=2  #ccd in use  Default variable DON'T TOUCH

active_mask = np.s_[:, 10:1057] # 
overscan_mask = np.s_[:, -91:-1]

nCCDs=16

# ------------------------------------------------------------------------------
#
# ----  Single Gaussian for the noise computation 
def gaussian1(x,m1,s1,a1):
    return a1*np.exp(-1/2*((x-m1)/s1)**2)
#
# ---- Two Gaussians shifter by g = gain 
def gaussian2(x,m1,s,a1,g,a2):
    return a1*np.exp(-1/2*((x-m1)/s)**2)+a2*np.exp(-1/2*((x-m1-g)/s)**2)
#
# ---- 
def convolution(x, mu, sigma, A, lamb, Nmax=10):
    y = 0.
    for i in range(0, Nmax+1):
        y += (lamb**i)/float(math.factorial(i)) * \
            np.exp(-0.5*((x-i-mu)/float(sigma))**2)
    return A*np.exp(-lamb)*y/(np.sqrt(2*np.pi*sigma**2))
#
# ------------------------------------------------------------------------------


### Main Rutine
path_files='/home/oem/datosFits/mcm_data/ansamp'
listFiles=os. listdir(path_files) 
numberList = []
dict_list = {}
noise_list = []
gain_list = []
ser_list = []

for i in listFiles:
    dict_list[(i.split('_')[-1]).split('.')[0]]=[i]

order2process=sorted(dict_list)
#order2process=['18']
for i in order2process:
    fileName=dict_list[i][0].split('.')[0]
    MCMNro=1
    outname='MCM'+str(MCMNro)+'_Demuxed_Test_'+re.sub(".fz","",fileName)+'.fits';
    dict_list[i].append(outname)
    step=16
    #Get 16-demuxed Images from the MCM
    #Primary HDU no data, only main header
    Primaryhdu_MCM = fits.PrimaryHDU() # Create primary HDU without data
    hdu_list_MCM = fits.HDUList([Primaryhdu_MCM]) #Create HDU list
    file_FZ=fits.open(path_files+'/'+dict_list[i][0])
    hdu_list_MCM[0].header=file_FZ[0].header
    NCOL=file_FZ[1].header["NCOL"] #NCOL=int(h[1].header['NCOL'])
    ANSAMP=file_FZ[1].header["ANSAMP"]
    for j in range(0,16):
        MapMCM_ColInit=MappingToMux[MappingToOHDUinfits[j]-1]  #REVISAR ACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaa
        datai=np.int32(GetSingleCCDImage(file_FZ,hduUse,MapMCM_ColInit,NCOL,step,1))
        #print(datai.dtype)
        image_hdu=fits.ImageHDU(datai);
        image_hdu.header=file_FZ[hduUse].header #Repear the header of the used hdu channel into all the 16-channles
        image_hdu.header.set('NSAMP',ANSAMP)
        hdu_list_MCM.append(image_hdu)

 


    hdu_list_MCM.writeto(outname,overwrite=True)
    hdu_list_MCM.close()

    hm=hdu_list_MCM
    
    
    plt.figure(figsize=(24,8))
    for ext in range(nCCDs):
        plt.subplot(4,4,ext+1)
        plt.imshow(hm[ext+1].data[overscan_mask]-np.median(hm[ext+1].data[overscan_mask]), vmin=-50, vmax=50)
        plt.title('MCM1 – ohdu = {:d}'.format(ext+1))
    #plt.show()
    plt.close()


    ohdusOK=True

    file_FITS=fits.open(dict_list[i][1])

    hmb = hm                    #se sobre escribe el dato, 
    for i in range(nCCDs):      #el valor de la imagen sera el de la imagen cruda menos la mediana
        hmb[i+1].data = hm[i+1].data.astype('float64') - np.median(hm[i+1].data[overscan_mask], axis=1, keepdims=True)

    noise_list.append(Noise(hmb, overscan_mask, MCMNro, nCCDs, ohdusOK, doPlot=False, pdfname='None.pdf'))
    gain_list.append(Gain(hmb, overscan_mask, MCMNro, nCCDs, ohdusOK, doPlot=False, pdfname='None.pdf'))
    #h, active_mask, iMCM, nCCDs, dataOK, gain, doPlot, pdfname, itera=10, thr=5
    ser_list.append(Ser(hmb, active_mask, MCMNro, nCCDs, ohdusOK, gain_list[-1],doPlot=False, pdfname='None.pdf'))

print("all ok")

nccds=16     #Se genera lista con las ganancias de cada canal con todos los nsamps
nnsamp=16          # es decir gainForNsamp=[[Ganancia canal 1 con nsamp1,2,3,4,...n ],[Ganancia canal 2 con nsamp1,2,3,4,...n],..,[Ganancia canal 16 con nsamp1,2,3,4,...n]]
chipList=[]
noise_trans=[]
for i in range(nccds):
    chipList=[]
    for j in range(nnsamp):
        noiseChip=noise_list[j][i]
        chipList.append(noiseChip)
    noise_trans.append(chipList)
        
gain_trans=[]
for i in range(nccds):
    chipList=[]
    for j in range(nnsamp):
        gainChip=gain_list[j][i]
        chipList.append(gainChip)
    gain_trans.append(chipList)


ser_trans=[]
for i in range(nccds):
    chipList=[]
    for j in range(nnsamp):
        serChip=ser_list[j][i]
        chipList.append(serChip)
    ser_trans.append(chipList)

print("all ok")

fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
i=0
for ncol in axs:
    for nrow in ncol:
        nrow.scatter(order2process,noise_list[i])
        i+=1
plt.show()

fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
i=0
for ncol in axs:
    for nrow in ncol:
        nrow.scatter(order2process,gain_trans[i])
        i+=1
plt.show()

fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
i=0
for ncol in axs:
    for nrow in ncol:
        nrow.scatter(order2process,ser_trans[i])
        i+=1
plt.show()

print("all ok")


# from matplotlib import pyplot as plt

# y=[0,1,2,3,4,5,6,7,8,9]
# x=[18,19,20,21,22,23,24,25,26,27]

# fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
# for ncol in axs:
#     for nrow in ncol:
#         nrow.scatter(x,y)
# plt.show()