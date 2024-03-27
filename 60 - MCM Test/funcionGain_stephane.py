

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
#Code based on AnaMCM_CONNIE.ipynb
def MCMGain(docName):
    plt.rcParams.update({
    "image.origin": "lower",
    "image.aspect": 1,
    #"text.usetex": True,
    "grid.alpha": .5, })

    def GetSingleCCDImage(hdul,LTA_channel,ColInit,NCOL,step,NrOfMCMs):
        #hdul: list of hdu of the muxed fit file
	    #LTA_channel
        #ColInit: First column in the muxed image
        #NCOL: Number of columns in the image
        MuxedImage=hdul[LTA_channel].data
        step2 = step*NrOfMCMs
        LastCol = ColInit+(int(NCOL)-1)*step2
        indexCol=list(range((ColInit-1),LastCol,step2))
        DeMuxedImage=MuxedImage[:, indexCol]
        return DeMuxedImage
 
    filePath='./'
    #fileName='mcm029_test__ANSAMP25_full_34.fz'
    fileName= docName
    #fileName='barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP20_20.fz'
    h=fits.open(fileName)
    h.info()
    #Get some header values
    ANSAMP=int(h[1].header['ANSAMP'])
    NCOL=int(h[1].header['NCOL'])
    print(ANSAMP,NCOL)
    #Channel mapping between MCM and Mux in the front-end==============================================================
    #MCM=>FLEX=>idb=>Front end electronics
    # 1  5  9 13
    # 2  6 10 14 => to MCM flex
    # 3  7 11 15 => to MCM flex
    # 4  8 12 16
    MappingToMux=[8,7,6,16,15,14,13,12,11,10,9,5,4,3,2,1] #SENSEICOPY Mapping from MCM positions to MUX inputs.CCD1=>Mux S8, CCD2=>Mux S7, CCD3=>Mux S6,...CCD16=>Mux S1 
    #MappingToMux=[1,3,2,4,5,9,10,11,12,13,14,15,16,7,6,8];  #Old mapping with idb and 50 pins front end:
    MappingToOHDUinfits=[1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16] #Mapping from Mux inputs to OHDU positions in fits file
    hduUse=2

    MCMNro=1
    outname=re.sub(".fz","",fileName)+"MCM"+str(MCMNro)+"_Demuxed_"+'.fits';
    pdfname=re.sub(".fz","",fileName)+'MCM'+str(MCMNro)+'_Demuxed_'
    step=16
#Get 16-demuxed Images from the MCM
#Primary HDU no data, only main header
    Primaryhdu_MCM = fits.PrimaryHDU() # Create primary HDU without data
    hdu_list_MCM = fits.HDUList([Primaryhdu_MCM]) #Create HDU list
    hdu_list_MCM[0].header=h[0].header
    for i in range(0,16):
        MapMCM_ColInit=MappingToMux[MappingToOHDUinfits[i]-1]  #REVISAR ACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaa!
        datai=np.int32(GetSingleCCDImage(h,hduUse,MapMCM_ColInit,NCOL,step,1))
	        #print(datai.dtype)
        image_hdu=fits.ImageHDU(datai);
        image_hdu.header=h[hduUse].header #Repear the header of the used hdu channel into all the 16-channles
        image_hdu.header.set('NSAMP',ANSAMP)
        hdu_list_MCM.append(image_hdu)
	
    #save demux file
    hdu_list_MCM.writeto(outname,overwrite=True);
    hdu_list_MCM.close()

    hm = hdu_list_MCM
    hmb = hdu_list_MCM
    
#print(hm[1].header)

# Some infos hmb = hdu_list_MCMfrom the header:
    CCDNROW = hm[1].header["CCDNROW"]
    CCDNCOL = hm[1].header["CCDNCOL"]
    NROW = hm[1].header["NROW"]
    NCOL = hm[1].header["NCOL"]
    NSAMP = hm[1].header["NSAMP"]
# MCMs/
    ANSAMP = hm[1].header["ANSAMP"]
    nCCDs=16

# define active and overscan masks
    active_mask = np.s_[:, 10:1057] # 
    overscan_mask = np.s_[:, -91:-1]
    start = parse(hm[1].header["DATESTART"]+"Z").timestamp()
    end = parse(hm[1].header["DATEEND"]+"Z").timestamp()
    deltatime = datetime.timedelta(seconds=end-start)
    readout_time = deltatime.total_seconds()
	
    def Baseline(h, overscan_mask, iMCM, nCCDs, doPlot, pdfname):
        mediana = []
        plt.figure(figsize=(15,6))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for i in range(0,nCCDs):
            m = np.median(h[i+1].data[overscan_mask], axis=1,keepdims=True)
            mediana.append(m)
            x = np.arange(len(m))
            plt.plot(x,m-m[0],label='ohdu = {:d} - ref = {}'.format(i+1,m[0]))
            plt.xlim(0,len(m)*1.30)
            plt.xlabel('iROW', fontsize=14)
            plt.ylabel('Baseline (ADUs)', fontsize=14)
            plt.title('MCM {:d}'.format(iMCM),fontsize=16)
            plt.legend(fontsize=12)
            # to save the plot
        if doPlot:
            pdf_filename = f'{pdfname}_baseline.pdf'
            plt.savefig(pdf_filename, format='pdf')
        plt.close()
        return mediana
	
    doPlot=1
    # made the baseline plot and print the baseline diffetences
    baseline = Baseline(hm, overscan_mask, MCMNro, nCCDs, doPlot, pdfname)
    #
    hmb = hm
    for i in range(nCCDs):
        hmb[i+1].data = hm[i+1].data.astype('float64') - np.median(hm[i+1].data[overscan_mask], axis=1, keepdims=True)
    
    ohdusOK = np.zeros((16,), dtype=int)
   # plt.figure(figsize=(16,16))
    for i in range(16):
      #  plt.subplot(4,4,i+1)
        y, x, _ = plt.hist(hmb[i+1].data[active_mask].flatten(),100,density=True, histtype='step', cumulative=True)
        #print(i,x.max())
        plt.close()
        if x.max()>100: ohdusOK[i]=1
        
    
    def gaussian2(x,m1,s,a1,g,a2):
        return a1*np.exp(-1/2*((x-m1)/s)**2)+a2*np.exp(-1/2*((x-m1-g)/s)**2)
    
    def Gain(h, active_mask, iMCM, nCCDs, dataOK, doPlot, pdfname):
        gain = []
        plt.figure(figsize=(24,24))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title("MCM {:d}".format(iMCM), fontsize=18)
        for i in range(nCCDs):
            if dataOK[i]:
                plt.subplot(4,4,i+1)
                y,xb=np.histogram(h[i+1].data[active_mask].flatten(), bins=np.linspace(-100,3500,300))
                x=(xb[1:]+xb[:-1])/2
                plt.plot(x,y,label='MCM {:d} – ohdu = {:d}'.format(iMCM,i+1))
                # gaussian2 fit
                try:
                    popt,pcov=curve_fit(gaussian2,x,y,p0=[-10,60,5000, 1200, 1000])
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
            pdf_filename = f'{pdfname}_gain.pdf'
            plt.savefig(pdf_filename, format='pdf')
        plt.close()
        return gain
    gain = Gain(hmb, active_mask, MCMNro, nCCDs, ohdusOK, doPlot, pdfname)
    return gain