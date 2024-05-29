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
import ana_connie_lib as ana
import pickle as pkl

plt.rcParams.update({
    "image.origin": "lower",
    "image.aspect": 1,
    #"text.usetex": True,
    "grid.alpha": .5,
    }) 

#Functions definition==============================================================================================


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




### Main Rutine
path_files='/home/oem/Software/cursoInstrumentacion_2022/ClasesPy/60 - MCM Test/'
listFiles=os. listdir(path_files) 
numberList = []
dict_list = {}
noise_list = []
gain_list = []
ser_list = []
ANSAMP_list=[]

for i in listFiles:
    if i.endswith('fits'):
        dict_list[(i.split('_')[-1]).split('.')[0]]=[i]
    else:
        continue

order2process=sorted(dict_list)  #generea lista con el orden en el que va a procesar los archivos
#order2process=['18']


for i in order2process:
    
    fileName=dict_list[i][0].split('.')[0]
    MCMNro=1
    outname='MCM'+str(MCMNro)+'_Demuxed_Test_'+re.sub(".fz","",fileName)+'.fits';
    dict_list[i].append(outname)
    dict_list[i].append({})
    step=16
    #Get 16-demuxed Images from the MCM
    #Primary HDU no data, only main header
    Primaryhdu_MCM = fits.PrimaryHDU() # Create primary HDU without data
    hdu_list_MCM = fits.HDUList([Primaryhdu_MCM]) #Create HDU list
    file_FZ=fits.open(path_files+'/'+dict_list[i][0])
    hdu_list_MCM[0].header=file_FZ[0].header
    NCOL=int(file_FZ[1].header["NCOL"]) #NCOL=int(h[1].header['NCOL'])
    NROW=int(file_FZ[1].header["NROW"])
    ANSAMP=int(file_FZ[1].header["ANSAMP"])
    dict_list[i][2]['NCOL']=NCOL
    dict_list[i][2]['NROW']=NROW
    dict_list[i][2]['ANSAMP']=ANSAMP
    ANSAMP_list.append(ANSAMP)

    for j in range(0,16):
        MapMCM_ColInit=MappingToMux[MappingToOHDUinfits[j]-1]  #REVISAR ACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaa
        datai=np.int32(ana.GetSingleCCDImage(file_FZ,hduUse,MapMCM_ColInit,NCOL,step,1))
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
        plt.imshow(hm[ext+1].data[overscan_mask]-np.median(hm[ext+1].data[overscan_mask]), vmin=-50, vmax=100)
        plt.title('MCM1 – ohdu = {:d}'.format(ext+1))
    #plt.show()
    plt.close()


    ohdusOK=True

    file_FITS=fits.open(dict_list[i][1])

    hmb = hm                    #se sobre escribe el dato, 
    for k in range(nCCDs):      #el valor de la imagen sera el de la imagen cruda menos la mediana
        hmb[k+1].data = (hm[k+1].data.astype('float64') - np.median(hm[k+1].data[overscan_mask], axis=1, keepdims=True))/ANSAMP
        
    f,(ax1,ax2) =plt.subplots(1,2,figsize=(15,30))
        #OVER SCAN#
    if int(ANSAMP)<10:
        hist,bins,_=ax1.hist(hmb[1].data[overscan_mask].flatten(), bins=100, range=(-50,100))
        x=(bins[1:]+bins[:-1])/2
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax1.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    elif int(ANSAMP)<20:
        hist,bins,_=ax1.hist(hmb[1].data[overscan_mask].flatten(), bins=500, range=(-50,100))
        x=(bins[1:]+bins[:-1])/2
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax1.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    elif int(ANSAMP)<30:
        hist,bins,_=ax1.hist(hmb[1].data[overscan_mask].flatten(), bins=1000, range=(-50,100))
        x=(bins[1:]+bins[:-1])/2
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax1.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    else:
        hist,bins,_=ax1.hist(hmb[1].data[overscan_mask].flatten(), bins=1000, range=(-50,100))
        x=(bins[1:]+bins[:-1])/2    
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax1.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    ax1.set_title("OS"+" ANSAMP ="+str(ANSAMP))
    ax1.set_xlabel("ADUs")
    ax1.legend()
    
            ##AREA ACTIVA##
    
    if int(ANSAMP)<10:
        hist,bins,_=ax2.hist(hmb[1].data[active_mask].flatten(), bins=100, range=(-20,100))
        x=(bins[1:]+bins[:-1])/2
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax2.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    elif int(ANSAMP)<20:
        hist,bins,_=ax2.hist(hmb[1].data[active_mask].flatten(), bins=500, range=(-50,100))
        x=(bins[1:]+bins[:-1])/2
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax2.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    elif int(ANSAMP)<30:
        hist,bins,_=ax2.hist(hmb[1].data[active_mask].flatten(), bins=1000, range=(-50,100))
        x=(bins[1:]+bins[:-1])/2
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax2.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    else:
        hist,bins,_=ax2.hist(hmb[1].data[active_mask].flatten(), bins=1000, range=(-50,100))
        x=(bins[1:]+bins[:-1])/2    
        popt,pcov=curve_fit(ana.gaussian1,x,hist)#,p0=[0,50,1000])
        popt=abs(popt)
        ax2.plot(x,ana.gaussian1(x,*popt),label="Gauss Fit $\sigma=$"+str(popt[1])+"\n mean="+str(popt[0]))
    ax2.set_title("AA"+" ANSAMP ="+str(ANSAMP))
    ax2.set_xlabel("ADUs")
    ax2.legend()
    plt.show()
    #plt.close()
    


    #noise_list.append(ana.Noise(hmb, overscan_mask, MCMNro, nCCDs, ohdusOK, doPlot=False, pdfname='None.pdf'))
    #gain_list.append(ana.Gain(hmb, overscan_mask, MCMNro, nCCDs, ohdusOK, doPlot=False, pdfname='None.pdf'))
    #h, active_mask, iMCM, nCCDs, dataOK, gain, doPlot, pdfname, itera=10, thr=5
    #ser_list.append(ana.Ser(hmb, active_mask, MCMNro, nCCDs, ohdusOK, gain_list[-1],doPlot=False, pdfname='None.pdf'))


#analysisDict={'noise':noise_list,'gain':gain_list,'ser':ser_list, 'ANSAMP':ANSAMP_list}

#filePkl=open('dataAnalysis.pkl','wb')
#pkl.dump(analysisDict,filePkl)
#filePkl.close()
