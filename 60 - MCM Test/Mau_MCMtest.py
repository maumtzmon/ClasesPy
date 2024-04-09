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
path_files='/home/mauricio/datosFits/mcm_data/ansamp'
listFiles=os. listdir(path_files) 
numberList = []
dict_list = {}
noise_list = []
gain_list = []
ser_list = []
ANSAMP_list=[]

for i in listFiles:
    dict_list[(i.split('_')[-1]).split('.')[0]]=[i]

order2process=sorted(dict_list)
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
        plt.imshow(hm[ext+1].data[overscan_mask]-np.median(hm[ext+1].data[overscan_mask]), vmin=-50, vmax=50)
        plt.title('MCM1 – ohdu = {:d}'.format(ext+1))
    #plt.show()
    plt.close()


    ohdusOK=True

    file_FITS=fits.open(dict_list[i][1])

    hmb = hm                    #se sobre escribe el dato, 
    for i in range(nCCDs):      #el valor de la imagen sera el de la imagen cruda menos la mediana
        hmb[i+1].data = (hm[i+1].data.astype('float64') - np.median(hm[i+1].data[overscan_mask], axis=1, keepdims=True))/ANSAMP

    #noise_list.append(ana.Noise(hmb, overscan_mask, MCMNro, nCCDs, ohdusOK, doPlot=False, pdfname='None.pdf'))
    gain_list.append(ana.Gain(hmb, overscan_mask, MCMNro, nCCDs, ohdusOK, doPlot=False, pdfname='None.pdf'))
    #h, active_mask, iMCM, nCCDs, dataOK, gain, doPlot, pdfname, itera=10, thr=5
    #ser_list.append(ana.Ser(hmb, active_mask, MCMNro, nCCDs, ohdusOK, gain_list[-1],doPlot=False, pdfname='None.pdf'))



analysisDict={'noise':noise_list,'gain':gain_list,'ser':ser_list, 'ANSAMP':ANSAMP_list}

filePkl=open('dataAnalysis.pkl','wb')
pkl.dump(analysisDict,filePkl)
filePkl.close()

# print("all ok")

# nccds=16     #Se genera lista con las ganancias de cada canal con todos los nsamps
# nnsamp=16          # es decir gainForNsamp=[[Ganancia canal 1 con nsamp1,2,3,4,...n ],[Ganancia canal 2 con nsamp1,2,3,4,...n],..,[Ganancia canal 16 con nsamp1,2,3,4,...n]]
# chipList=[]
# noise_trans=[]
# for i in range(nccds):
#     chipList=[]
#     for j in range(nnsamp):
#         noiseChip=noise_list[j][i]
#         chipList.append(noiseChip)
#     noise_trans.append(chipList)
        
# gain_trans=[]
# for i in range(nccds):
#     chipList=[]
#     for j in range(nnsamp):
#         gainChip=gain_list[j][i]
#         chipList.append(gainChip)
#     gain_trans.append(chipList)


# ser_trans=[]
# for i in range(nccds):
#     chipList=[]
#     for j in range(nnsamp):
#         serChip=ser_list[j][i]
#         chipList.append(serChip)
#     ser_trans.append(chipList)

# print("all ok")

# fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
# i=0
# for ncol in axs:
#     for nrow in ncol:
#         #nrow.scatter(order2process,noise_list[i])
#         nrow.scatter(ANSAMP_list,noise_trans[i])
#         i+=1
# plt.show()

# fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
# i=0
# for ncol in axs:
#     for nrow in ncol:
#         #nrow.scatter(order2process,gain_list[i])
#         nrow.scatter(ANSAMP_list,gain_trans[i])
#         i+=1
# plt.show()

# fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
# i=0
# for ncol in axs:
#     for nrow in ncol:
#         #nrow.scatter(order2process,ser_list[i])
#         nrow.scatter(ANSAMP_list,ser_trans[i])
#         i+=1
# plt.show()

# print("all ok")


# from matplotlib import pyplot as plt

# y=[0,1,2,3,4,5,6,7,8,9]
# x=[18,19,20,21,22,23,24,25,26,27]

# fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
# for ncol in axs:
#     for nrow in ncol:
#         nrow.scatter(x,y)
# plt.show()