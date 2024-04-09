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
    "figure.subplot.hspace":.5,
    "figure.subplot.top":.9,
    "figure.subplot.left":0.089,
    "figure.subplot.right":0.964,
    }) 

filePkl=open('dataAnalysis.pkl','rb')
dataAnalysis=pkl.load(filePkl)

noise_list=dataAnalysis['noise']
gain_list=dataAnalysis['gain']
ser_list=dataAnalysis['ser']
ANSAMP_list=dataAnalysis['ANSAMP']
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

plotList=[0,0,1,1,0] #[NoiseAcrossANSAMP,Noise, GainAcrossANSAMP, Gain, , ,]

#"Noise across the ANSAMP"
if plotList[0]==1:
    fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
    fig.suptitle("Noise across the ANSAMP")
    i=0
    for nrow in axs:
        for ncol in nrow:
            ncol.scatter(range(1,17),noise_list[i])
            ncol.set_title('ANSAMP = {:d}'.format(ANSAMP_list[i]))
            # nrow.scatter(ANSAMP_list,noise_trans[i])
            # nrow.set_xscale('log')
            i+=1
        nrow[0].set_ylabel('Noise [ADU]')
    for ncol in axs[-1]:
        ncol.set_xlabel('OHDU')
    plt.show()

#"Noise[ADUs] by OHDU"
if plotList[1]==1:
    fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
    fig.suptitle("Noise[ADUs] by OHDU")
    i=0
    for nrow in axs:
        for ncol in nrow:
            ncol.scatter(ANSAMP_list,noise_trans[i])
            # ncol.plot(ANSAMP_list,noise_trans[i])
            ncol.set_xscale('log')
            
            ncol.set_title('OHDU = {:d}'.format(i+1))
            i+=1
        nrow[0].set_ylabel('Noise [ADU]')
    
    for ncol in axs[-1]:
        ncol.set_xlabel('ANSAMP')
        
    plt.show()

#"Gain across the ANSAMP"
if plotList[2]==1:
    fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
    fig.suptitle("Gain across the ANSAMP")
    i=0
    for nrow in axs:
        for ncol in nrow:
            ncol.scatter(range(1,17),gain_list[i])
            ncol.set_title('ANSAMP = {:d}'.format(ANSAMP_list[i]))
            # nrow.scatter(ANSAMP_list,noise_trans[i])
            # nrow.set_xscale('log')
            i+=1
        nrow[0].set_ylabel('Gain [ADU]')
    for ncol in axs[-1]:
        ncol.set_xlabel('OHDU')
    plt.show()

#"Gain [ADUs]"
if plotList[3]==1:
    fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
    fig.suptitle("Gain [ADUs]")
    i=0
    for nrow in axs:
        for ncol in nrow:
            ncol.scatter(ANSAMP_list,gain_trans[i])
            ncol.set_xscale('log')
            
            ncol.set_title('OHDU = {:d}'.format(i+1))
            i+=1
        nrow[0].set_ylabel('Gain[ADU]')
    
    for ncol in axs[-1]:
        ncol.set_xlabel('ANSAMP')
        
    plt.show()
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

print("all ok")


# from matplotlib import pyplot as plt

# y=[0,1,2,3,4,5,6,7,8,9]
# x=[18,19,20,21,22,23,24,25,26,27]

# fig, axs = plt.subplots(ncols=4,nrows=4,figsize=(12,8))
# for ncol in axs:
#     for nrow in ncol:
#         nrow.scatter(x,y)
# plt.show()