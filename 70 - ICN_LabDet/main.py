#!/usr/bin/python3
import sys
import pandas as pd
import os
import glob
import numpy as np
#from scipy import stats
from astropy.io import fits
import multiprocessing as mp
from ReconLib import *
#import math
#from scipy import ndimage as ndi
#from skimage import feature, measure
#import scipy.signal as sig

# from ReconLib import ClusterDict,ClusterRecon,PlotImage,maskedStatistics,WriteTXTCSVFiles,HistogramDict,SavePlot2File,saveObject2File

def main(argv):

#-----------------------------------------------------
#Files to be used
#-----------------------------------------------------

    RootFolder = argv[0].split('main.py')[0]
    folderpath = argv[1]
    validExtensions = [".fits"]
    if os.path.isdir(folderpath):
        if folderpath[-1] != '/':
            folderpath += '/'
    
        filesToSearch = '*.fits'
        FileNameL = glob.glob(folderpath + filesToSearch)

    elif os.path.isfile(folderpath):
        FileNameL = [folderpath]

    else:
        print(folderpath + " is not a valid directory or file")
        return 1

    dataL = []
    plotImageFlag = False
    CCDMaskL = []
    gradCCDMaskL = []
    serialRegisterEventsL = []
    CCDStatsL = []
    for FileName in FileNameL:
        FitsObj = fits.open(FileName)
        for Ext in range(0,len(FitsObj),1):
            dataL.append(FitsObj[Ext].data)

            if plotImageFlag == True:
                PlotImage(FitsObj[Ext].data)
            
            # serialRegisterEvents, CCDMask, gradCCDMask, CCDStats = makeSerialRegisterEventMask(FitsObj[Ext].data,n = 4,extend=True)
            serialRegisterEvents, CCDMask, gradCCDMask, CCDStats = makeSerialRegisterEventAdvancedMask(FitsObj[Ext].data,n = 4,extend=True,overscan=550,frames=15)
            CCDMaskL.append(CCDMask)
            gradCCDMaskL.append(gradCCDMask)
            serialRegisterEventsL.append(serialRegisterEvents)
            CCDStatsL.append(CCDStats)

    return 0

if __name__=="__main__":
    argv = sys.argv
    exitcode = main(argv)
    exit(exitcode)