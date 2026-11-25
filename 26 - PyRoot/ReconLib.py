import os
import pandas as pd
import numpy as np
from skimage import feature, measure
import cv2
import numpy.ma as ma
from scipy.ndimage.measurements import center_of_mass
from matplotlib import pyplot as plt
import pickle
import sqlite3
from astropy.io import fits as fits
from sqlite3 import Error
from joblib import Parallel, cpu_count

import pickle


def parallelAnalysis(nameString,nL,ignoreExtL,OSX,medianImage,numberOfFrames=10):
    resultL = []
    FileName, filenumber = nameString.split(",")
    FitsObj = fits.open(FileName)

    for Ext in range(0,len(FitsObj),1):
        if Ext not in ignoreExtL:
            for nvar in nL:
              
                xsizeAR, ysizeAR = FitsObj[Ext].data.shape
                xsizeOS, ysizeOS = FitsObj[Ext].data[:,OSX:].shape
                dataProcceced = FitsObj[Ext].data-medianImage[Ext]
                serialRegisterEvents, CCDMask, gradCCDMask, CCDStats = makeSerialRegisterEventAdvancedMask(dataProcceced,n = nvar,extend=True,overscan=OSX,frames=numberOfFrames)
                if OSX > 0:
                    labelSRE_OS = measure.label(serialRegisterEvents[:,OSX:], connectivity=2, background=0)
                    labelSRE_AA = measure.label(serialRegisterEvents[:,:OSX], connectivity=2, background=0)
                    labelSRE_all = measure.label(serialRegisterEvents, connectivity=2, background=0)
                    numSRE_AAOS = int(np.sum(serialRegisterEvents[:,OSX]*serialRegisterEvents[:,OSX-1]))
                    # ratioA = (xsizeAR*(ysizeAR-ysizeOS))/(xsizeOS*ysizeOS)
                    numSRE_OS = np.max(labelSRE_OS)
                    numSRE_AA = np.max(labelSRE_AA)
                    numSRE_all = np.max(labelSRE_all)
                else:
                    labelSRE_AA = measure.label(serialRegisterEvents[:,:OSX+1], connectivity=2, background=0)
                    numSRE_AA = np.max(labelSRE_AA)
                    numSRE_OS = 0
                    numSRE_all = numSRE_AA
                    numSRE_AAOS = 0

                if numSRE_all == numSRE_AA + numSRE_OS - numSRE_AAOS:
                    countIsCorrect = 1
                else:
                    countIsCorrect = 0
                # numSRE_A = numSRE_all - numSRE_OS
                print("{},{},{}".format(filenumber,Ext,nvar))
                resultL.append([filenumber,Ext,nvar,xsizeAR,ysizeAR,xsizeOS,ysizeOS,numSRE_all,numSRE_AA,numSRE_OS,numSRE_AAOS,CCDStats[0],CCDStats[1],countIsCorrect])

    return resultL


def numberOfJobs(numberOfFiles,machine="PC",maxjobs=4):
    if machine == "PC":
        return maxjobs
    else:
        cores = cpu_count()
        if numberOfFiles >= cores:
            return cores
        else:
            return numberOfFiles

def extendSerialRegisterMask(serialRegisterEvents,minlength = 20, OS = None):
    
    ylen, xlen = np.shape(serialRegisterEvents)
    
    # if OS != None:
    #     ylen, xlen = np.shape(serialRegisterEvents)
    #     xlen = OS + 1
    # else:
    #     ylen, xlen = np.shape(serialRegisterEvents)


    for i in range(0,ylen-1,1):
            if np.any(serialRegisterEvents[i,:]):
                numberOfPixels = np.sum(serialRegisterEvents[i,:])
                if numberOfPixels == 1:
                    serialRegisterEvents[i,:] = np.where(serialRegisterEvents[i,:] == 1,0,serialRegisterEvents[i,:])
                else:
                    array = np.where(serialRegisterEvents[i,:] == 1,np.arange(0,xlen,1),np.infty)
                    xmin = np.min(array)
                    array = np.where(serialRegisterEvents[i,:] == 1,np.arange(0,xlen,1),-np.infty)
                    xmax = np.max(array)

                    lengthEvent = xmax-xmin
                    if lengthEvent < minlength:
                        lengthEvent = minlength

                    lengthEvent = np.round(3*lengthEvent/4)

                    # if OS != None:

                    if xmin < OS and xmax < OS and OS != None: #Inició en AA y terminó en AA
                        if xmin - lengthEvent < 0:
                            xmin = 0
                        else:
                            xmin = xmin - lengthEvent
                        
                        if xmax + lengthEvent > OS - 1:
                            xmax = OS - 1
                        else:
                            xmax = xmax + lengthEvent

                        serialRegisterEvents[i,int(xmin):int(xmax)] = 1  
                    
                    
                    elif xmin >= OS and xmax <= xlen and OS != None: #Inició en OS y terminó en OS
                        if xmin - lengthEvent < OS:
                            xmin = OS
                        else:
                            xmin = xmin - lengthEvent
                        
                        if xmax + lengthEvent > xlen - 1:
                            xmax = xlen
                        else:
                            xmax = xmax + lengthEvent

                        serialRegisterEvents[i,int(xmin):int(xmax)] = 1  

                    # elif xmin < OS and  xmax > OS: #Inició en AA y terminó en OS
                    else:
                        if xmin - lengthEvent < 0:
                            xmin = 0
                        else:
                            xmin = xmin - lengthEvent
                        
                        if xmax + lengthEvent > xlen - 1:
                            xmax = xlen
                        else:
                            xmax = xmax + lengthEvent

                        serialRegisterEvents[i,int(xmin):int(xmax)] = 1                              

                    # else:
                    #     if xmin - lengthEvent < 0:
                    #         xmin = 0
                    #     else:
                    #         xmin = xmin - lengthEvent
                        
                    #     if xmax + lengthEvent > xlen - 1:
                    #         xmax = xlen
                    #     else:
                    #         xmax = xmax + lengthEvent

                    #     serialRegisterEvents[i,int(xmin):int(xmax)] = 1
    
    return serialRegisterEvents

def makeSerialRegisterEventMask(image,n = 5,extend=True):
    """image -- numpy array which contains image,\n
    n = 5 -- number of standard deviation,\n 
    extend=True -- Make Serial register events twice larger,\n"""
    
    ylen, xlen = np.shape(image)
    serialRegisterEvents = np.zeros(np.shape(image))
    CCDDataMean, CCDDataStdDev, _, CCDMask = maskedStatistics(image,n = n)
    CCDStats = [CCDDataMean,CCDDataStdDev]
    gradCCDMask = [np.gradient(CCDMask.astype('float'),axis=0),np.gradient(CCDMask.astype('float'),axis=1)]
    for i in range(1,ylen-1,1):
        for j in range(1,xlen-1,1):
            centralPixel = [gradCCDMask[0][i,j],gradCCDMask[1][i,j]]
            upperPixel = [gradCCDMask[0][i+1,j],gradCCDMask[1][i+1,j]]
            lowerPixel = [gradCCDMask[0][i-1,j],gradCCDMask[1][i-1,j]]
            leftPixel = [gradCCDMask[0][i,j-1],gradCCDMask[1][i,j-1]]
            rightPixel = [gradCCDMask[0][i,j+1],gradCCDMask[1][i,j+1]]
            upperleftPixel = [gradCCDMask[0][i+1,j-1],gradCCDMask[1][i+1,j-1]]
            upperrightPixel = [gradCCDMask[0][i+1,j+1],gradCCDMask[1][i+1,j+1]]
            lowerleftPixel = [gradCCDMask[0][i-1,j-1],gradCCDMask[1][i-1,j-1]]
            lowerrightPixel = [gradCCDMask[0][i-1,j+1],gradCCDMask[1][i-1,j+1]]
            if centralPixel[0] == 0 and upperPixel[0] < 0 and lowerPixel[0] > 0 and  upperleftPixel[0] < 0 and upperrightPixel[0] < 0 and lowerleftPixel[0] > 0 and lowerrightPixel[0] > 0:
                if centralPixel[1] == 0 and upperPixel[1] == 0 and lowerPixel[1] == 0 and upperleftPixel[1] == 0 and upperrightPixel[1] == 0 and lowerleftPixel[1] == 0 and lowerrightPixel[1] == 0:
                    if centralPixel[0] == 0 and leftPixel[0] == 0 and rightPixel[0] == 0:
                        serialRegisterEvents[i,j] = 1

    if extend:
        serialRegisterEvents = extendSerialRegisterMask(serialRegisterEvents)
                        
    return serialRegisterEvents, CCDMask, gradCCDMask, CCDStats

def makeSerialRegisterEventAdvancedMask(image,n = 5, nOS=5, extend=True,overscan=None,frames=10):
    """image -- numpy array which contains image,\n
    n = 5 -- number of standard deviations (SD),\n
    nOS = 5 -- number of SD used to analyze overscan region if present,\n
    extend=True -- Make Serial register events twice larger,\n
    overscan=None -- Considers overscan region to be analyzed independently,\n
    frames=10 -- Default number of frames to divide horizontally full image."""
    ylen, xlen = np.shape(image)
    framelen = ylen // frames
    serialRegisterEvents = np.zeros(np.shape(image))
    CCDMask=np.zeros(np.shape(image))
    if overscan == None or overscan == 0 or overscan == False:
        totalFrames = frames
        overscanFlag = False
    else:
        totalFrames = frames + 1      
        overscanFlag = True

    for ps in range(1,totalFrames,1):

        if overscanFlag and ps == totalFrames - 1:
            imageAux = image[:,overscan:]
            # nOS = 5
            CCDDataMean, CCDDataStdDev, _, CCDMaskAux = maskedStatistics(imageAux,n = nOS )
            print(ps, CCDDataMean, np.median(imageAux))
            serialRegisterEventsAux = np.zeros(np.shape(CCDMaskAux))
        elif overscanFlag and ps < totalFrames - 1:
            imageAux = image[((ps-1)*framelen):((ps+1)*framelen),0:overscan]
            CCDDataMean, CCDDataStdDev, _, CCDMaskAux = maskedStatistics(imageAux,n = n)
            print(ps, CCDDataMean, np.median(imageAux))
        elif not(overscanFlag):
            imageAux = image[((ps-1)*framelen):((ps+1)*framelen),:]
            CCDDataMean, CCDDataStdDev, _, CCDMaskAux = maskedStatistics(imageAux,n = n)
            print(ps, CCDDataMean, np.median(imageAux))
        # if (overscanFlag and ps < totalFrames - 1) or (not(overscanFlag)):

        CCDStats = [CCDDataMean,CCDDataStdDev]
        gradCCDMask = [np.gradient(CCDMaskAux.astype('float'),axis=0),np.gradient(CCDMaskAux.astype('float'),axis=1)]
        ylenaux, xlenaux = np.shape(imageAux)
        serialRegisterEventsAux = np.zeros([ylenaux,xlenaux])
        for i in range(1,ylenaux-1,1):
            for j in range(1,xlenaux-1,1):
                centralPixel = [gradCCDMask[0][i,j],gradCCDMask[1][i,j]]
                upperPixel = [gradCCDMask[0][i+1,j],gradCCDMask[1][i+1,j]]
                lowerPixel = [gradCCDMask[0][i-1,j],gradCCDMask[1][i-1,j]]
                leftPixel = [gradCCDMask[0][i,j-1],gradCCDMask[1][i,j-1]]
                rightPixel = [gradCCDMask[0][i,j+1],gradCCDMask[1][i,j+1]]
                upperleftPixel = [gradCCDMask[0][i+1,j-1],gradCCDMask[1][i+1,j-1]]
                upperrightPixel = [gradCCDMask[0][i+1,j+1],gradCCDMask[1][i+1,j+1]]
                lowerleftPixel = [gradCCDMask[0][i-1,j-1],gradCCDMask[1][i-1,j-1]]
                lowerrightPixel = [gradCCDMask[0][i-1,j+1],gradCCDMask[1][i-1,j+1]]
                if centralPixel[0] == 0 and upperPixel[0] < 0 and lowerPixel[0] > 0 and  upperleftPixel[0] < 0 and upperrightPixel[0] < 0 and lowerleftPixel[0] > 0 and lowerrightPixel[0] > 0:
                    if centralPixel[1] == 0 and upperPixel[1] == 0 and lowerPixel[1] == 0 and upperleftPixel[1] == 0 and upperrightPixel[1] == 0 and lowerleftPixel[1] == 0 and lowerrightPixel[1] == 0:
                        if centralPixel[0] == 0 and leftPixel[0] == 0 and rightPixel[0] == 0:
                            serialRegisterEventsAux[i,j] = 1
            
        
        if overscanFlag and ps == totalFrames - 1:
            serialRegisterEvents[:,overscan:] = np.logical_or(serialRegisterEvents[:,overscan:],serialRegisterEventsAux)
            CCDMask[:,overscan:] = np.logical_or(CCDMask[:,overscan:],CCDMaskAux)
        
        elif overscanFlag and ps < totalFrames - 1:
            serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),0:overscan] = np.logical_or(serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),0:overscan],serialRegisterEventsAux)
            CCDMask[((ps-1)*framelen):((ps+1)*framelen),0:overscan] = np.logical_or(CCDMask[((ps-1)*framelen):((ps+1)*framelen),0:overscan],CCDMaskAux)        
        elif not(overscanFlag):
            serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),:] = np.logical_or(serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),:],serialRegisterEventsAux)
            CCDMask[((ps-1)*framelen):((ps+1)*framelen),:] = np.logical_or(CCDMask[((ps-1)*framelen):((ps+1)*framelen),:],CCDMaskAux)        
    if extend:
        serialRegisterEvents = extendSerialRegisterMask(serialRegisterEvents, OS=overscan)
                        
    return serialRegisterEvents, CCDMask, gradCCDMask, CCDStats


def saveObject2File(filename,object):
    with open(filename, 'wb') as file:
        pickle.dump(object,file)
    return 0

def readObjectFromFile(filename):
    with open(filename, 'rb') as file:
        object = pickle.load(file)
    return object

def maskedStatistics(npArray,n = 4,mask=None,tol=0.1):
    """npArray -- Contains imaga data as numpy array,\n
    n = 5 -- number of standard deviation,\n 
    mask -- a previous mask to be considered previously to computation,\\
    tol = 0.1 -- Tolerance to convergence
    """
    mean = float(np.ma.average(npArray))
    stdDev = float(np.ma.std(npArray))
    Count = 0
    MaxIter = 20
    flag = True
    meanPre = mean
    if np.any(mask) != None:
        Mask = mask
    while (Count < MaxIter) and (flag == True):
        Lrange = mean - n * stdDev
        Hrange = mean + n * stdDev
        Mask = np.logical_or(np.less_equal(npArray,Lrange), np.greater_equal(npArray,Hrange))
        npArrayMasked = np.ma.array(npArray, mask=Mask)
        mean = float(np.ma.average(npArrayMasked))
        stdDev = float(np.ma.std(npArrayMasked))
        Count += 1

        if np.abs(mean - meanPre) < tol:
            flag = False
        else:
            meanPre = mean
    return mean, stdDev, npArrayMasked, Mask

def PlotImage(Image,MinRange=None,MaxRange=None):
    try:
        if MinRange > MaxRange:
            Aux = MaxRange
            MaxRange = MinRange
            MinRange = Aux
    except TypeError:
        MinRange=None
        MaxRange=None
    plt.imshow(Image, cmap='viridis',vmin=MinRange,vmax=MaxRange,origin="lower")
    plt.colorbar()
    plt.show()

def Plot2Images(Image,Mask,MinRange=None,MaxRange=None, colorBar_1=True, colorBar_2=True):
    # fig, (ax1,ax2) = plt.subplots(1,2)
    try:
        if MinRange > MaxRange:
            Aux = MaxRange
            MaxRange = MinRange
            MinRange = Aux
    except TypeError:
        MinRange=None
        MaxRange=None
    # ax1.imshow(Mask, cmap='viridis',origin="lower")
    # ax2.imshow(Image, cmap='viridis',vmin=MinRange,vmax=MaxRange,origin="lower")

    fig=plt.figure(figsize=(20,10))

    plt.subplot(1,2,2)
    
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')

    if Mask.max()==1:
        im=plt.imshow(Mask, cmap='viridis',vmin=Mask.min(),vmax=Mask.max(), origin="lower")
    else:
        im=plt.imshow(Mask, cmap='viridis',vmin=MinRange,vmax=MaxRange, origin="lower")
    if colorBar_1:
        cbar=fig.colorbar(im)
    #cbar.set_label('boolean')

    plt.subplot(1,2,1)
    # plt.title(Image)  
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')
    im=plt.imshow(Image, cmap='viridis', vmin=MinRange,vmax=MaxRange,origin="lower")
    if colorBar_2:
        cbar=fig.colorbar(im)
    #cbar.set_label('[e-]')

    
    plt.show()

def ClusterRecon(CCDimage,CCDMean,CCDStdDev,Max_Thres=2**16-1,KThres=4,minPixelSize=0,Ext='Not Available',FileID=0,ADUtokEVConst=0,EventInitNum=0):
    
    # CCDThresBin_2 = cv2.threshold(CCDimage,CCDMean+KThres*CCDStdDev,Max_Thres,cv2.THRESH_BINARY)

    CCDThresBin = np.where(CCDimage >= CCDMean+KThres*CCDStdDev, Max_Thres,0)
    labelCCD = measure.label(CCDThresBin, connectivity=2, background=0)
    EventList = []
    CCDDataMasked = np.ma.copy(CCDimage)
    CCDThresMasked = np.ma.copy(CCDThresBin)
    for ps in range(np.min(labelCCD),np.max(labelCCD),1):
        if ps != 0:
            labelMaskCCD = labelCCD != ps
            numPixels = np.sum(np.logical_not(labelMaskCCD))
            if numPixels > minPixelSize:
                CCDDataMasked.mask = labelMaskCCD
                CCDThresMasked.mask = labelMaskCCD
                Sum = np.ma.sum(CCDDataMasked)
                TotalSum =Sum - numPixels*CCDMean
                CenterMass = center_of_mass(CCDDataMasked)
                CMX = CenterMass[0]
                CMY = CenterMass[1]
                Coord = ma.nonzero(CCDDataMasked)
                CX = np.average(Coord[0])
                CY = np.average(Coord[1])
                if ADUtokEVConst == 0:
                    Energy = '----'
                else:
                    Energy = TotalSum/ADUtokEVConst
                Event = [ps+EventInitNum,Sum.real,TotalSum,Energy,FileID,Ext,[CMX,CMY],[CX,CY],numPixels,CCDMean,CCDStdDev]
                EventList.append(Event)

    return EventList, CCDThresBin, labelCCD

def MakeEventLists(EventList):
    LabelList = []
    SumList = []
    TotalSumList = []
    CenterCharge = []
    Coord = []
    Ext = []
    numPixelList = []
    FileID = []
    Energy = []
    Mean = []
    StdDev = []
    for Event in EventList:
        LabelList.append(Event[0])
        SumList.append(Event[1])
        TotalSumList.append(Event[2])
        Energy.append(Event[3])
        FileID.append(Event[4])
        Ext.append(Event[5])
        CenterCharge.append(Event[6])
        Coord.append(Event[7])
        numPixelList.append(Event[8])
        Mean.append(Event[9])
        StdDev.append(Event[10])

    return LabelList, SumList, TotalSumList, Energy, FileID, Ext, CenterCharge, Coord, numPixelList, Mean, StdDev 

def ClusterDict(EventList):
    LabelList, SumList, TotalSumList, Energy, FileID, Ext, CenterCharge, Coord, numPixelList, Mean, StdDev = MakeEventLists(EventList)
    OutputDict = {'1. Number of event':LabelList,'2. Sum of pixels values/ADU':SumList,
        '3. Charge of events/ADU':TotalSumList,'4. Energy/keV':Energy,'5. FileID':FileID,'6. Ext':Ext,'7. Center of Charge':CenterCharge, '8. Coordinates':Coord,
        '9. Pixels in the cluster':numPixelList,'10. Background Mean':Mean, '11. Background Standard Deviation':StdDev}
    
    return OutputDict

def VerifySizeXY(X,Y):
    if type(X) is list:
        LenX = []
        LenY = []
        for Xs,Ys in zip(X,Y):
            LenX.append(len(Xs))
            LenY.append(len(Ys))
    else:
        LenX = len(X)
        LenY = len(Y)
    
    if type(X) is list:
        Xaux = []
        Yaux = []
        count = 0 
        for Xs,Ys in zip(X,Y):
            diff = LenX[count] - LenY[count]
            if diff > 0:
                Xaux.append(Xs[:(-diff)])
                Yaux.append(Ys)
            elif diff < 0:
                Yaux.append(Ys[:diff])
                Xaux.append(Xs)
        X = Xaux
        Y = Yaux
    else:
        if LenX != LenY:
            diff = LenX - LenY
            if diff > 0:
                X = X[:(-diff)]
            elif diff < 0:
                Y = Y[:diff]
    return X,Y

def SavePlot2File(X,Y,LabelL,Marker='line',Log=None):
    Title = LabelL[0]
    Xlabel = LabelL[1]
    Ylabel = LabelL[2]
    filename = LabelL[3]
    Legend = LabelL[4]
    FontDict={'color':  'black','weight': 'normal','size': 18}
    X,Y = VerifySizeXY(X,Y)
    if Marker == 'symbol': 
        plt.scatter(X,Y,marker='+',c='b',label=Legend)
    elif Marker == 'line':
        if type(X) is list:
            for Xs,Ys,Legends in zip(X,Y,Legend):
                plt.plot(Xs,Ys,label=Legends)
        else:
            plt.plot(X,Y,label=Legend)
        
    if Log == 'log':
        plt.yscale('log')

    plt.xlabel(Xlabel,fontdict=FontDict)
    plt.ylabel(Ylabel,fontdict=FontDict)
    plt.title(Title,fontdict=FontDict)
    plt.savefig(filename,format='pdf',dpi=300,orientation='landscape')
    plt.clf()

def DAMICCalibrationConstantsDict():
    k = 2.6e-4
    CalConst = {'1':1.009/k,'2':0.956/k,'3':0.979/k,'4':0.984/k,'6':1.010/k,'11':0.969/k,'12':0.999/k}
    return CalConst

def DAMICGoodExt():
    return [1,2,3,4,6]

def WriteTXTCSVFiles(OutputDict,FileName):
    df = pd.DataFrame(OutputDict)
    OutputFile = FileName.split('.fits')[0] + '_RECON.txt'
    File = open(OutputFile,'w+')
    File.write(df.to_string())
    File.close()
    OutputFile = FileName.split('.fits')[0] + '_RECON.csv'
    File = open(OutputFile,'w+')
    File.write(df.to_csv())
    File.close()

def WriteIDFile(FileIDName,FileName,ID):
    File = open(FileIDName,'a+')
    FileIDstr = str(ID) + ". - " + FileName + '\n'
    File.write(FileIDstr)
    File.close()

def CenterBins(BinsList):
    Arr1 = np.array(BinsList[1:])
    Arr2 = np.array(BinsList[:-1])
    BinCenter = (Arr1 + Arr2)/2
    return BinCenter

def HistogramDict(HistogramData):
    xData = HistogramData[1]
    yData = HistogramData[0]
    CenterX = CenterBins(xData)
    Dict = {'Energy/keV':CenterX,'Counts':yData}
    return Dict

def ReadCoordDict(Filename):
    try:
        dt = pd.read_csv(Filename,delimiter=':',index_col=0,squeeze=True,header=None).to_dict()
        #CoordDict = {}
        CoordDict = {'NoiseX':[dt['NoiseX1'],dt['NoiseX2']],
        'CCDX':[dt['CCDX1'],dt['CCDX2']],
        'CCDOSX':[dt['CCDOSX1'],dt['CCDOSX2']],
        'NoiseOSX':[dt['NoiseOSX1'],dt['NoiseOSX2']],
        'NoiseY':[dt['NoiseY1'],dt['NoiseY2']],
        'CCDY':[dt['CCDY1'],dt['CCDY2']],
        'CCDOSY':[dt['CCDOSY1'],dt['CCDOSY2']],
        'NoiseOSY':[dt['NoiseOSY1'],dt['NoiseOSY2']]}
        return CoordDict
    except:
        print('ERROR: An unexpected error reading .coord file.')
        print('The execution is going to finish.')
        CoordDict = None
        return CoordDict 
    
def ReadGoodExtFile(ExtFile):
    try:
        File = open(ExtFile,'r')
        List = File.split('\n')
        GoodExt = [float(num) for num in List]
    except:
        print('ERROR: An unexpected error reading .coord file.')
        print('The execution is going to finish.')
        GoodExt = None
    return GoodExt
    
def ReadConstFile(ConstFile,GoodExt):
    try:
        File = open(ConstFile,'r')
        List = File.split('\n')
        CCDConst = [float(num) for num in List]
        CCDConstDict = {}
        for key, value in zip(GoodExt,CCDConst):
            CCDConstDict[str(key)] = value

    except:
        print('ERROR: An unexpected error reading .coord file.')
        print('The execution is going to finish.')
        CCDConstDict = None
    
    if bool(CCDConstDict):
        return CCDConstDict
    else:
        return None
    
def OS_comparison(Image_1,Image_2=None, Image_3=None, Image_4=None, Image_5=None,MinRange=None,MaxRange=500, colorBar_1=True, colorBar_2=True, titleImage=None):
    # fig, (ax1,ax2) = plt.subplots(1,2)
    imageList=[Image_1,Image_2,Image_3,Image_4, Image_5]
    MaxRangeList=[1,1,1,1,MaxRange]
    if titleImage==None: 
        titleList=['n=2.5','n=3','n=3.5','n=4','Image'] #Cambia el valor de la Std Dev que vayas a plotear para que no haya confusion
    else: 
        titleList=['n='+str(titleImage[0]),'n='+str(titleImage[1]),'n='+str(titleImage[2]),'n='+str(titleImage[3]),'Image'] #Cambia el valor de la Std Dev que vayas a plotear para que no haya confusion
    try:
        if MinRange > MaxRange:
            Aux = MaxRange
            MaxRange = MinRange
            MinRange = Aux
    except TypeError:
        MinRange=None
        MaxRange=None
    # ax1.imshow(Mask, cmap='viridis',origin="lower")
    # ax2.imshow(Image, cmap='viridis',vmin=MinRange,vmax=MaxRange,origin="lower")

    fig=plt.figure(figsize=(15,8))
    
    for Image in range(len(imageList)):
        plt.subplot(1,len(imageList),Image+1)
        plt.ylabel('Y_pix')
        plt.xlabel('X_pix')
        plt.title(titleList[Image])
        im=plt.imshow(imageList[Image][:,550:], cmap='viridis',origin="lower",vmin=0,vmax=MaxRangeList[Image])
        
        if colorBar_1:
            cbar=fig.colorbar(im)
    
    fig.tight_layout()

    plt.show()

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

#Calculo de Incertidumbre

#Generar Imagen Mediana
def MedianImage(path):
    data_list=[]
    flag=True
    # path='/home/oem/datosFits/serialRegEvents/400x700/setNuevo/'
    n=0
    files =os.listdir(path)
    validFiles=[]
    medianImage=[]

    for file in files:
        if file.endswith(".fits"):
            validFiles.append(path+file)
            print(file+'....OK')
        else:
            print(file+'Not Fits')

    for image_name in validFiles:
        dataFromImage=fits.open(image_name) #imagen en curso
        if flag:
            (x,y)=np.shape(dataFromImage[0])
            flag=False
            emptyArray=np.empty((x,y,len(validFiles)),dtype=float)
            for ext in range(0,4):
                data_list.append(np.copy(emptyArray))

        for ext in range(0,4):
            data_list[ext][:,:,n]=dataFromImage[ext].data
        n+=1
        

    for ext in range(0,4):
        medianImage.append(np.median(data_list[ext],axis=2))
    
    for ext in range(4):
        expImg(medianImage[ext],name=('medianImage_'+str(ext)))

    return medianImage, validFiles 


def findUncertaintyQ(mean1,stddev1,mean2,stddev2,n1,n2):
    Q = np.abs(mean1/mean2)
    uncertainty = Q*np.sqrt( (mean1/stddev1)**2+(mean2/stddev2)**2 )
    return Q, uncertainty

def getStatsFromDict(file_Dict):
    means = {}
    stdDev = {}
    for Ext in file_Dict.keys():
        nL = np.array(list(file_Dict[Ext].keys()),dtype=float) #Revisar tamaño de nL
        means[Ext] = 0*np.copy(nL)
        stdDev[Ext] = 0*np.copy(nL)
        for ps,nvar in zip(range(0,len(nL),1),list(file_Dict[Ext].keys())):
            means[Ext][ps] = np.mean(file_Dict[Ext][nvar])
            stdDev[Ext][ps] = np.std(file_Dict[Ext][nvar],ddof=1)

        print("Medias extension {}: ".format(Ext), means[Ext])
        print("Std Dev extension {}: ".format(Ext), stdDev[Ext]) 

    return means, stdDev, nL


def makeDictionary(array):
    file_Dict = {}
    for r in array:
        filenumber,Ext,nvar,xsizeAR,ysizeAR,xsizeOS,ysizeOS,numSRE_all,numSRE_AA,numSRE_OS,numSRE_AAOS,mean_CCD,stdDev_CCD,countIsCorrect = r

        if Ext not in file_Dict:
            file_Dict[Ext] = {} 

        if nvar not in file_Dict[Ext]:
            file_Dict[Ext][nvar] = np.array([float(numSRE_AA)])
        else:
            file_Dict[Ext][nvar] = np.append(file_Dict[Ext][nvar],float(numSRE_AA))
    return file_Dict

def expImg(data, name='fig_1'):
    fig=plt.imshow(data, origin='lower', cmap='gray')
    file=open(name+'_img.pkl', 'wb')
    pickle.dump(fig, file)
    plt.close("all")
    file.close()
    file=open(name+'_data.pkl', 'wb')
    pickle.dump(data, file)
    file.close()

    return 'ok'