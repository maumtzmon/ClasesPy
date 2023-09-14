import pandas as pd
import numpy as np
from skimage import feature, measure
import cv2
import numpy.ma as ma
from scipy.ndimage.measurements import center_of_mass
from matplotlib import pyplot as plt
import pickle
import pickle


def makeSerialRegisterEventMask(image,n = 5,extend=True):
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
                    lengthEvent = np.round(lengthEvent/2)

                    if xmin - lengthEvent < 0:
                        xmin = 0
                    else:
                        xmin = xmin - lengthEvent
                    
                    if xmax + lengthEvent > xlen - 1:
                        xmax = xlen
                    else:
                        xmax = xmax + lengthEvent

                    serialRegisterEvents[i,int(xmin):int(xmax)] = 1
                    # for j in range(0,xlen-1,1):
                        
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

# def maskedStatistics(npArray,n = 4):
#     mean = float(np.average(npArray))
#     stdDev = float(np.std(npArray))
#     print(mean,stdDev)
#     Count = 0
#     MaxIter = 100
#     flag = True
#     meanPre = None
#     while (Count < MaxIter) and (flag == True):
#         Lrange = mean - n * stdDev
#         Hrange = mean + n * stdDev
#         Mask = np.logical_or(np.less_equal(npArray,Lrange), np.greater_equal(npArray,Hrange))
#         npArrayMasked = np.ma.array(npArray, mask=Mask)
#         mean = float(np.average(npArrayMasked))
#         stdDev = float(np.std(npArrayMasked))
#         Count += 1
#         if mean == meanPre:
#             flag = False
#         else:
#             meanPre = mean
#     return mean, stdDev, npArrayMasked, Mask

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

def ClusterRecon(CCDimage,CCDMean,CCDStdDev,Max_Thres=2**16-1,KThres=4,minPixelSize=0,Ext='Not Available',FileID=0,ADUtokEVConst=0,EventInitNum=0):
    
    # CCDThresBin_2 = cv2.threshold(CCDimage,CCDMean+KThres*CCDStdDev,Max_Thres,cv2.THRESH_BINARY)

    CCDThresBin = np.where(CCDimage >= CCDMean+KThres*CCDStdDev, Max_Thres,0)
    #labelCCD = measure.label(CCDThresBin[1], neighbors=8, background=0,)
    labelCCD = measure.label(CCDThresBin, connectivity=2, background=0,)
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
    
    
    