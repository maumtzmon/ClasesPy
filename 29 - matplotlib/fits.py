#!/usr/bin/python3
#from hamcrest import none
import matplotlib
from matplotlib.legend import Legend
import pandas as pd
import os
import sys
import glob
import numpy as np
from scipy import stats
from scipy.stats import norm
from matplotlib import pyplot as plt
from astropy.io import fits
import math
import argparse
import pylab
from scipy.optimize import curve_fit

def gaus(x,k,x0,sigma,c=0):
    return k*np.exp(-(x-x0)**2/(2*sigma**2)) + c



def calculateHistogram(data):

    Numbins= np.histogram_bin_edges(data.flatten(),bins='fd')
    data_histo = np.histogram(data.flatten(),bins=Numbins)

    return Numbins, data_histo[0] 

def parser():
    parser = argparse.ArgumentParser(prog="FITSanlyzer", description='High particle FITS images analyzer. \n Aqui un breve comentario sobre el proposito y objetivo de este codigo')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    parser.add_argument('-f','--pdf',action='store_true',help="Save results in PDF file.")

    subparsers=parser.add_subparsers(help="sub command Help")

    ###
    # Positional Arguments
    ###
    sub_action_1 = subparsers.add_parser('skp', help = 'skipping tool help \n Creates an regurlar CCD image using a skipped image')
    sub_action_1.add_argument('file', type=str, nargs = '+',help='file or path of FITS images')
    sub_action_1.add_argument('n', type=int, nargs = 1, help = 'number of skipping cycles of the images' )
    #run skipped to regular imege function

    sub_action_2 = subparsers.add_parser('other_subOPT', help = 'help')
    sub_action_2.add_argument('Variable_PATH', type=int, help='help')
    #run function

    ###
    # Optional arguments
    ###
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i','--histogram',type=str, # action='append',
                    help='Makes the histogram of charge in the active region (AA) and in overscan regions (OSX, OSY, OSXY).')
    group.add_argument('-c','--charge',type=str,nargs='+',# action='store_true',
                        help='Estimates the charge in the active region (AA) and in overscan regions (OSX, OSY, OSXY).')
    group.add_argument('-d','--dCurrent', type=str,nargs='+', # Corriente obscura
                        help='Estimates Dark Current in the region selected')

    group.add_argument('-e','--eventDet', type=str,nargs='+',# Detección de eventos
                        help='')

    # subparsers = parser.add_subparsers()
    # parser_x = subparsers.add_parser("-x",help="x coordinates x1 x2 x3 x4.")
    parser.add_argument('-x',type=int,nargs=4,help="x coordinates x1 x2 x3 x4.")
    parser.add_argument('-y',type=int,nargs=4,help="y coordinates y1 y2 y3 y4.")
    parser.add_argument('-b','--baseline',action='store_true',help="enable option Baseline substraction.") #va a llevar algun otro argumento adicional
    parser.add_argument('--ext', type=int, nargs='+', help= 'Indicate which extentions do you want to use' )


   

    #output option

    argObj = parser.parse_args()
    return argObj


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

def PlotImage(Image):
    plt.imshow(Image, cmap='gray')
    plt.colorbar()
    plt.show()

def chi2_DOF(p,x,y):
    chi_squared = np.ma.sum((np.polyval(p, x) - y) ** 2)
    chi2_DOF = chi_squared/(len(y)-1)
    return chi2_DOF

def SavePlot2File(X,Y,LabelL,Marker): #Usar como base para generar imagen de histograma. marker no se usara para histogramas
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
        plt.yscale('symlog')

    plt.xlabel(Xlabel,fontdict=FontDict)
    plt.ylabel(Ylabel,fontdict=FontDict)
    plt.title(Title,fontdict=FontDict)
    plt.savefig(filename,format='pdf',dpi=300,orientation='landscape')
    plt.clf()
    #plt.show()

def ShowHisto2PDF(binsDict,LabelL,saveFile=False, baseline=False):
        
    Title = LabelL
    Xlabel = "ADU"
    Ylabel = "Counts"

    FontDict={'color': 'black','weight': 'normal','size': 18}
    
    # if baseline:
    #     pass
    # elif baseline:


    if baseline==False:
        fig, axes=plt.subplots(2, 2, figsize=[9.5,8.5])
        fig.suptitle(Title)
        fig=pylab.gcf()
        fig.canvas.manager.set_window_title('Histogram')
        axes[0, 0].set_title('Active Area')
        axes[0, 1].set_title('Overscan X')
        axes[1, 0].set_title('Overscan Y')
        axes[1, 1].set_title('Overscan XY')

        axes[0, 0].tick_params(width=1.0, labelsize=8)
        axes[0, 1].tick_params(width=1.0, labelsize=8)
        axes[1, 0].tick_params(width=1.0, labelsize=8)
        axes[1, 1].tick_params(width=1.0, labelsize=8)

        axes[0, 0].ticklabel_format(useMathText=True)


        axes[0, 0].set(ylabel='Counts')
        axes[1, 0].set(xlabel='ADUs', ylabel='Counts')
        axes[1, 1].set(xlabel='ADUs')


        numList = [1,2,3,4]
        for a, bins, weights, classMarks in zip(numList, binsDict["bins"],binsDict["data_histo"],binsDict["class"]): 
            if a==1:
                b, c = 0, 0
            elif a==2:
                b, c = 0, 1
            elif a==3:
                b, c = 1, 0
            elif a==4:
                b, c = 1, 1
                                                                    # bins[], bins y weights* 1e5 - 1e10  genera la notacion cientifica
                                                                    # en el eje Y
            histogram=axes[b,c].hist(bins[:-1],bins,weights=weights, color='forestgreen')#Genera el histograma
                                                                    #la curva y el histograma 
            const = 0; k = np.max(weights); mean = np.sum(classMarks*weights)/np.sum(weights)
            sigma = np.sqrt(np.sum(weights*(classMarks-mean)**2)/(np.sum(weights)))
            # sigma /= 500
            popt_gaus,_ = curve_fit(gaus,classMarks,weights,p0=[k,mean,sigma,const])
            # mu, std =norm.fit(bins[:-1])                            #genera los parametros de la curva gaussiana
            x=np.linspace(histogram[1].min(), histogram[1].max(), 10*histogram[1].size) #genera los datos en el eje X
            # y = norm.pdf(x, mu, std)                                #genera los datos en y de la curva
            y = gaus(x,popt_gaus[0],popt_gaus[1],popt_gaus[2],c=popt_gaus[3])
            axes[b,c].plot(x,y,'k--', lw=1, label='fit')                    #hace el plot de la curva y lo monta sobre el histograma
            
            yMax=y.max()
            xMax=x[np.where(y == yMax)[0][0]]
            axes[b,c].plot([xMax], [yMax], lw=0, marker='o', markerfacecolor='k',markeredgecolor='k', label='mean='+str(round(xMax,ndigits=2)))
            
            xStd=[xMax-sigma,xMax+sigma]
            yStd=gaus(xStd,popt_gaus[0],popt_gaus[1],popt_gaus[2],c=popt_gaus[3])

            axes[b,c].plot([xStd[0],xStd[0]],[0,yStd[0]], 'k--', lw=1)
            axes[b,c].plot([xStd[1],xStd[1]],[0,yStd[1]], 'k--', lw=1, label=r"$\sigma=\pm$"+str(round(sigma,ndigits=2)))

            axes[b,c].legend(loc="upper right")

                        

        if saveFile:
            plt.savefig(Title+".pdf",format='pdf',dpi=300,orientation='landscape')
            #papertypestr
            #One of 'letter', 'legal', 'executive', 'ledger', 'a0' through 'a10', 'b0' through 'b10'. Only supported for postscript output

        plt.show()
    #plt.clf()


def markClass(array):
    mark = (array[:-1]+array[1:])/2
    return mark

def MkDir(ResultsPath):
    if not os.path.isdir(ResultsPath):
        try:
            os.makedirs(ResultsPath)
        except OSError:
            print('---------------------------------------------------')
            print('ERROR: The folder ' + ResultsPath + ' could not be created.')
            print('Aborting execution.')
            print('---------------------------------------------------')
            exit(1)

def ReportHisto(filename,X,Y):
    X,Y = VerifySizeXY(X,Y)
    HistoDict = {'1. ADU':X,'2. Counts':Y}
    df = pd.DataFrame(data=HistoDict)
    df.to_csv(filename + '.csv')
    datafile = open(filename + '.txt','w')
    datafile.write(df.to_string())
    datafile.close()

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

def NumpyList2FloatList(npList):
    FloatList = [float(i) for i in npList]
    return FloatList

def UnusedFiles(filesList):
    for file in filesList:
        Name = file.split('/')[-1]
        if file.endswith('0.fits') or file.endswith('1.fits') or file.endswith('3.fits') or Name.startswith('Trash') :
        #if not (file.endswith('2.fits')):
            os.rename(file,file + '.unused')

#-----------------------------------------------
#The directory of the images.
#The directory where the results are going to be saved.
#-----------------------------------------------

def old_main(argv): 

    
    if len(argv) == 2:
        folderpath = argv[1]
        filesToSearch = '*.fits'
        filesList = glob.glob(folderpath + filesToSearch)
        UnusedFiles(filesList)
        filesList = glob.glob(folderpath + filesToSearch)
        ResultsPath = argv[1] + 'Results'
        MkDir(ResultsPath)
        ResultsPath += '/'
        
        #-----------------------------------------------
        #Definition of the regions to be analyzed
        #This variables are list that contains N coordinates
        #to analyze the CCD in those places
        #-----------------------------------------------

        yos1L = [1050]
        yos2L = [1350]
        yaa1L = [550]
        yaa2L = [950]
        xos1L = [20]
        xos2L = [350]
        xaa1L = [450]
        xaa2L = [4200]

        # yos1L = [100]
        # yos2L = [300]
        # yaa1L = [600]
        # yaa2L = [105 0]
        # xos1L = [4100]
        # xos2L = [4140]
        # xaa1L = [1000]
        # xaa2L = [4000]
        # yos1L = [100,100]
        # yos2L = [300,300]
        # yaa1L = [600,600]
        # yaa2L = [1050,1050]
        # xos1L = [4100,4100]
        # xos2L = [4140,4140]
        # xaa1L = [1000,3500]
        # xaa2L = [3500,4000]

        #yos1L = [1050,1050]
        #yos2L = [1350,1350]
        #yaa1L = [500,500]
        #yaa2L = [980,980]
        #xos1L = [4140,4140]
        #xos2L = [4480,4480]
        #xaa1L = [400,1950]
        #xaa2L = [1900,3450]


        #-----------------------------------------------
        #Defintion of rectangles
        #Rectangle format: [x1,y1,x2,y2]
        #-----------------------------------------------

        RectActiveAreaList_L = [] 
        RectOS_X_L = []
        RectOS_Y_L = []
        RectOS_OS_L = []
        XrangeAAL = []
        YrangeAAL = []
        XrangeOSL = []
        YrangeOSL = []
        DictDeltaFit = {}
        DictDeltaAvg = {}
        DictResAvg = {}
        #x1 -> xos1, x2 -> xos2, x3 -> xaa1, x4 -> xaa2 
        for xos1,xos2,xaa1,xaa2,yos1,yos2,yaa1,yaa2 in zip(xos1L,xos2L,xaa1L,xaa2L,yos1L,yos2L,yaa1L,yaa2L):
            RectActiveAreaList_L.append([xaa1,yaa1,xaa2,yaa2])
            RectOS_Y_L.append([xaa1,yos1,xaa2,yos2])
            RectOS_X_L.append([xos1,yaa1,xos2,yaa2])
            RectOS_OS_L.append([xos1,yos1,xos2,yos2])
            XrangeAAL = np.ma.array(range(xaa1,xaa2,1))
            YrangeAAL = np.ma.array(range(yaa1,yaa2,1))
            XrangeOSL = np.ma.array(range(xos1,xos2,1))
            YrangeOSL = np.ma.array(range(yos1,yos2,1))
            
        #-----------------------------------------------
        #Main cycle.
        #-----------------------------------------------

        for file in filesList:
            FitsObj = fits.open(file)
            filename = file.split('/')
            filename = filename[-1]
            basename = filename.split('.fits')[0]
            StatResultL = []
            StepResultL = []

            for ps in range(0,len(FitsObj),1): #Este ciclo for itera sobre las extensiones de la imagen
                count = 1
                lps = str(ps)
                if lps not in DictDeltaFit:
                    DictDeltaFit[lps] = {'Name':[]}
                if lps not in DictDeltaAvg:
                    DictDeltaAvg[lps] = {'Name':[]}
                if lps not in DictResAvg:
                    DictResAvg[lps] = {'Name':[]}

                if file not in DictDeltaFit[lps]['Name']:
                    DictDeltaFit[lps]['Name'].append(file.split('/')[-1])
                if file not in DictDeltaAvg[lps]['Name']:
                    DictDeltaAvg[lps]['Name'].append(file.split('/')[-1])
                if file not in DictResAvg[lps]['Name']:
                    DictResAvg[lps]['Name'].append(file.split('/')[-1])

                for ROS_X,ROS_Y,RActiveArea,RectOS_OS in zip(RectOS_X_L,RectOS_Y_L,RectActiveAreaList_L,RectOS_OS_L):
                    #-----------------------------------------------
                    #The folders where the files are going to be saved 
                    #are created. The idea is that a folder for each file is created
                    # and, inside of it, subfolder per extension and subfolder per region
                    #analyzed are created.
                    #-----------------------------------------------
                    SaveDir = ResultsPath + '_' + basename + '/Ext_' + str(ps + 1) + '/R' + str(count)
                    MkDir(SaveDir)
                    SaveDir += '/'
                    #----------------------------------------------- 
                    #Printing on the screen to indicte user the progess.
                    #-----------------------------------------------
                    print("\n----------------------------------")
                    print("File to be analized: " + file)
                    print("----------------------------------")
                    print("\n\n")
                    #-----------------------------------------------
                    #Here the data is masked and some parameters of the 
                    #region are calculated for the four regions of the CCD
                    # considering the mask. 
                    #-----------------------------------------------
                    FitsObj = fits.open(file)
                    RectOS_X_mean,RectOS_X_stddev,RectOS_X_data,RectOS_X_mask = maskedStatistics(FitsObj[ps].data[ROS_X[0]:ROS_X[2],ROS_X[1]:ROS_X[3]])
                    RectOS_Y_mean,RectOS_Y_stddev,RectOS_Y_data,RectOS_Y_mask = maskedStatistics(FitsObj[ps].data[ROS_Y[0]:ROS_Y[2],ROS_Y[1]:ROS_Y[3]])
                    AA_mean,AA_stddev,AA_data,AA_mask = maskedStatistics(FitsObj[ps].data[RActiveArea[0]:RActiveArea[2],RActiveArea[1]:RActiveArea[3]])
                    OS_OS_mean,OS_OS_stddev,OS_OS_data,OS_OS_mask = maskedStatistics(FitsObj[ps].data[RectOS_OS[0]:RectOS_OS[2],RectOS_OS[1]:RectOS_OS[3]])
                    #-----------------------------------------------
                    #The average of each row is calculated here for each region
                    #-----------------------------------------------
                    AA_Avg = np.ma.average(AA_data,axis=0)
                    ROS_X_Avg = np.ma.average(RectOS_X_data,axis=0)
                    ROS_Y_Avg = np.ma.average(RectOS_Y_data,axis=0)
                    OS_OS_Avg = np.ma.average(OS_OS_data,axis=0)
                    #-----------------------------------------------
                    #The baseline is substracted in order to get data around zero.
                    #-----------------------------------------------
                    OS_BLS_Avg,OS_BLS_Std,OS_BLS,OS_BLS_Mask = maskedStatistics(ROS_Y_Avg - OS_OS_Avg,n=3)
                    AA_BLS_Avg,AA_BLS_Std,AA_BLS,AA_BLS_Mask = maskedStatistics(AA_Avg - ROS_X_Avg,n=3)
                    
                    #OS_BLS = ROS_Y_Avg - OS_OS_Avg  #Overscan baseline substracted
                    #AA_BLS = AA_Avg - ROS_X_Avg #Overscan baseline substracted
                    #OS_BLS_Avg = np.ma.average(OS_BLS)
                    #AA_BLS_Avg = np.ma.average(AA_BLS)
                    YrangeAAL.mask = AA_BLS.mask
                    YrangeOSL.mask = OS_BLS.mask
                    #-----------------------------------------------
                    #The data is fitted, also the chi2 is calculated
                    #-----------------------------------------------
                    OS_BLS_fit = stats.linregress(YrangeOSL,OS_BLS)
                    AA_BLS_fit = stats.linregress(YrangeAAL,AA_BLS)
                    OS_BLS_chi2 = chi2_DOF([OS_BLS_fit[0],OS_BLS_fit[1]],YrangeOSL,OS_BLS)
                    AA_BLS_chi2 = chi2_DOF([AA_BLS_fit[0],AA_BLS_fit[1]],YrangeAAL,AA_BLS)
                    Delta1 = abs(AA_BLS_Avg - OS_BLS_Avg)
                    Delta2 =  abs(AA_BLS_fit[1] - OS_BLS_fit[1])
                    #-----------------------------------------------
                    #Histograms are calculated
                    #-----------------------------------------------
                    Step = 5
                    Range = range(int(math.floor(OS_BLS.min())),int(math.ceil(OS_BLS.max())),Step)
                    Numbins=len(Range)# + 1
                    OS_BLS_histo = np.histogram(OS_BLS,range=[Range.start,Range.stop],bins=Numbins)
                    Step = 5
                    Range = range(int(math.floor(AA_BLS.min())),int(math.ceil(AA_BLS.max())),Step)
                    Numbins=len(Range)# + 1
                    AA_BLS_histo = np.histogram(AA_BLS,range=[Range.start,Range.stop],bins=Numbins)
                    Step = 10
                    Numbins=len(Range)# + 1
                    Range = range(int(math.floor(AA_data.min())),int(math.ceil(AA_data.max())),Step)
                    AA_data_histo = np.histogram(AA_data,range=[Range.start,Range.stop],bins=Numbins)
                    Step = 10
                    Numbins=len(Range)# + 1
                    Range = range(int(math.floor(OS_OS_data.min())),int(math.ceil(OS_OS_data.max())),Step)
                    OS_OS_data_histo = np.histogram(OS_OS_data,range=[Range.start,Range.stop],bins=Numbins)
                    Step = 10
                    Numbins=len(Range)# + 1
                    Range = range(int(math.floor(RectOS_X_data.min())),int(math.ceil(RectOS_X_data.max())),Step)
                    RectOS_X_data_histo = np.histogram(RectOS_X_data,range=[Range.start,Range.stop],bins=Numbins)
                    Step = 10
                    Numbins=len(Range)# + 1
                    Range = range(int(math.floor(RectOS_Y_data.min())),int(math.ceil(RectOS_Y_data.max())),Step)
                    RectOS_Y_data_histo = np.histogram(RectOS_Y_data,range=[Range.start,Range.stop],bins=Numbins)

                    #-----------------------------------------------
                    #Plots are saved and reports for histograms are generated
                    #-----------------------------------------------
                    Title = 'Dark current - ' + filename
                    legend = 'Ext - ' + str(ps+1) + '. Region - ' + str(count)
                    picfolder = 'step/'
                    MkDir(SaveDir + picfolder)
                    imagename = 'Step_'+ basename + '_'+ str(count) + '.pdf'
                    Ylabel = 'ADU'
                    Xlabel = 'Pixels'
                    StepX = np.ma.append(YrangeAAL,YrangeOSL)
                    StepY = np.ma.append(AA_BLS,OS_BLS)
                    SavePlot2File(StepX,StepY,[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'symbol')
                    reportName = 'Step_'+ basename + '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,np.ma.append(YrangeOSL,YrangeAAL),np.ma.append(OS_BLS,AA_BLS))

                    #Title = 'Pixels Average (OS)- ' + filename
                    Title = 'Step Overscan X-OS regions'
                    legend = 'Ext - ' + str(ps+1) + '. Region - ' + str(count)
                    picfolder = 'step/'
                    imagename = 'Step_OS_'+ basename + '_'+ str(count) + '.pdf'
                    Ylabel = 'ADU'
                    Xlabel = 'Pixels'
                    StepX = np.ma.append(YrangeOSL,YrangeAAL)
                    StepY = np.ma.append(OS_OS_Avg,ROS_X_Avg)
                    SavePlot2File(StepX,StepY,[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'symbol')
                    reportName = 'Step_OS_'+ basename + '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,np.append(YrangeOSL,YrangeAAL),np.append(OS_OS_Avg,ROS_X_Avg))

                    Title = 'Pixels Average (AA)- ' + filename
                    legend = 'Ext - ' + str(ps+1) + '. Region - ' + str(count)
                    imagename = 'Step_AA_'+ basename + '_'+ str(count) + '.pdf'
                    picfolder = 'step/'
                    Ylabel = 'ADU'
                    Xlabel = 'Pixels'
                    StepX = np.ma.append(YrangeOSL,YrangeAAL)
                    StepY = np.ma.append(ROS_Y_Avg,AA_Avg)
                    SavePlot2File(StepX,StepY,[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'symbol')
                    reportName = 'Step_AA_'+ basename + '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,np.ma.append(YrangeOSL,YrangeAAL),np.ma.append(ROS_Y_Avg,AA_Avg))

                    Title = "Histogram of AA BLS - " + filename
                    imagename = 'AA_BLS_' + basename + '_'+ str(count) + '.pdf'
                    picfolder = 'histograms/'
                    MkDir(SaveDir + picfolder)
                    Ylabel = 'Counts'
                    Xlabel = 'ADU'
                    SavePlot2File(AA_BLS_histo[1],AA_BLS_histo[0],[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'line')
                    reportName = 'AA_BLS_' + basename + '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,AA_BLS_histo[1],AA_BLS_histo[0])

                    Title = "Histogram of OS BLS - " + filename
                    imagename = 'OS_BLS_' + basename+ '_'+ str(count) + '.pdf'
                    picfolder = 'histograms/'
                    SavePlot2File(OS_BLS_histo[1],OS_BLS_histo[0],[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'line')
                    reportName = 'OS_BLS_' + basename+ '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,OS_BLS_histo[1],OS_BLS_histo[0])

                    Title = "Histogram of AA data - " + filename
                    imagename = 'AA_histo_' + basename+ '_'+ str(count) + '.pdf'
                    picfolder = 'histograms/'
                    SavePlot2File(AA_data_histo[1],AA_data_histo[0],[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'line')
                    reportName = 'AA_histo_' + basename+ '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,AA_data_histo[1],AA_data_histo[0])

                    Title = "Histogram of OS X-Y data - " + filename
                    imagename = 'OS_XY_histo_' + basename+ '_'+ str(count) + '.pdf'
                    picfolder = 'histograms/'
                    SavePlot2File(OS_OS_data_histo[1],OS_OS_data_histo[0],[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'line')
                    reportName = 'OS_XY_histo_' + basename+ '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,OS_OS_data_histo[1],OS_OS_data_histo[0])

                    Title = "Histogram of OS-X data - " + filename
                    imagename = 'OS_X_histo_' + basename+ '_'+ str(count) + '.pdf'
                    picfolder = 'histograms/'
                    SavePlot2File(RectOS_X_data_histo[1],RectOS_X_data_histo[0],[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'line')
                    reportName = 'OS_X_histo_' + basename+ '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,RectOS_X_data_histo[1],RectOS_X_data_histo[0])

                    Title = "Histogram of OS-Y data  - " + filename
                    imagename = 'OS_Y_histo_' + basename+ '_'+ str(count) + '.pdf'
                    picfolder = 'histograms/'
                    SavePlot2File(RectOS_Y_data_histo[1],RectOS_Y_data_histo[0],[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'line')
                    reportName = 'OS_Y_histo_' + basename+ '_'+ str(count)
                    ReportHisto(SaveDir + picfolder + reportName,RectOS_Y_data_histo[1],RectOS_Y_data_histo[0])

                    Title = "Normalized AA-BLS and OS-BLS - " + filename
                    imagename = 'AA_OS_' + basename + '_'+ str(count) + '.pdf'
                    picfolder = 'histograms/'
                    Ylabel = 'Counts'
                    Xlabel = 'ADU'
                    legend= ['AA_BLS','OS_BLS']
                    SavePlot2File([AA_BLS_histo[1],OS_BLS_histo[1]],[AA_BLS_histo[0]/len(AA_BLS_histo[0]),OS_BLS_histo[0]/len(OS_BLS_histo[0])],[Title,Xlabel,Ylabel,SaveDir + picfolder + imagename,legend],'line')

                    #-----------------------------------------------
                    #Reports are generated and saved.
                    #-----------------------------------------------
                    #-----------------------------------------------
                    #This report saves the coordinates of the squares
                    #-----------------------------------------------
                    reporfolder = 'reports/'
                    CoorDict = {'1. Coordinates':['x1','y1','x2','y2'],'2. Active region':RActiveArea,'3. X overscan':ROS_X,'4. Y overscan':ROS_Y,'5. X-Y overscan':RectOS_OS}
                    df = pd.DataFrame(data=CoorDict)
                    MkDir(SaveDir + reporfolder)
                    df.to_csv(SaveDir + reporfolder + 'RegionCoordinates' + '.txt')
                    datafile = open('RegionCoordinates' + '.txt','w')
                    datafile.write(df.to_string())
                    datafile.close()
                    
                    #-----------------------------------------------
                    #This section save the statistics parameters.
                    #-----------------------------------------------
                    reporfolder = 'reports/'
                    filename = SaveDir+ reporfolder + 'Stat_Report_' + basename + '.txt'
                    fileReport = open(filename,'w+')
                    TextToFile = '#---------------------------------------\n#This is a report of statistical information of\n#the linear fit performed on overscan averaged region with\n#baseline substracted (OS_BLS) and the CCD active region with the baseline\n#substracted (AA_BLS). The calculation of delta has been made using two methods:\n#using the intercept obtained with the linear fit and the average.\n\n#The idea of using both methods is that in the case where fitting is good,\n#both must converge to the same result.\n#---------------------------------------\n\n'
                    fileReport.write(TextToFile)
                    fileReport.close()
                    StatDict = {'0. Name':['OS-BLS','AA-BLS'],\
                                '1. Intercept':[OS_BLS_fit[1],AA_BLS_fit[1]],\
                                '2. Slope':[OS_BLS_fit[0],AA_BLS_fit[0]],\
                                '3. R^2':[OS_BLS_fit[2]**2,AA_BLS_fit[2]**2],\
                                '4. P-value':[OS_BLS_fit[3],AA_BLS_fit[3]],\
                                '5. chi^2/DOF':[OS_BLS_chi2,AA_BLS_chi2],\
                                '6. Std Err': [OS_BLS_fit[4],AA_BLS_fit[4]]}
                    df = pd.DataFrame(StatDict)
                    fileReport = open(filename,'a')
                    fileReport.write(df.to_string() + '\n\n\n')
                    StatDict = {'1. Method':['Average','fit'],'2. Delta':[Delta1,Delta2]}
                    df = pd.DataFrame(StatDict)
                    fileReport.write(df.to_string())
                    fileReport.close()
                    KeyDict = 'R' + str(count)
                    if KeyDict not in DictDeltaFit[lps]:
                        DictDeltaFit[lps][KeyDict] = [Delta2]
                    else:
                        DictDeltaFit[lps][KeyDict].append(Delta2)
                    if KeyDict not in DictDeltaAvg[lps]:
                        DictDeltaAvg[lps][KeyDict] = [Delta1]
                    else:
                        DictDeltaAvg[lps][KeyDict].append(Delta1)
                    if KeyDict not in DictResAvg[lps]:
                        npXaux = np.ma.copy(StepX)
                        #npXaux = np.ma.expand_dims(npXaux,axis=1)
                        npYaux = np.ma.copy(StepY)
                        #npYaux = np.ma.expand_dims(npYaux,axis=1)
                        DictResAvg[lps][KeyDict] = {'0':[npXaux],'1':[npYaux]}
                        
                    else:
                        StepXAux = np.ma.copy(StepX)
                        #StepXAux = np.ma.expand_dims(StepXAux,axis=1)
                        StepYAux = np.ma.copy(StepY)
                        #StepYAux = np.ma.expand_dims(StepYAux,axis=1)
                        DictResAvg[lps][KeyDict]['0'].append(StepXAux)
                        DictResAvg[lps][KeyDict]['1'].append(StepYAux)
                    
                    count += 1
                    pass
                
        for ps in DictResAvg:
            for region in DictResAvg[ps]:
                if  'R' in region:
                    DictResAvg[ps][region]['TotalAvg'] = sum(DictResAvg[ps][region]['1'])/len(DictResAvg[ps][region]['1'])
                    DictResAvg[ps][region]['TotalAvgAA'] = np.ma.average(DictResAvg[ps][region]['TotalAvg'][0:len(AA_BLS)])
                    DictResAvg[ps][region]['TotalAvgOS'] = np.ma.average(DictResAvg[ps][region]['TotalAvg'][len(AA_BLS):])
                    DictResAvg[ps][region]['TotalAvgStep'] = abs(DictResAvg[ps][region]['TotalAvgAA'] - DictResAvg[ps][region]['TotalAvgOS'])

        dirFinalReport = folderpath + 'Reports/'
        MkDir(dirFinalReport)
        filename = dirFinalReport + 'Delta_Report_Fit.txt'
        fileReport = open(filename,'w+')
        TextToFile = '#---------------------------------------\n#This is a report of delta calculated for each image using the method of fitting.\n#---------------------------------------\n\n'
        fileReport.write(TextToFile)
        for ps in DictDeltaFit:
            df = pd.DataFrame(DictDeltaFit[ps])
            fileReport.write(df.to_string() + '\n\n')
        fileReport.close()

        filename = dirFinalReport + 'Delta_Report_Avg.txt'
        fileReport = open(filename,'w+')
        TextToFile = '#---------------------------------------\n#This is a report of delta calculated for each image using the method of average.\n#---------------------------------------\n\n'
        fileReport.write(TextToFile)
        for ps in DictDeltaAvg:
            df = pd.DataFrame(DictDeltaAvg[ps])
            fileReport.write(df.to_string() + '\n\n')
        fileReport.close()

        

        
        # OutDict = {'1. Y-Pixels':list(DictResAvg['0']['R1'][0][:,0])}
        # OutTotal = {'Delta':['Delta']}
        for DictRes in DictResAvg:
            ExtLabel = 'Ext_'+ str(int(DictRes) + 1) + '_'
            ListKey = list(DictResAvg[DictRes].keys())
            ListKey.remove('Name')
            for Region in ListKey:
                filename = dirFinalReport + ExtLabel + Region + '_Average_Step.txt'
                fileReport = open(filename,'w+')
                TextToFile = '#---------------------------------------\n#This is a report of the average of rows in images\n#---------------------------------------\n\n' 
                fileReport.write(TextToFile)
                OutDict = {'1. Y-Pixels':DictResAvg[DictRes][Region]['0'][0],'2. Total Step Avg AA-OS':DictResAvg[DictRes][Region]['TotalAvg']}
                df = pd.DataFrame(OutDict)
                fileReport.write(df.to_string())
                fileReport.close()
                
                OutTotal = {'Delta':DictResAvg[DictRes][Region]['TotalAvgStep']}
                filename2 = dirFinalReport + ExtLabel + Region + 'Delta.txt'
                fileReport = open(filename2,'w+')
                TextToFile = '#---------------------------------------\n#This is a report of the delta\n#---------------------------------------\n\n'
                fileReport.write(TextToFile)
                df = pd.DataFrame(OutTotal,index=[0])
                fileReport.write(df.to_string())
                fileReport.close()

                Title = 'Average of AA-BLS and OS-BLS of ' + 'Ext-' + DictRes + ' ' + Region
                Xlabel = 'Pixels'
                Ylabel = 'ADU'
                legend = 'Row average'
                imagename = 'AverageResult_' + 'Ext_' + DictRes + '_' + Region + '.pdf'
                SavePlot2File(DictResAvg[DictRes][Region]['0'][0],DictResAvg[DictRes][Region]['TotalAvg'],[Title,Xlabel,Ylabel,dirFinalReport + imagename,legend],'symbol')

        return 0
    else:
        print('ERROR: The number of arguments should be 1.')
        return 1     

def findRegionsCorners(x,y):
    xaa1,xaa2,xos1,xos2 = x
    yaa1,yaa2,yos1,yos2 = y
    return {"AA":[xaa1,yaa1,xaa2,yaa2],"OS_Y":[xaa1,yos1,xaa2,yos2],"OS_X":[xos1,yaa1,xos2,yaa2],"OS_OS":[xos1,yos1,xos2,yos2]}

def validExtensions(FitsObj):
    for ext in FitsObj:
        if hasattr( ext, 'data' ):
            print(ext)
        #list(range(1,len(FitsObj),1))
    return list(range(0,len(FitsObj),1))


def histogram(argObj):
    Regions = findRegionsCorners(argObj.x,argObj.y)
    FitsObj = fits.open(argObj.histogram)
    

    if argObj.ext is None:
        extensions = validExtensions(FitsObj)
    else:
        extensions = argObj.ext

    histogramList=[]

    for ps in extensions:

        RectOS_X_mean,RectOS_X_stddev,OS_X_data,RectOS_X_mask = maskedStatistics(FitsObj[ps].data[Regions["OS_X"][0]:Regions["OS_X"][2],Regions["OS_X"][1]:Regions["OS_X"][3]]) 
        RectOS_Y_mean,RectOS_Y_stddev,OS_Y_data,RectOS_Y_mask = maskedStatistics(FitsObj[ps].data[Regions["OS_Y"][0]:Regions["OS_Y"][2],Regions["OS_Y"][1]:Regions["OS_Y"][3]])
        AA_mean,AA_stddev,AA_data,AA_mask = maskedStatistics(FitsObj[ps].data[Regions["AA"][0]:Regions["AA"][2],Regions["AA"][1]:Regions["AA"][3]])
        OS_OS_mean,OS_OS_stddev,OS_OS_data,OS_OS_mask = maskedStatistics(FitsObj[ps].data[Regions["OS_OS"][0]:Regions["OS_OS"][2],Regions["OS_OS"][1]:Regions["OS_OS"][3]])

       
        # Numbins= np.histogram_bin_edges(AA_data.flatten(),bins='fd')
        # AA_data_histo = np.histogram(AA_data.flatten(),bins=Numbins)
        if argObj.baseline:
            AA_data_copy = AA_data.copy()
            OS_X_data_copy = OS_X_data.copy()
            OS_Y_data_copy = OS_Y_data.copy()
            OS_OS_data_copy = OS_OS_data.copy()

            blAvg = OS_X_data.mean()
            AA_data -= blAvg
            OS_X_data -= blAvg
            OS_Y_data -= blAvg
            OS_OS_data -= blAvg

        bins_AA, AA_data_histo = calculateHistogram(AA_data)
        bins_OS_X, OS_X_data_histo = calculateHistogram(OS_X_data)
        bins_OS_Y, OS_Y_data_histo = calculateHistogram(OS_Y_data)
        bins_OS_OS, OS_OS_data_histo = calculateHistogram(OS_OS_data) 

        binsDict={"bins":[bins_AA,bins_OS_X,bins_OS_Y,bins_OS_OS], "data_histo":[AA_data_histo, OS_X_data_histo,OS_Y_data_histo,OS_OS_data_histo],"class":[markClass(bins_AA),markClass(bins_OS_X),markClass(bins_OS_Y),markClass(bins_OS_OS)]}
  
        fileName=argObj.histogram.split('/')[-1].split('.')[0]+'_'+str(ps)

        ShowHisto2PDF(binsDict, fileName, saveFile=argObj.pdf)
      

        # plt.hist(AA_data_histo[1][:-1] ,bins=AA_data_histo[1],weights=AA_data_histo[0])
        # plt.show()
            #guardar la imagen en PDf

        # Step = 5
        # Range = range(int(math.floor(OS_BLS.min())),int(math.ceil(OS_BLS.max())),Step)
        # Numbins=len(Range)# + 1
        # OS_BLS_histo = np.histogram(OS_BLS,range=[Range.start,Range.stop],bins=Numbins)
        # Step = 5
        # Range = range(int(math.floor(AA_BLS.min())),int(math.ceil(AA_BLS.max())),Step)
        # Numbins=len(Range)# + 1
        # AA_BLS_histo = np.histogram(AA_BLS,range=[Range.start,Range.stop],bins=Numbins)
        # Step = 10
        # Numbins=len(Range)# + 1
        # Range = range(int(math.floor(AA_data.min())),int(math.ceil(AA_data.max())),Step)
        # AA_data_histo = np.histogram(AA_data,range=[Range.start,Range.stop],bins=Numbins)
        # Step = 10
        # Numbins=len(Range)# + 1
        # Range = range(int(math.floor(OS_OS_data.min())),int(math.ceil(OS_OS_data.max())),Step)
        # OS_OS_data_histo = np.histogram(OS_OS_data,range=[Range.start,Range.stop],bins=Numbins)
        # Step = 10
        # Numbins=len(Range)# + 1
        # Range = range(int(math.floor(RectOS_X_data.min())),int(math.ceil(RectOS_X_data.max())),Step)
        # RectOS_X_data_histo = np.histogram(RectOS_X_data,range=[Range.start,Range.stop],bins=Numbins)
        # Step = 10
        # Numbins=len(Range)# + 1
        # Range = range(int(math.floor(RectOS_Y_data.min())),int(math.ceil(RectOS_Y_data.max())),Step)
        # RectOS_Y_data_histo = np.histogram(RectOS_Y_data,range=[Range.start,Range.stop],bins=Numbins)




    return 0

def charge(argObj):
    return 0

def darkCurrent(argObj):
    return 0

def eventDetect(argObj):
    return 0

def main(argObj):
    #histograma
    #charge
    #darkCurrent
    #eventDetect
    #-----------
    #skp
    
    #-----------
    #pdf
    #baseline
    


    if argObj.histogram is not None:
        exitcode = histogram(argObj)
        return exitcode
    elif argObj.charge is not None:
        exitcode = charge(argObj)
        return exitcode
    elif argObj.dCurrent is not None:
        exitcode = darkCurrent(argObj)
        return exitcode
    elif argObj.eventDet is not None:
        exitcode = eventDetect(argObj)
        return exitcode


if __name__ == "__main__":
    argObj = parser()
    exitcode = main(argObj)
    exit(code=exitcode)      






