import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftn
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from astropy.io.fits.hdu import image
from astropy.io import fits
import sys
import os


#detlaT = tiempo de lectura ¿como calcularlo?
#Generar histograma
#crear mascara de los pixeles que esten fuera de la media
#asignar tiemṕo de exposicion de cada píxel
#Leer la columna, sacar la pendiente, el eje y es la carga y el eje x es el tiempo de exposicion, b = energia en 0


def func(x, m, b):
    return (m*x+b)

def expoCharge():
    
    return 0

# clean example
def getSpurious(filePath):
    spuriousFactor={}

    fig_all, axs_all = plt.subplots(2, 4, figsize=(20, 10))		# Create figures

    hdul=fits.open(filePath)# fits file to analyze

    for ext in range(0, len(hdul)):
        data=hdul[ext].data #numpy array
        header=hdul[ext].header
        x=data[10:640,0] #analisis de registro vertical, Columnas V

        bins=np.histogram_bin_edges(x, bins='fd')
        data_histo, bins = np.histogram(x,bins=bins) 
        u, std = norm.fit(x)
        print("for ext=["+str(ext)+"] => media="+str(u)+", stdDev="+str(std))

        xdata=[]
        ydata=[]
        i=0
        for data in x:  
            ydata.append(data)
            xdata.append(i)
            i+=1

        Xdata=np.array(xdata)
        Ydata=np.array(ydata)

        #Create a mask using spectrum values
     
        m_data=np.ma.masked_where(((Ydata < (np.mean(Ydata)- 1.5*std)) | (Ydata > (np.mean(Ydata)+1.5*std))), Ydata) #To do
        axs_all[0][ext].plot(Xdata, m_data)
        popt, pcov = curve_fit(func, Xdata, m_data) #ajustar valores de x y yRuido a la funcion "func"

        #axs_all[0][ext].plot(Xdata, Ydata)
        #popt, pcov = curve_fit(func, Xdata, Ydata)

        axs_all[0][ext].plot(Xdata, func(Xdata, popt[0], popt[1]), 'r-', label='fit: m=%5.3f, b=%5.3f' % tuple(popt)) #plot de xdata vs f(xdata)
        axs_all[0][ext].plot(np.mean(Ydata), label="mean=%5.3f" % np.mean(Ydata) )
        axs_all[0][ext].set_title('hdul=%1i getSpurious' % ext)
        axs_all[0][ext].set_xlabel('row')
        axs_all[0][ext].legend()

        spuriousFactor['row']=popt

        
    axs_all[0][0].set_ylabel('ADU, plot of column')

    #plt.show()

    for ext in range(0, len(hdul)):
        data=hdul[ext].data #numpy array
        header=hdul[ext].header
        x=data[10,10:690] #analisis de Registro horizontal, renglones H

        xdata=[]
        ydata=[]
        i=0
        for data in x:  
            ydata.append(data)
            xdata.append(i)
            i+=1

        Xdata=np.array(xdata)
        Ydata=np.array(ydata)

        m_data=np.ma.masked_where(((Ydata < (np.mean(Ydata)- 1.5*std)) | (Ydata > (np.mean(Ydata)+1.5*std))), Ydata) #To do
        axs_all[1][ext].plot(Xdata, m_data)
        popt, pcov = curve_fit(func, Xdata, m_data) #ajustar valores de x y yRuido a la funcion "func"

        #axs_all[1][ext].plot(Xdata, Ydata)
        #popt, pcov = curve_fit(func, Xdata, Ydata)

        axs_all[1][ext].plot(Xdata, func(Xdata, popt[0], popt[1]), 'r-', label='fit: m=%5.3f, b=%5.3f' % tuple(popt)) #plot de xdata vs f(xdata)
        axs_all[1][ext].plot(np.mean(Ydata), label="mean=%5.3f" % np.mean(Ydata) )
        #axs_all[1][ext].set_title('hdul=%1i' % ext)
        axs_all[1][ext].set_xlabel('column')
        axs_all[1][ext].legend()


        
    axs_all[1][0].set_ylabel('ADU, plot of row')
    plt.show()

    return spuriousFactor

def hist_RowColumn(filePath):
    fig_all, axs_all = plt.subplots(2, 4, figsize=(20, 10))		# Create figures

    hdul=fits.open(filePath)# fits file to analyze

    for ext in range(0, len(hdul)):
        data=hdul[ext].data #numpy array
        header=hdul[ext].header
        x=data[10:640,0] #analisis de registro vertical, Columnas V

        bins=np.histogram_bin_edges(x, bins='fd')
        data_histo, bins = np.histogram(x,bins=bins) 
        u, std = norm.fit(x)
        print("for ext=["+str(ext)+"] => media="+str(u)+", stdDev="+str(std))
    
        axs_all[0][ext].hist(x, bins, density=True)
        #axs_all[0][ext].
        axs_all[0][ext].set_title('hdul=%1i hist RowColumn' % ext)
        axs_all[0][ext].set_xlabel('ADU, plot of column')
        #axs_all[0][ext].legend()
    
    axs_all[0][0].set_ylabel('counts')

    for ext in range(0, len(hdul)):
        data=hdul[ext].data #numpy array
        header=hdul[ext].header
        x=data[10,10:690] #analisis de Registro horizontal, renglones H

        bins=np.histogram_bin_edges(x, bins='fd')
        data_histo, bins = np.histogram(x,bins=bins) 
        u, std = norm.fit(x)
        print("for ext=["+str(ext)+"] => media="+str(u)+", stdDev="+str(std))
    
        axs_all[1][ext].hist(x, bins, density=True)
        #axs_all[0][ext].
        axs_all[1][ext].set_title('hdul=%1i' % ext)
        axs_all[1][ext].set_xlabel('ADU, plot of row')
        #axs_all[1][ext].legend()
    
    axs_all[0][0].set_ylabel('counts')

    #plt.show()


def totTime(filePath):
    hdul=fits.open(filePath)# fits file to analyze
    header=hdul[0].header
    tStartList=str(header._cards[159]).split("'")[1].split('T')[1].split(':')
    tEndList=str(header._cards[160]).split("'")[1].split('T')[1].split(':')

    tStart=int(tStartList[0])*3600+int(tStartList[1])*60+int(tStartList[2])
    tEnd=int(tEndList[0])*3600+int(tEndList[1])*60+int(tEndList[2])
    Ttot=tEnd-tStart

    NRow=int(str(header._cards[15]).split("'")[1])
    NCol=int(str(header._cards[16]).split("'")[1])
    NSamp=int(str(header._cards[17]).split("'")[1])


    deltaTperPix=Ttot/(NCol*NRow)
    deltaTperRow=Ttot/NRow

    expoTimes=[]
    
    for mRow in range(0,NRow):  #Fill Exposure Matrix
        expoTimes.append([])
        for nCol in range(0,NCol):
            expoTimes[mRow].append(deltaTperRow*mRow+deltaTperPix*nCol)


    ExpoMatrix=np.array(expoTimes)
        
    
    return ExpoMatrix

def main():
    deltaTvertical=.01 #valor de prueba, hay que verificar ese valor para la exposicion de cada pixel en la vertical



    try:
        filePath=sys.argv[1]
        getSpurious(filePath)
    except:
        # path='/home/oem/datosFits/testMITLL/20DIC22/'
        # file="proc_skp_module24_MITLL01_externalVr-5_Vtest_T170_testLeakage_vtest_vdd-22__NSAMP9_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img29.fits"

        path='/home/oem/datosFits/testMITLL/11ENE23/'
        file="proc_skp_module24_MITLL01_externalVr-5_Vtest_T170_testLeakage_vtest_vdd-22__NSAMP81_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img21.fits"
        filePath=path+file
        hist_RowColumn(filePath)
        getSpurious(filePath)
        ExpoMatrix=totTime(filePath)
        popt, pcov = curve_fit(func, range(0, len(ExpoMatrix[0])),ExpoMatrix[0]) #ajustar valores de x y yRuido a la funcion "func"
        plt.plot(range(0,len(ExpoMatrix[0])), func(range(0,len(ExpoMatrix[0])), popt[0], popt[1]),'r-', label='fit: m=%5.3f, b=%5.3f' % tuple(popt))
        popt, pcov = curve_fit(func, range(0, len(ExpoMatrix[:,0])),ExpoMatrix[:,0])
        plt.plot(range(0,len(ExpoMatrix[:,0])), func(range(0,len(ExpoMatrix[:,0])), popt[0], popt[1]),'b-', label='fit by column: m=%5.3f, b=%5.3f' % tuple(popt))
        plt.legend()
        plt.show()
    
    




if __name__ == "__main__":
    exitcode = main()
    exit(code=exitcode) 