from ast import arg
from re import M
from zlib import Z_DEFAULT_STRATEGY
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftn

from astropy.io.fits.hdu import image
from astropy.io import fits

import argparse

path_def='/home/mauricio/datosFits/27JUL2022/lta_disconected/'
file_def='proc_skp_mod9_wfr13_Vvtest_vthswings2.5_sh2.5_oh-2.5_EXPOSURE0_NSAMP1_NROW700_NCOL800_img6.fits'

#jsonTest '/home/mauricio/datosFits/27JUL2022/lta_disconected/proc_skp_mod9_wfr13_Vvtest_vthswings2.5_sh2.5_oh-2.5_EXPOSURE0_NSAMP1_NROW700_NCOL800_img6.fits'



def parser():
    parser = argparse.ArgumentParser(prog="FITSanlyzer", description='High particle FITS images analyzer. \n It looks for spurious signals embedded in CCDs images ')

    
    parser.add_argument('-f', type=str, nargs = '+',help='file or path of the images')
    #parser.add_argument('-f','--pdf',action='store_true',help="Save results in PDF file.")
    parser.add_argument('--ext', type=int, nargs='+', help= 'Indicate which extentions do you want to use' )
    parser.add_argument('--nsamp', type=int, nargs = 1, help = 'number of skipping cycles of the images' )
    parser.add_argument('--td',action='store_true', help='plot a 3d graph of the image')
    subparsers=parser.add_subparsers(help="sub command Help")

    ###
    # Positional Arguments
    ###
    sub_action_1 = subparsers.add_parser('', help = '')

    #run skipped to regular imege function

    sub_action_2 = subparsers.add_parser('other_subOPT', help = 'help')
    sub_action_2.add_argument('Variable_PATH', type=int, help='help')

    sub_action_2.add_argument('--XX',action='store_true', help='plot a 3d graph of the image')
    
    #run function

    ###
    # Optional arguments
    ###
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ZZ',action='store_true', help='plot a 3d graph of the image')
        
    #output option

    argObj = parser.parse_args()
    return argObj

def spectralPlot(Z_data=1,fftFreqZ_row=1,fftFreqZ_col=1,fft_Z_row=1,fft_Z_col=1,col=1,row=1,n_cols=1,n_rows=1):
    

    fig, ax=plt.subplots(2,2,constrained_layout=True)
    fig.canvas.manager.set_window_title('Spectral Analysis')
    # row analysis
    ax[0,0].set_xlabel('Columns')
    ax[0,0].set_ylabel('ADUs')
    ax[0,0].set_title('Horizontal')
    #ax[0,0].locator_params(nbins=5)
    ax[0,0].plot(Z_data[row])
    

    ax[1,0].set_ylabel('Signal Amplitud')
    ax[1,0].set_xlabel('Frecuency [Hz]')
    #ax[1,0].locator_params(nbins=5)
    ax[1,0].plot(np.abs(fftFreqZ_row),(np.abs(fft_Z_row))/n_cols)
    
    # column analysis
    ax[0,1].set_xlabel('Rows')
    ax[0,1].set_title('Vertical')
    #ax[0,1].locator_params(nbins=5)
    ax[0,1].plot(Z_data[:,col])
    
    ax[1,1].set_xlabel('Frecuency [Hz]')
    #ax[1,1].locator_params(nbins=5)
    ax[1,1].plot(np.abs(fftFreqZ_col),(np.abs(fft_Z_col))/n_rows)

    plt.show()

def plot3d(X,Y,m_data,Z_data):
    
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    fig.canvas.manager.set_window_title('3d projection')
    
    #ax.set_title("")
    ax.set_xlabel("NColumn in X")
    ax.set_ylabel("NRows in Y")
    ax.set_zlabel("ADUs in Z")
    
    surf=ax.plot_surface(X,Y,m_data.filled(fill_value=np.mean(Z_data)),cmap='gray') #cmap='plasma' looks good
    fig.colorbar(surf, shrink=0.5)
    plt.show()

def main(argObj):
    
    ###
    #  Extention section
    ###
    #0-3 fits; 1-4 fz
    file    =argObj.f[0]
    plot_3d =argObj.td
    #nsamp   =argObj.nsamp[0]
    ext     =argObj.ext[0]

    if file.endswith('.fz'):
        if ext==0:
            ext=1
        
    if file.endswith('.fits'):
        if ext==4:
            ext=3

    ###
    # Adjust the shape automatically
    ###
    hdul=fits.open(file)# fits file to analyze
    Z_data=hdul[ext].data #numpy array #choose the extention you want to use
    n_rows,n_cols=hdul[ext].shape
    x_data=np.arange(0,n_cols,1)  
    y_data=np.arange(0,n_rows,1)
    X, Y =np.meshgrid(x_data,y_data) #creates a grid with the axis on XY plain


    ###
    # mask values over the mean
    ###
    #
    m_data=np.ma.masked_where(~((np.mean(Z_data*.95)<Z_data)&(Z_data<np.mean(Z_data*1.05))), Z_data) #To do

    fig=plt.figure()
    fig.canvas.set_window_title(file)
    fitsImage=plt.imshow(m_data.filled(fill_value=np.mean(Z_data)),cmap='bone')
    fig.colorbar(fitsImage, shrink=0.5)
    plt.show()


    ###
    # Plot 3d Image
    ###
    if plot_3d == True: plot3d(X,Y,m_data,Z_data)

    ###
    # Fourier Analysis
    ###

    timestep_pix_col=.0001               #intervalo en segundos de cada pixel en X
    timestep_pix_row=.1                  #intervalo en segundos de cada pixel en y
                                         #fix time pix row acording number of pix in columns by time clock


    row=205                                                 #which row Default value = 205
    fft_Z_row=np.fft.fft(a=Z_data[row,:])                     #Amplitud de la espiga
    """
    Calcule la transformada de Fourier discreta unidimensional. Esta función calcula la transformada 
    discreta de Fourier (DFT) unidimensional de n puntos con el eficiente algoritmo de transformada 
    rápida de Fourier (FFT) [CT].
    a : arreglo, señal de entrada

    """


    fftFreqZ_row=np.fft.fftfreq(n_cols,d=timestep_pix_col)  #frecuencia de la espiga
    """
    np.fft.fftfreq(n,d)
    n: longitud de la ventana, numero de muestras (pixeles en este caso)
    d: duracion entre cada muestra
    
    E.g. freq = np.fft.fftfreq(n, d=timestep) #nuestro stimestep es segundos

    Devuelve las frecuencias de muestra de la transformada discreta de Fourier. La matriz flotante 
    devuelta f contiene los centros de intervalos de frecuencia en ciclos por unidad de espaciado d
    de muestra (con cero al principio). Por ejemplo, si el espaciado de la muestra está en segundos, 
    entonces la unidad de frecuencia es ciclos/segundo.
    """

    col=250                                                 #which col 
    fft_Z_col=np.fft.fft(Z_data[:,col])
    fftFreqZ_col=np.fft.fftfreq(n_rows,d=timestep_pix_row)

    spectralPlot(Z_data,fftFreqZ_row,fftFreqZ_col,fft_Z_row,fft_Z_col,col,row,n_cols,n_rows)
    


if __name__ == "__main__":
    argObj = parser()
    exitcode = main(argObj)
    exit(code=exitcode)  