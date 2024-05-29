from astropy.io import fits as fits
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import numpy as np

def plotHistogram(data):
    histo, bins_edges = np.histogram(data,bins='fd',range=(-200,300))
    class_marks = (bins_edges[:-1]+bins_edges[1:])/2
    #plt.hist(class_marks,bins=bins_edges,weights=histo)
    return histo, bins_edges, class_marks

def gaussian2(x,m1,s,a1,g,a2):
    return a1*np.exp(-1/2*((x-m1)/s)**2)+a2*np.exp(-1/2*((x-m1-g)/s)**2)


file_list=['/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP1_18.fits',
           '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP10_19.fits',
           '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP20_20.fits',
           '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP40_21.fits',
           '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP60_22.fits',
           '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP80_23.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP100_24.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP200_25.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP300_26.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP400_27.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP500_28.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP600_29.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP700_30.fits',
        #    '/home/oem/datosFits/mcm_data/ansamp/fits/MCM1_Demuxed_Test_barrido_ANSAMP_exp_1s_ignorando_2muestras_ANSAMP800_31.fits'
        ]

noise_list=[]

for file in file_list:
        
    file_FITS=fits.open(file)
    ext=2
    datos=file_FITS[ext].data[:,:1100]
    mediana=np.median(file_FITS[ext].data[:,:1100])
    ANSAMP=int(file_FITS[ext].header['ANSAMP'])
    print('mediana:'+str(mediana)+'\nANSAMP'+str(ANSAMP))

    data=(datos-mediana)/ANSAMP
    mediana_2=np.median(data)
    print('mediana:'+str(mediana_2))
    
    histo, bins, class_marks=plotHistogram(data)

    popt,pcov=curve_fit(gaussian2,class_marks,histo,p0=[[0,3,1500, 44, 100]]) ##data, mean, stdDev, h1, gain, h2
    popt=abs(popt)
    
    fig, (img, hist)=plt.subplots(ncols=2, figsize=(12, 4))
    
    left=img.imshow(data)
    fig.colorbar(left,ax=img,orientation='horizontal')

    right=hist.bar(class_marks,histo)
    right=hist.plot(class_marks,gaussian2(class_marks,*popt),linewidth=1,c='r', label=r'$\sigma$={:.2f}  gain={:.2f}'.format(popt[1],popt[3]),)
    #hist.ylim(1,3e3)
    plt.yscale('log')
    plt.ylim(1,25e3)
    plt.legend()
    #plt.show()
    plt.close()
    noise_list.append(popt[1])
    
for noise in noise_list:
    print(r'noise={:.2f}'.format(noise))

x=[1,10,20,40,60,80]
plt.plot(x,noise_list)
plt.show()