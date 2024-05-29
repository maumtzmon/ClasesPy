import numpy as np
import math
from scipy.optimize import curve_fit
import argparse
import sys
import os
import warnings
import datetime
from scipy.optimize import OptimizeWarning
from scipy.signal import find_peaks
warnings.simplefilter("error", OptimizeWarning) # If the fit are not good get an error
warnings.simplefilter("error", RuntimeWarning) # for overflow in exponentials and ivalid values get an error

def gauss(x,x0,s,A):
    return A*np.exp(-(x-x0)**2/(2*s**2))/np.sqrt(np.pi*2*s**2)

def ax(x,a,b):
    return a*x+b

def Get_GlobalGain(data_adu,gain_list,noise_list,path_cal,verbosity=False,acds=0, chid=0):
    
    # Filtering images w high fluctuation parameters
    gain_in=np.argwhere(abs(np.array(gain_list)-np.median(gain_list))/np.median(gain_list)<0.05).flatten().tolist()
    noise_in=np.argwhere(abs(np.array(noise_list)-np.median(noise_list))/np.median(noise_list)<0.5).flatten().tolist()
    inter_ind=np.intersect1d(gain_in,noise_in).flatten().tolist()
#     print('Len data adu:', len(data_adu))
#     print('Gain_id: ',gain_in,'type:',type(gain_in))
#     print('Noise_ind: ',noise_in,'type:',type(noise_in))
#     print('inter_ind: ',inter_ind, 'type:',type(inter_ind))
    
    RunData_adu=[]
    for i in inter_ind:
        RunData_adu+=data_adu[i]
    
    RunGain_list=[gain_list[i] for i in inter_ind]
    RunNoise_list=[noise_list[i] for i in inter_ind]
#     print('Len run data adu:', len(RunData_adu))
    #------------------------------------------------------------------------------------------------
    work_dir=os.getcwd()
    global_hist, xb = np.histogram(RunData_adu, bins = np.arange(-250, np.median(RunGain_list)*100+250, 1) ) 
    x = (xb[1:]+xb[:-1])/2
    
    global_sigma=np.median(RunNoise_list)*np.median(RunGain_list)
    peaks, prop = find_peaks(global_hist, distance=4*global_sigma, height=10)
#     print('RunGainList median: ', np.median(RunGain_list))
#     print('RunNoise_list median: ',np.median(RunNoise_list))
#     print('Global Sigma: ', global_sigma)
    trunc=abs(np.diff(peaks)-np.median(np.diff(peaks)))/np.median(np.diff(peaks))
    med=[]
    amp=[]
    
    pk=0
    electron_list=[0] #electron number correspondig to each peak 
    for t in trunc.round():
        elec_pk=pk+1+t
        pk=elec_pk
        electron_list.append(elec_pk)
        
    amp=list(prop['peak_heights']) # initial paramter to curve_fit
    med=list(x[peaks])  # initial paramter to curve_fit
    
    #Final lists to calibration fit linear
    means_fit=[]
    electrons_fit=[]
    err_means_fit=[]
    out_peaks='PeaksParams_'+str(acds)+'_'+str(chid)+'.csv'
    
    if not os.path.isfile(path_cal+'/'+out_peaks):
        os.chdir(path_cal)
        with open(out_peaks, "w") as o:
            o.write("#peak, mean, mean_err, sigma, sigma_err, amp, amp_err\n")
    else: 
        print('\033[93m'+'The File >> '+out_peaks+' << already exists.'+'\033[0m')
        out_peaks='PeaksParams'+datetime.datetime.now().strftime('%m-%d-%Y-%Hh%M')+'_'+str(acds)+'_'+str(chid)+'.csv'
        if verbosity: print('The calibration data will be storage in: ',out_peaks)
        os.chdir(path_cal)
        with open(out_peaks, "w") as o:
            o.write("#peak, mean, mean_err, sigma, sigma_err, amp, amp_err\n")
    
    
    if verbosity: print('\nCalibration process started:\n')
    
    for k in range(len(med)):
        y,xb=np.histogram(RunData_adu,bins=np.arange(med[k]-global_sigma*3,med[k]+global_sigma*3,1))
        x=(xb[1:]+xb[:-1])/2
        try:
            popt,pcov=curve_fit(gauss,x,y,p0=[med[k],global_sigma,amp[k]])
            means_fit.append(popt[0])
            err_means_fit.append(np.sqrt(np.diag(pcov))[0])
            electrons_fit.append(electron_list[k])
            # progress bar
            if verbosity: 
                sys.stdout.write('\r')
                sys.stdout.write("[%-98s] %d%%" % ('='*int(k*100/len(med)-1), (100)/(len(med)-1)*k))
                sys.stdout.flush()
            
            perr = np.sqrt(np.diag(pcov))
            with open(out_peaks, "a") as o:
                o.write( f"{electron_list[k]}, {popt[0]}, {perr[0]}, {abs(popt[1])}, {perr[1]}, {popt[2]}, {perr[2]}\n" )
        except (RuntimeError,OptimizeWarning,RuntimeWarning):
            continue
            
    os.chdir(work_dir)     
    try:
        popt_g, pcov_g = curve_fit(ax, electrons_fit, means_fit,sigma=err_means_fit, absolute_sigma=True)
    
    except (RuntimeError,OptimizeWarning,RuntimeWarning):
        print(f"Error - global gain fit failed")
        sys.exit()
        
    perr=np.sqrt(np.diag(pcov_g))
    
    if verbosity: 
        print("")
        print("Global Gain a : ",popt_g[0],"+-",perr[0]," ADU/e-")
        print("Global Gain b : ",popt_g[1],"+-",perr[1]," ADU")
    return (popt_g[0],popt_g[1]),len(electrons_fit)
