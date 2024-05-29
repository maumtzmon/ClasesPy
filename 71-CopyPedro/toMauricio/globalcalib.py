from functions_py import *

def GlobalCalib(RunData_adu, RunGain_list, RunNoise_list ,csv_file,extensions=4):
    if extensions!=1:
        list_param=[]
        list_peaks=[]
        for j in range(extensions):
            global_sigma=RunNoise_list[j]*RunGain_list[j]
            global_hist, xb = np.histogram(RunData_adu[j], bins = np.arange(-3*global_sigma, RunGain_list[j]*100+3*global_sigma, 1) ) 
            x = (xb[1:]+xb[:-1])/2
            peaks, prop = find_peaks(global_hist, distance=4*global_sigma, height=10)
            trunc=abs(np.diff(peaks)-np.median(np.diff(peaks)))/np.median(np.diff(peaks))
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
            out_peaks=csv_file.replace('.csv','chid'+str(j)+'.csv')
            with open(out_peaks, "w") as o:
                o.write("#peak, mean, mean_err, sigma, sigma_err, amp, amp_err\n")
            for k in range(len(med)):
                y,xb=np.histogram(RunData_adu[j],bins=np.arange(med[k]-global_sigma*2,med[k]+global_sigma*2,1))
                x=(xb[1:]+xb[:-1])/2
                try:
                    popt,pcov=curve_fit(gauss,x,y,p0=[med[k],global_sigma,amp[k]])
                    means_fit.append(popt[0])
                    err_means_fit.append(np.sqrt(np.diag(pcov))[0])
                    electrons_fit.append(electron_list[k])
                    perr = np.sqrt(np.diag(pcov))
                    with open(out_peaks, "a") as o:
                        o.write( f"{electron_list[k]}, {popt[0]}, {perr[0]}, {abs(popt[1])}, {perr[1]}, {popt[2]}, {perr[2]}\n" )
                except (RuntimeError,OptimizeWarning,RuntimeWarning):
                    continue
            try:
                popt_g, pcov_g = curve_fit(ax, electrons_fit, means_fit,sigma=err_means_fit, absolute_sigma=True)

            except (RuntimeError,OptimizeWarning,RuntimeWarning):
                print(f"Error - global gain fit failed")
                sys.exit()

            perr=np.sqrt(np.diag(pcov_g))


            print("/n CHID "+str(j))
            print("Global Gain a : ",popt_g[0],"+-",perr[0]," ADU/e-")
            print("Global Gain b : ",popt_g[1],"+-",perr[1]," ADU")
            list_param.append((popt_g[0],popt_g[1]))
            list_peaks.append(len(electrons_fit))
    return list_param,list_peaks


for path in glob.glob('/home/zilvespedro/work/MICROCHIP/MANA/AnaParamsCSV/AnaParams_2x*_1.csv'):

    data=pd.read_csv(path,na_values=[' nan',' inf',-1000])
    data=data.dropna(how='any')
    RunGain_list=[np.median(data[data['CHID']==i]['Gain'].values) for i in range(4)]
    RunNoise_list =[np.median(data[data['CHID']==i]['Noise'].values) for i in range(4)]
    rundata_path=path.replace('/home/zilvespedro/work/MICROCHIP/MANA/AnaParamsCSV/AnaParams_',
                              '/home/zilvespedro/work/MICROCHIP/MANA/dataADU').replace('_1.csv','.npy')
    RunData_adu = np.load(rundata_path)
    csv_file=path.replace('/home/zilvespedro/work/MICROCHIP/MANA/AnaParamsCSV/AnaParams',
                              '/home/zilvespedro/work/MICROCHIP/MANA/PeaksParams').replace('_1.csv','.csv')
    list_param,list_peaks = GlobalCalib(RunData_adu, RunGain_list, RunNoise_list ,csv_file,extensions=4)
    
    dic={'Params(a,b)':list_param,'N peaks':list_peaks}
    df=pd.DataFrame(dic)
    
    df.to_csv(rundata_path.replace('.npy','.csv'),index=False)
    