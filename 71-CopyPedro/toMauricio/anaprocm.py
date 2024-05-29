from functions_py import *

def AnaProcM(path,v=False):
    csv_name='AnaParams'+path.replace('/share/storage2/connie/data/microchip/proc_bkgd', '')+'.csv'

    sigma_list=[]
    sigma_err_list=[]
    gain_list=[]
    gain_err_list=[]
    ser_list=[]
    ser_err_list=[]
    imgID_list=[]
    rd_time_list=[]
    chids=[]
    
    for path in glob.glob(path+'/proc_skp_*.fits'):
        imgID = re.findall( "_img([0-9]+).fits", path )[0]
        hdu_list = fits.open(path)
        readout_time = GetRDtime(hdu_list)
        data_pre = precal(hdu_list,extensions=4)
        gain, gain_err, data= LocalCalib(data_pre,extensions=4)
        sigma, sigma_err = LocalSigma(data,extensions=4)
        
        mask_bleeding_array, HE_events_array = mask_bleeding(data,direction='xy',iterations=[40,20],extensions=4,he_th=80)
        SRE_mask_array=mask_SRE(DataMminus(data,(mask_bleeding_array+HE_events_array)),extensions=4,SRE_th=1.5)
        GlobalMask=np.array([32*mask_bleeding_array.astype('bool'),128*SRE_mask_array.astype('bool')]).max(axis=0)

        event_halo_SR_mask = np.array(GlobalMask,dtype=bool)|EventHalo_Mask(data,mask_bleeding_array).astype(bool)
        ser,ser_err = LocalSER(data,event_halo_SR_mask,sigma,readout_time)
        
        sigma_list+=sigma
        sigma_err_list+=sigma_err
        gain_list+=gain
        gain_err_list+=gain_err
        ser_list+=ser
        ser_err_list+=ser_err
        imgID_list+=[imgID]*4
        rd_time_list+=[readout_time]*4
        chids+=[0,1,2,3]
        if v==True:
            for i in range(4):
                print("CHID "+str(i)+' Gain:({:.2f} +- {:.2f})ADUs'.format(gain[i],gain_err[i])+r'| Noise: ({:.3f} +- {:.3f}) e-'.format(sigma[i],sigma_err[i])+r'| SER: ({:.3f} +- {:.3f}) e-/pix/day'.format(ser[i],ser_err[i]))
    dict={'CHID':chids, 'IMGID':imgID_list, 'Gain': gain_list, 'Gain_err': gain_err_list, 'Noise': sigma_list, 'Noise_err': sigma_err_list, 'Ser': ser_list, 'Ser_err': ser_err_list, 'ReadoutTime': rd_time_list}
    df=pd.DataFrame.from_dict(dict)
    df.to_csv(csv_name,index=False)
    print('Done!')
    
for p in sorted(glob.glob('/share/storage2/connie/data/microchip/proc_bkgd_2x*')): AnaProcM(p,v=True)
    
# for p in ['/share/storage2/connie/data/microchip/proc_bkgd_2x2']: AnaProcM(p,v=True)