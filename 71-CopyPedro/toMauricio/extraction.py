from functions_py import *

def extraction(data,mask,NAC_list, EXP_list,imgID,extensions=4,thr=4):
    if extensions!=1:
        catalog_list=[]
        for i in range(extensions):
            
            catalog=ndimage.label(data[i]>thr,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]
            rps=sk.regionprops(catalog,intensity_image=data[i],cache=False, extra_properties=[sum_intensity])
            energy=[r.sum_intensity for r in rps]
            centr=[r.weighted_centroid for r in rps]
            bbox=[r.bbox for r in rps]
            
            ymin=[r.bbox[0] for r in rps]
            xmin=[r.bbox[1] for r in rps]
            ymax=[r.bbox[2] for r in rps]
            xmax=[r.bbox[3] for r in rps]
            
            flag_list=[]
            ybary=[int(r.weighted_centroid[0]) for r in rps]
            xbary=[int(r.weighted_centroid[1]) for r in rps]
            
            for k in range(len(centr)):
                bary=tuple(int(item) for item in centr[k])
                flag=mask[i][bary[0],bary[1]]
                flag_list.append(flag)
                 
            catalog_prop={'ImgID':[imgID]*len(centr),'CHID':[i]*len(centr),
                          'Event':np.arange(1,len(centr)+1,1).tolist(),
                          'Energy':energy,
                          'Flag':flag_list,
                          'xBary':xbary,
                          'yBary':ybary,
                          'yMin':ymin,
                          'xMin':xmin,
                          'yMax':ymax,
                          'xMax':xmax,
                          'NpixAC':[NAC_list[i]]*len(centr),
                          'ExpTime':[EXP_list[i]]*len(centr)}
            df=pd.DataFrame(catalog_prop)
            catalog_list.append(df)
            
    return pd.concat(catalog_list)

def CalibGB(data_adu,GBparams,extensions=4):
    if extensions!=1:
        data_cal = np.zeros_like(data_adu)
        for k in range(extensions):
            data_cal[k] = (data_adu[k]-GBparams[k][1])/GBparams[k][0]
    return data_cal

def Get_exposuretime(readout_time,mask,extensions=4):
    NAC_list=[]
    EXP_list=[]
    if extensions!=1:
        
        for k in range(extensions):
            global_mask=mask[k]
            Npix=global_mask.shape[0]*global_mask.shape[1]
            time_map=np.linspace(0,readout_time,num=Npix,endpoint=True).reshape(global_mask.shape[0],global_mask.shape[1])
            time_map_mask=ma.masked_where( global_mask.astype(bool), time_map)
            Nactive=ma.count(time_map_mask)
            exposure_time=ma.mean(time_map_mask)
            NAC_list.append(Nactive)
            EXP_list.append(exposure_time)
            
    return NAC_list, EXP_list

import ast
GBparamsDic={}
data_set_name=['1x10','1x2','1x4','2x1','2x2']
calibparam=['/home/zilvespedro/work/MICROCHIP/MANA/dataADU'+name+'.csv' for name in data_set_name]
for k in range(5):
    globalpar=pd.read_csv(calibparam[k],usecols=['Params(a,b)'],converters={'Params(a,b)': ast.literal_eval})
    GBparamsDic[data_set_name[k]]=(list(globalpar['Params(a,b)']))
    
def RunCatalog(path_run,csvname,GBparams,seed):

    catalogs_run=[]
    for path in glob.glob(path_run+'/proc_skp_*.fits'):
        imgID = re.findall( "_img([0-9]+).fits", path )[0]
        hdu_list = fits.open(path)
        readout_time = GetRDtime(hdu_list)
        data_pre = precal(hdu_list,extensions=4)
#         gain, gain_err, data= LocalCalib(data_pre,extensions=4)
        data = CalibGB(data_pre,GBparams,extensions=4)
    
        mask_bleeding_array, HE_events_array = mask_bleeding(data,direction='xy',iterations=[40,20],extensions=4,he_th=80)
        SRE_mask_array=mask_SRE(DataMminus(data,(mask_bleeding_array+HE_events_array)),extensions=4,SRE_th=1.5)
        GlobalMask=np.array([32*mask_bleeding_array.astype('bool'),128*SRE_mask_array.astype('bool')]).max(axis=0)
        
        NAC_list, EXP_list = Get_exposuretime(readout_time,GlobalMask,extensions=4)
        
        catalog=extraction(data,GlobalMask,NAC_list, EXP_list,imgID=imgID,extensions=4,thr=seed)
        catalogs_run.append(catalog)

    catalog_run_save=pd.concat(catalogs_run)
    path_save=os.path.join(os.getcwd(),csvname)

    catalog_run_save.to_csv(path_save)  
    
    return

seed = 4

path_to_img='/share/storage2/connie/data/microchip/'
for p in sorted(os.listdir('/share/storage2/connie/data/microchip/')):
    if p[:4]=='proc':

        binned = p.replace('proc_bkgd', '')
        catalog_name = "Catalog"+binned+".csv"

        RunCatalog(path_to_img+p,catalog_name,GBparamsDic[re.findall( "proc_bkgd_([0-9x]+)_", p )[0]],seed)