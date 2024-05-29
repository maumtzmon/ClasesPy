from functions_py import *

def DataRunADU(path,v=False):
    npy_name='DataADU'+path.replace('/share/storage2/connie/data/microchip/proc_bkgd', '')+'.npy'
    npy_run='RunADU'+path.replace('/share/storage2/connie/data/microchip/proc_bkgd', '')+'.npy'
    list_flat=[]
    for path in glob.glob(path+'/proc_skp_*.fits'):
        hdu_list=fits.open(path)
        data_pre = precal(hdu_list,extensions=4)
        data_flat = np.reshape(data_pre[:,:, 9:-(osc+5)], (4,np.prod(data_pre[:,:, 9:-(osc+5)].shape[-2:])))
        list_flat.append(data_flat)
    np.save(npy_name, np.array(list_flat))
    run_data=np.concatenate(list_flat,axis=1)
    np.save(npy_run,run_data)
    return 
    
for p in sorted(glob.glob('/share/storage2/connie/data/microchip/proc_bkgd_2x*')): DataRunADU(p,v=True)
