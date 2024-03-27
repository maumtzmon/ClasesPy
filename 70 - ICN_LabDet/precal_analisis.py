from functions_py import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import chisquare as chisquare
import pickle

#ICN
sys.path.insert(0, '/home/oem/Software/Serial_Register_Events_Detection')
#LapTop Mau
#sys.path.insert(0, '/home/mauricio/Software/Serial_Register_Events_Detection')

from ReconLib import *

plt.rcParams.update({
    "image.origin": "lower",
    "image.aspect": 1,
    #"text.usetex": True,
    "grid.alpha": .5,
    "axes.linewidth":2,
    "lines.linewidth" : 1,
    "font.size":    15.0,
    "xaxis.labellocation": 'right',  # alignment of the xaxis label: {left, right, center}
    "yaxis.labellocation": 'top',  # alignment of the yaxis label: {bottom, top, center}
    "xtick.top":           True ,  # draw ticks on the top side
    "xtick.major.size":    8    ,# major tick size in points
    "xtick.minor.size":    4      ,# minor tick size in points
    "xtick.direction":     'in',
    "xtick.minor.visible": True,
    "ytick.right":           True ,  # draw ticks on the top side
    "ytick.major.size":    8    ,# major tick size in points
    "ytick.minor.size":    4      ,# minor tick size in points
    "ytick.direction":     'in',
    "ytick.minor.visible": True,
    "ytick.major.width":   2   , # major tick width in points
    "ytick.minor.width":   1 ,
    "xtick.major.width":   2   , # major tick width in points
    "xtick.minor.width":   1 ,
    "legend.framealpha": 0 ,
    "legend.loc": 'best',
    

})


# ICN
# path='/home/oem/datosFits/spuriousCharge/Microchip/14AUG23/proc_skp_module24_MITLL01_externalVr-4_Vv2_T140__NSAMP324_NROW250_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img105.fits'
path = '/home/oem/datosFits/Data_2023/serialRegEvents/21NOV23/proc_skp_m-009_microchip_T_170__Vv82_NSAMP_324_NROW_400_NCOL_700_EXPOSURE_0_NBINROW_1_NBINCOL_1_img_68.fits'
# LapTop Mau
# path='/home/mauricio/datosFits/spuriousCharge/Microchip/14AUG23/proc_skp_module24_MITLL01_externalVr-4_Vv2_T140__NSAMP225_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img109.fits'
#path='/home/oem/datosFits/serialRegEvents/13OCT23/All/nsamp_324/proc_skp_m-009_microchip_T_170__Vv82_NSAMP_324_NROW_650_NCOL_700_EXPOSURE_0_NBINROW_1_NBINCOL_1_img_103.fits'

hdu_list = fits.open(path)
print(hdu_list.info())
print('----------------')
hdu_list[0].header


fig,axs=plt.subplots(ncols=2,nrows=2, sharex=True, sharey=True,figsize=(20,10))

raw_1=axs[0][0].imshow(hdu_list[0].data, vmin=np.median(hdu_list[0].data)-800,vmax=np.median(hdu_list[0].data)+800)
fig.colorbar(raw_1,ax=axs[0][0], location='bottom')
raw_2=axs[0][1].imshow(hdu_list[1].data, vmin=np.median(hdu_list[1].data)-800,vmax=np.median(hdu_list[1].data)+800)
fig.colorbar(raw_2,ax=axs[0][1], location='bottom')
raw_3=axs[1][0].imshow(hdu_list[2].data, vmin=np.median(hdu_list[2].data)-800,vmax=np.median(hdu_list[2].data)+800)
fig.colorbar(raw_3,ax=axs[1][0], location='bottom')
raw_4=axs[1][1].imshow(hdu_list[3].data, vmin=np.median(hdu_list[3].data)-800,vmax=np.median(hdu_list[3].data)+800)
fig.colorbar(raw_4,ax=axs[1][1], location='bottom')


plt.show()

raw_data=[]
for i in range(4):
    raw_data.append(np.copy(hdu_list[i].data))

data_copy=np.copy(hdu_list[0].data)
data_copy -= np.median( data_copy[os_median_mask], axis=1, keepdims=True ) 


Plot2Images_v2(hdu_list[0].data, data_copy, MinRange=np.median(hdu_list[0].data)-800, MaxRange=np.median(hdu_list[0].data)+800, )


data_precal=[]
for i in range(4):
    data_precal.append(precal(hdu_list, chid=i))




fig,axs=plt.subplots(ncols=4,nrows=2, sharex=True, sharey=True,figsize=(20,10))
raw_1=axs[0][0].imshow(raw_data[0], vmin=np.median(raw_data[0])-1000,vmax=np.median(raw_data[0])+1000)
fig.colorbar(raw_1,ax=axs[0][0], location='bottom')
raw_2=axs[0][1].imshow(raw_data[1], vmin=np.median(raw_data[1])-1000,vmax=np.median(raw_data[1])+1000)
fig.colorbar(raw_2,ax=axs[0][1], location='bottom')
raw_3=axs[0][2].imshow(raw_data[2], vmin=np.median(raw_data[2])-1000,vmax=np.median(raw_data[2])+1000)
fig.colorbar(raw_3,ax=axs[0][2], location='bottom')
raw_4=axs[0][3].imshow(raw_data[3], vmin=np.median(raw_data[3])-1000,vmax=np.median(raw_data[3])+1000)
fig.colorbar(raw_4,ax=axs[0][3], location='bottom')

data_1=axs[1][0].imshow(data_precal[0])
fig.colorbar(data_1,ax=axs[1][0], location='bottom')
data_2=axs[1][1].imshow(data_precal[1])
fig.colorbar(data_2,ax=axs[1][1], location='bottom')
data_3=axs[1][2].imshow(data_precal[2])
fig.colorbar(data_3,ax=axs[1][2], location='bottom')
data_4=axs[1][3].imshow(data_precal[3])
fig.colorbar(data_4,ax=axs[1][3], location='bottom')
plt.tight_layout()

plt.show()



fig,axs=plt.subplots(ncols=2,nrows=2,figsize=(18,13))

top_left=axs[0][0].imshow(raw_data[3])
fig.colorbar(top_left,ax=axs[0][0], location='bottom')
bottom_left=axs[1][0].imshow(data_precal[3])
fig.colorbar(bottom_left,ax=axs[1][0], location='bottom')

top_right=axs[0][1].hist(raw_data[3][100:300,550:].flatten(), bins=3000, range=(np.median(raw_data[3][100:300,550:])-1000,np.median(raw_data[3][100:300,550:])+1900))
bottom_right=axs[1][1].hist(data_precal[3][100:300,550:].flatten(), bins=3000, range=(-1000,1900))
plt.tight_layout()
plt.show()