from functions_py import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import chisquare as chisquare
import pickle

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
path='/home/oem/datosFits/Fe55_test2025/proc_skp_m-009_microchip_T_170__Vv82_NSAMP_324_NROW_400_NCOL_700_EXPOSURE_0_NBINROW_1_NBINCOL_1_img_258.fits'

hdu_list = fits.open(path)
print(hdu_list.info())
print('----------------')
# hdu_list[0].header
plt.figure(figsize=(20,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(hdu_list[i].data)
    plt.title('CHID '+str(i))
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')
plt.show()

irow=302
icol=115
hdu_list[i].data[irow,icol]

NROW=int(hdu_list[0].header['NROW'])
NCOL=int(hdu_list[0].header['NCOL'])
xlow=231
xhi=278
ylow=134
yhi=197
plt.imshow(hdu_list[0].data[113:118,300:306])
plt.show()

irow=117     # Y
icol=302     # X
Qij=hdu_list[0].data[irow-1,icol-1]
print(Qij)



irow=117
icolOS_low=550
icolOS_hi=NCOL
np.median(hdu_list[0].data[irow-1,icolOS_low:icolOS_hi])

irow=117     # Y
icol=302     # X
Qij=hdu_list[0].data[irow-1,icol-1]
print(Qij)
Qos=Qij-np.median(hdu_list[0].data[irow-1,icolOS_low:icolOS_hi])
print(Qos)

#def medianSustract(hdulObj[ext]):
new_image=np.zeros(shape=(NROW,NCOL))
for row in range(0,NROW,1):
    for col in range(0,icolOS_low, 1):
        new_image[row][col]=hdu_list[0].data[row][col]-np.median(hdu_list[0].data[row,icolOS_low:icolOS_hi])
    for col in range(icolOS_low, NCOL, 1):
        new_image[row][col]=hdu_list[0].data[row][col]-np.median(hdu_list[0].data[row,icolOS_low:icolOS_hi])
#return newimage

fig, (raw, median_sus)=plt.subplots(ncols=2, figsize=(15,10))
cruda=raw.imshow(hdu_list[0].data, vmin=14600, vmax=15600)
ok=median_sus.imshow(new_image, vmin=0, vmax=1000)

fig.colorbar(cruda, ax=raw)
fig.colorbar(ok, ax=median_sus)

irow=117     # Y
icol=302     # X
Qij_medianSus=new_image[irow-1,icol-1]
print("carga cruda Qij="+str(Qij))
print("carga media sustraida="+str(Qij_medianSus))

plt.show()