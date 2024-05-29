from functions_py import *
import matplotlib.pyplot as plt
import skimage.measure as sk
import skimage as ski


plt.rcParams.update({
    "image.origin": "lower",
    "image.aspect": 1,
    #"text.usetex": True,
    "grid.alpha": .5,
    "axes.linewidth":2,
    "lines.linewidth" : 1,
    "font.size":    8.0,
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

def sum_intensity(region, intensities):
    return np.sum(intensities[region])

plotList=[0, # precal plot, codigo original devuelve ganancia mayor. Ajustar p0 en LocalCalib
          0, # imagenes Calibradas
          0, # establecer un umbral en e- para enmascarar eventos 
          1,
          0,
          0,
          0,
          0,
          0]

path='/home/oem/Software/cursoInstrumentacion_2022/ClasesPy/71-CopyPedro/proc_skp_m-009_microchip_T_170__Vv82_NSAMP_324_NROW_400_NCOL_700_EXPOSURE_0_NBINROW_1_NBINCOL_1_img_100.fits'
hdu_list = fits.open(path)

data_pre = precal(hdu_list,extensions=4)
gain, gain_err, data, hist, x, popt= LocalCalib(data_pre,extensions=4)

if plotList[0]==1:
    plt.plot(x,hist)
    plt.plot(x,gaussian2(x,*popt), label=popt[2])
    plt.legend()
    plt.show()

if plotList[1]==1:
    fig=plt.figure(figsize=(20,10))

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(data[i],vmin=-1,vmax=1)
        plt.title('CHID '+str(i))
        plt.ylabel('Y_pix')
        plt.xlabel('X_pix')
        print("CHID "+str(i)+' Gain:({:.2f} +- {:.2f})ADUs'.format(gain[i],gain_err[i]))
    plt.suptitle('Calibrated Images')
    #plt.savefig('IMAGES_POSCALIB.png', bbox_inches='tight', dpi=100)
        
    plt.show()

sigma, sigma_err = LocalSigma(data,extensions=4)
for i in range(4):
    print('CHID '+str(i),r' Noise: ({:.3f} +- {:.3f}) e-'.format(sigma[i],sigma_err[i]))

events_th=4 # e-

if plotList[2]==1: 
    fig=plt.figure(figsize=(20,20))
    for i in range(4):
        plt.subplot(4,2,2*(i+1)-1)
        plt.imshow(data[i],vmin=-1,vmax=1)
        plt.title('CHID '+str(i))
        plt.ylabel('Y_pix')
        plt.xlabel('X_pix')
        plt.subplot(4,2,2*(i+1))
        plt.imshow(data[i]>events_th,vmin=-1,vmax=1)
        plt.title('CHID '+str(i)+' > '+str(events_th)+'e-')
        plt.ylabel('Y_pix')
        plt.xlabel('X_pix')
    plt.show()


label=ndimage.label(data[1]>4,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]


rps=sk.regionprops(label,intensity_image=data[1],cache=False, extra_properties=[sum_intensity])
areas=[r.area for r in rps]
energy=[r.sum_intensity for r in rps]
centr=[r.weighted_centroid for r in rps]
ecce=[r.eccentricity for r in rps]
dic_props={"areas":areas,"energias":energy,'centroid':centr, "excentricidade":ecce}

hee_th=80

energias=dic_props["energias"]
exce=dic_props["excentricidade"]
hee_list=np.where((np.array(energias)>hee_th) & (np.array(exce)!=1))[0].tolist() #index od the dictionary of events ---> TRESHOLD OF 80 ELECTRONS TO BE HE

areas=dic_props["areas"]
see_list=np.where((np.array(exce)==1) & (np.array(areas)>9))[0].tolist() 


HE_events=np.zeros_like(label)
for event in hee_list:
    [x,y]=np.where(label==event+1)
    for i in range(len(x)):
        HE_events[x[i],y[i]]=1

SRE_events=np.zeros_like(label)
for event in see_list:
    [x,y]=np.where(label==event+1)
    for i in range(len(x)):
        SRE_events[x[i],y[i]]=1

if plotList[3]==1:
    fig=plt.figure(figsize=(25,15))
    plt.subplot(2,2,1)
    plt.imshow(data[1],vmin=-1,vmax=1)  #imagen original
    plt.title('Original Image')
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')
    plt.savefig('IMAGES_RECOHE.png', bbox_inches='tight', dpi=100)

    plt.subplot(2,2,2)
    plt.title('All Events Reconstructed')
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')
    plt.imshow(label!= 0) #todos los eventos en el catalogo

    plt.subplot(2,2,3)
    plt.imshow(HE_events)
    plt.title('Events: Summed charge > 80 e-, eccentricity<1')
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')

    plt.subplot(2,2,4)
    plt.imshow(SRE_events)
    plt.title('Serial Register Events')
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')
    # plt.colorbar()

    plt.show()

   
