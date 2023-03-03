#####
#
#Program to mask events and calculate Spurious charge due the
#Temperature
#
#
#####

from functions_py import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from os import listdir as listdir
import pandas as pd


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


# imagenes=['/home/oem/datosFits/testMITLL/16FEB23/spurious/proc_skp_module24_MITLL01_externalVr-4_Vv2_T140__NSAMP324_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img169.fits',
# '/home/oem/datosFits/testMITLL/16FEB23/spurious/proc_skp_module24_MITLL01_externalVr-4_Vv2_T150__NSAMP324_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img147.fits',
# '/home/oem/datosFits/testMITLL/16FEB23/spurious/proc_skp_module24_MITLL01_externalVr-4_Vv2_T160__NSAMP324_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img126.fits',
# '/home/oem/datosFits/testMITLL/16FEB23/spurious/proc_skp_module24_MITLL01_externalVr-4_Vv2_T170__NSAMP324_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img103.fits']

path='/home/oem/datosFits/testMITLL/spuriousCharge/01Mar23/'
#path='/home/oem/datosFits/testMITLL/spuriousCharge/15FEB23/85/'
lista_de_archivos=listdir(path)
file_dict={}

for archivo in lista_de_archivos:
    file_dict.setdefault(archivo.split('_')[-1].split('.')[0],[archivo.split('_')[6].split('T')[-1]])


# slope=[]
# slope_masked=[]

for imagen in lista_de_archivos:
    #path=imagenes[i]

    hdu_list = fits.open(path+imagen)
    print(hdu_list.info())

    
    ##
    # plot imagen cruda
    # print('----------------')
    # # hdu_list[0].header
    # plt.figure(figsize=(20,10))
    # for i in range(4):
    #     plt.subplot(2,2,i+1)
    #     plt.imshow(hdu_list[i].data-np.median(hdu_list[i].data),cmap=mpl.cm.cividis, vmin=-800,vmax=800)
    #     plt.title('CHID '+str(i))
    #     plt.ylabel('Y_pix')
    #     plt.xlabel('X_pix')
    # plt.show()

    #########
    # Calibrated Image
    #########
    data_pre = precal(hdu_list,extensions=4)
    gain, gain_err, data= LocalCalib(data_pre,extensions=4)

    ###
    # Plot Imagen Calibrada
    # fig=plt.figure(figsize=(20,10))
    # for i in range(4):
    #     plt.subplot(2,2,i+1)
    #     plt.imshow(data[i],vmin=-1,vmax=1,cmap=mpl.cm.cividis)
    #     plt.title('CHID '+str(i))
    #     plt.ylabel('Y_pix')
    #     plt.xlabel('X_pix')
    #     print("CHID "+str(i)+' Gain:({:.2f} +- {:.2f})ADUs'.format(gain[i],gain_err[i]))
    # plt.suptitle('Calibrated Images')
    # plt.savefig('IMAGES_POSCALIB.png', bbox_inches='tight', dpi=100)
    # plt.show()

    ########
    # Exposure Matrix
    ########
    ExpoMatrix, Ttot, NRow, NCol, NSamp= totTime(path+imagen)

    # fig=plt.figure(figsize=(15,10))

    # plt.imshow(ExpoMatrix,cmap=mpl.cm.cividis,vmin=0,vmax=Ttot)#Ttot/3600)

    # plt.title('Exposure Time Matrix')
    # plt.ylabel('Y_Rows')
    # plt.xlabel('X_Cols')

    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
    # plt.ylabel('Seg')

    # plt.show()


    ########
    HEF, VEF = exposureFactor(path + imagen) #Exposure Factor in [s/pix]


    ########
    data_masked=data[0]

    data_Row, data_Col, Row_bins, Col_bins, Row_hist, Col_hist = hist_RowColumn(data_masked)

    # fig_all, axs_all = plt.subplots(2, 2, figsize=(20, 20))		# Create figures


    # axs_all[0][0].hist(data_Row, Row_bins, density=True)

    # axs_all[0][1].hist(data_Col, Col_bins, density=True)

    # axs_all[1][0].plot(data_Row)

    # axs_all[1][1].plot(data_Col)

    # axs_all[1][0].set_ylabel('e-/pix')
    # axs_all[0][1].set_xlabel('e-/pix')

    # plt.show()

    ########
    #
    # Fit to median Col or Row array
    #
    ########
    #NROW650_NCOL700
    #fig_all, axs_all = plt.subplots(1, 2, figsize=(20, 10))		# Create figures
    x_data=[]
    i=0

    for value in data_Row:  
                x_data.append(i)
                i+=1
    Xdata=np.array(x_data)*VEF[0]/86400

    #axs_all[0].plot(Xdata, data_Row) #Xdata[Pix]*VEF[S/Pix], data_Row[e-]*86400[s/day]/VEF[s/pix]
    popt, pcov = curve_fit(line, Xdata, data_Row)
    #axs_all[0].plot(Xdata, line(Xdata, popt[0], popt[1]), 'r-', label='fit: m=%f, b=%f' % tuple(popt)) #plot de xdata vs f(xdata)
    rowValues=popt
    # axs_all[0].plot(np.mean(data_Row), label="mean=%5.3f" % np.mean(data_Row) )
    # axs_all[0].set_ylabel('[e-/pix]')
    # axs_all[0].set_xlabel('Days')
    # axs_all[0].legend()
    # print('m=%f, b=%f' % tuple(popt))
    file_dict[imagen.split('_')[-1].split('.')[0]].append(popt[0])
    # print(slope[0])

    x_data=[]
    i=0

    for value in data_Col:  
                x_data.append(i)
                i+=1
    Xdata=np.array(x_data)*HEF[0]/86400

    #axs_all[1].plot(Xdata, data_Col)
    popt, pcov = curve_fit(line, Xdata[:550], data_Col[:550])
    #axs_all[1].plot(Xdata[:550], line(Xdata[:550], popt[0], popt[1]), 'r-', label='fit: m=%f, b=%f' % tuple(popt)) #plot de xdata vs f(xdata)
    colValues=popt
    # axs_all[1].plot(np.mean(data_Col[:550]), label="mean=%5.3f" % np.mean(data_Col) )
    # axs_all[1].set_xlabel('Days')
    # axs_all[1].legend()
    # plt.show()



    #########
    # Do the Mask 
    #########


    label=ndimage.label(data[0]>4,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]

    rps=sk.regionprops(label,intensity_image=data[0],cache=False, extra_properties=[sum_intensity])
    areas=[r.area for r in rps]
    energy=[r.sum_intensity for r in rps]
    centr=[r.weighted_centroid for r in rps]
    ecce=[r.eccentricity for r in rps]
    dic_props={"areas":areas,"energias":energy,'centroid':centr, "excentricidade":ecce}

    df=pd.DataFrame.from_dict(dic_props)
    df.index = np.arange(1, len(df)+1)
    # df.head(35) #show dataframe

    energias=dic_props["energias"]
    exce=dic_props["excentricidade"]
    hee_list=np.where((np.array(energias)>80) & (np.array(exce)!=1))[0].tolist() #index od the dictionary of events ---> TRESHOLD OF 80 ELECTRONS TO BE HE

    areas=dic_props["areas"]
    see_list=np.where((np.array(exce)==1) & (np.array(areas)>9))[0].tolist() 
    see_list

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

    mask_bleeding=ndimage.binary_dilation(HE_events>0,structure=[[0,0,0],[0,1,1],[0,0,0]],iterations=40)*1.0#-(HE_events>0)*1.0

    maskV_bleeding=ndimage.binary_dilation(HE_events>0,structure=[[0,0,0],[0,1,0],[0,1,0]],iterations=20)*1.0#-(HE_events>0)*1.0

    matriz2, dmask=no_sr(data[0]+((maskV_bleeding+mask_bleeding+HE_events))*-1e7,threshold=3,trem=20)

    pre_sre=ndimage.label((data[0]+((maskV_bleeding+mask_bleeding+HE_events))*-1e7)>1.5,structure=[[0,0,0],[1,1,1],[0,0,0]])[0]

    rps=sk.regionprops(pre_sre,cache=False)
    areas=[r.area for r in rps]

    SRE_events=np.zeros_like(pre_sre)
    for event in np.where(np.array(areas)>1)[0].tolist():
        [x,y]=np.where(pre_sre==event+1)
        for i in range(len(x)):
            SRE_events[x[i],y[i]]=1

    SRE_mask=np.zeros_like(SRE_events)
    for y in np.unique(np.where(SRE_events==1)[0]).tolist(): SRE_mask[y,:]=1

    rps=sk.regionprops(label,intensity_image=data[1],cache=False, extra_properties=[sum_intensity])
    areas=[r.area for r in rps]
    energy=[r.sum_intensity for r in rps]
    centr=[r.weighted_centroid for r in rps]
    ecce=[r.eccentricity for r in rps]
    dic_props={"areas":areas,"energias":energy,'centroid':centr, "excentricidade":ecce}

    flag0=[]
    for i in range(len(centr)):
        if int(centr[i][0]) not in np.unique(np.where(SRE_events==1)[0]).tolist():
            
            flag0.append(i)

    flag0_events=np.zeros_like(label)
    for event in flag0:
        [x,y]=np.where(label==event+1)
        for i in range(len(x)):
            flag0_events[x[i],y[i]]=1

    # fig=plt.figure(figsize=(25,15))
    # plt.subplot(121)
    # plt.title('Original Image')
    # plt.imshow(data[0],vmin=-1,vmax=1,cmap=mpl.cm.cividis)
    # plt.ylabel('Y_pix')
    # plt.xlabel('X_pix')
    # plt.subplot(122)
    # plt.imshow(-(maskV_bleeding+mask_bleeding-2*HE_events+SRE_mask),vmin=-1,vmax=1,cmap=mpl.cm.cividis)
    # plt.title('Mask SRE+Bleeding')
    # plt.tick_params('y', labelleft=False)
    # plt.subplots_adjust(wspace=0.01)
    # plt.xlabel('X_pix')
    # plt.show()

    #########
    #
    #########

    event_mask=maskV_bleeding+mask_bleeding
    event_halo_mask = ndimage.binary_dilation(
            event_mask>0,
            iterations = 10,
            structure = ndimage.generate_binary_structure(rank=2, connectivity=2) # == [[1,1,1],[1,1,1],[1,1,1]]
        )
    event_halo_SR_mask = np.array(SRE_mask,dtype=bool)|event_halo_mask

    # fig=plt.figure(figsize=(15,10))
    # plt.imshow(data[0]-1e7*event_halo_SR_mask,vmin=-1,vmax=1,cmap=mpl.cm.cividis)
    # plt.ylabel('Y_pix')
    # plt.xlabel('X_pix')
    # plt.title('SRE+Bleeding Mask+ Bleeding Halo')
    # # plt.savefig('SER.png', bbox_inches='tight', dpi=100)
    # plt.show()


    #########
    # Masking image
    #########

    #data_masked=data[0]-event_halo_SR_mask
    dataMasked=ma.masked_array(data[0], mask=(event_halo_SR_mask))

    # fig=plt.figure(figsize=(10,8))
    # plt.subplot(121)
    # #plt.imshow(data[0]*event_halo_SR_mask,vmin=-1,vmax=1,cmap=mpl.cm.cividis) #plt.imshow(data[0]-1e7*event_halo_SR_mask,vmin=-1,vmax=1,cmap=mpl.cm.cividis)
    # plt.imshow(data[0],vmin=-1,vmax=1)#[0:699,550:649]
    # plt.ylabel('Y_pix')
    # plt.xlabel('X_pix')
    # plt.title('SRE+Bleeding Mask+ Bleeding Halo')
    # plt.subplot(122)
    # plt.imshow(ma.masked_array(data[0], mask=(event_halo_SR_mask)),vmin=-1,vmax=1)
    # #plt.savefig('SER.png', bbox_inches='tight', dpi=100)
    # plt.show()

    data_Row_masked, data_Col_masked, Row_bins_masked, Col_bins_masked, Row_hist_masked, Col_hist_masked = hist_RowColumn(dataMasked)
    ##########
    #
    # Fitting curve to masked Col and Row Array
    #
    ##########


    # fig_all, axs_all = plt.subplots(2, 2, figsize=(10, 10))		# Create figures
  

    # axs_all[0][0].hist(data_Row_masked, Row_bins_masked, density=True)

    # axs_all[0][1].hist(data_Col_masked, Col_bins_masked, density=True)

    # axs_all[1][0].plot(data_Row_masked)

    # axs_all[1][1].plot(data_Col_masked)

    # axs_all[1][0].set_ylabel('e-/pix')
    # axs_all[0][1].set_xlabel('e-/pix')

    # plt.show()

    # fig_all, axs_all = plt.subplots(1, 2, figsize=(20, 7))		# Create figures
    # fig_all.suptitle('masked')
    x_data=[]
    i=0

    for value in data_Row_masked:  
                x_data.append(i)
                i+=1
    Xdata=np.array(x_data)*VEF[0]/86400

    # axs_all[0].plot(Xdata, data_Row_masked) #Xdata[Pix]*VEF[S/Pix], data_Row[e-]*86400[s/day]/VEF[s/pix]
    popt, pcov = curve_fit(line, Xdata, data_Row_masked)
    # axs_all[0].plot(Xdata, line(Xdata, popt[0], popt[1]), 'r-', label='fit: m=%f, b=%f' % tuple(popt)) #plot de xdata vs f(xdata)
    file_dict[imagen.split('_')[-1].split('.')[0]].append(popt[0])
    

    rowValues=popt
    # axs_all[0].plot(np.mean(data_Row_masked), label="mean=%5.3f" % np.mean(data_Row_masked) )
    # axs_all[0].set_ylabel('[e-/pix]')
    # axs_all[0].set_xlabel('Days')
    # axs_all[0].legend()

    x_data=[]
    i=0

    for value in data_Col_masked:  
                x_data.append(i)
                i+=1
    Xdata=np.array(x_data)*HEF[0]/86400

    # axs_all[1].plot(Xdata, data_Col_masked)
    popt, pcov = curve_fit(line, Xdata[:550], data_Col_masked[:550])
    # axs_all[1].plot(Xdata[:550], line(Xdata[:550], popt[0], popt[1]), 'r-', label='fit: m=%f, b=%f' % tuple(popt)) #plot de xdata vs f(xdata)
    colValues=popt
    # axs_all[1].plot(np.mean(data_Col_masked[:550]), label="mean=%5.3f" % np.mean(data_Col_masked) )
    # axs_all[1].set_xlabel('Days')
    # axs_all[1].legend()

    # plt.show()
print('\n\n')
    
pd.DataFrame.from_dict(file_dict,orient='index').to_csv('datos_mean_28Feb.csv',sep=',')




