import pandas as pd
import matplotlib.pyplot as plt

dataframe=pd.read_csv('/home/oem/datosFits/MicrochipTest_Marzo/datos/26MAY23/skp_m-009_microchip_T_170__Vv82_NSAMP_1_NROW_50_NCOL_650_EXPOSURE_0_NBINROW_1_NBINCOL_1_img_2.csv')

dataframe.columns=['col_0','col_1','col_2','col_3']

data_0=dataframe['col_0']
data_1=dataframe['col_1']
data_2=dataframe['col_2']
data_3=dataframe['col_3']
pSamp=[]
sSamp=[]


j=0 
sinit=140
for i in range(0, len(data_2)):
   
    if (-1*data_3[i]>0) and (data_2[i]==1):
        j+=1
        
        if j<150 and data_2[i]>0:
            sSamp.append(0)
            pSamp.append(0)
        elif j>150 and data_2[i]>0:
            sSamp.append(0)
            pSamp.append(1)
        else:
                sSamp.append(0)
                pSamp.append(0)
    
    
    elif (-1*data_3[i]>0) and (data_2[i]==2):
        j+=1
        
        if j<sinit and data_2[i]==2:
            sSamp.append(0)
            pSamp.append(0)
        elif j>30 and data_2[i]==2:
            sSamp.append(1)
            pSamp.append(0)
        else:
                sSamp.append(0)
                pSamp.append(0)

    else:
        sSamp.append(0)
        pSamp.append(0)
        j=0


fig, ax1=plt.subplots()

color = 'tab:red'

ax1.set_xlabel('time (s)')
ax1.set_ylabel('Video', color=color)
ax1.plot(-1*data_3[0:3000], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Video clk', color=color)  # we already handled the x-label with ax1
# ax2.plot(data_2[0:3000], color=color)
# ax2.tick_params(axis='y', labelcolor=color)


ax3 = ax1.twinx()

color = 'tab:green'
t_ini=7000
t_end=8000
#ax3.set_xlabel('time (s)')
ax3.set_ylabel('SSAMP   ______', color=color)
ax3.plot(sSamp[0:3000], color=color)
ax3.tick_params(axis='y', labelcolor=color)

ax4 = ax1.twinx()

color = 'tab:orange'
t_ini=7000
t_end=8000
#ax4.set_xlabel()
ax4.set_ylabel('_______   PSAMP', color=color)
ax4.plot(pSamp[0:3000], color=color)
ax4.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()