from functions_py import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

data_set_name=['1x10','1x2','1x4','2x1','2x2']    
calibparam=['dataADU'+name+'.csv' for name in data_set_name]
npydata=['dataADU'+name+'.npy' for name in data_set_name]
csv1x10=['AnaParamsCSV/AnaParams_1x10_1.csv',
             'AnaParamsCSV/AnaParams_1x10_2.csv',
             'AnaParamsCSV/AnaParams_1x10_3.csv']
csv1x2=['AnaParamsCSV/AnaParams_1x2_1.csv',
            'AnaParamsCSV/AnaParams_1x2_2.csv',
            'AnaParamsCSV/AnaParams_1x2_3.csv']
csv1x4=['AnaParamsCSV/AnaParams_1x4_1.csv',
            'AnaParamsCSV/AnaParams_1x4_2.csv']
csv2x1=['AnaParamsCSV/AnaParams_2x1_1.csv',
            'AnaParamsCSV/AnaParams_2x1_2.csv']
csv2x2=['AnaParamsCSV/AnaParams_2x2_1.csv',
            'AnaParamsCSV/AnaParams_2x2_2.csv']

data_set=[csv1x10,csv1x2,csv1x4,csv2x1,csv2x2]
exptimes=[]
for i in range(5):
    runs=[]
    for csv in data_set[i]:
        df = pd.read_csv(csv,usecols=['CHID','IMGID','ReadoutTime'])
        runs.append(pd.read_csv(csv,usecols=['CHID','IMGID','ReadoutTime']))
    csv_cg = pd.concat(runs)
    exptimes.append(csv_cg[csv_cg['CHID']==0]['ReadoutTime'].sum()/2)

datas=[]
for file in npydata:
    dt=np.load(file,mmap_mode='r')
    datas.append(dt)

    
import ast
GBparams=[]
for k in range(5):
    globalpar=pd.read_csv(calibparam[k],usecols=['Params(a,b)'],converters={'Params(a,b)': ast.literal_eval})
    GBparams.append(list(globalpar['Params(a,b)']))
    
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
ax=[(0,0),(0,1),(1,0),(1,1)]
for k in range(5):
    for j in range(4):
        data = datas[k][j]
        data = (data-GBparams[k][j][1])/GBparams[k][j][0]
        global_hist, xb = np.histogram(data.flatten(), bins = np.arange(-0.5, 3.5, .01) ) 
        x = (xb[1:]+xb[:-1])/2
        axs[ax[j]].plot(x,global_hist/exptimes[k],linewidth=2,label=data_set_name[k])
        axs[ax[j]].set_yscale('log')
        axs[ax[j]].set_ylabel('#pix/day')
        axs[ax[j]].set_xlabel('e-')
        axs[ax[j]].set_title('CHID '+str(j))
axs[ax[j]].legend()

plt.savefig('/home/zilvespedro/work/MICROCHIP/MANA/PEAKS.png', bbox_inches='tight', dpi=100)