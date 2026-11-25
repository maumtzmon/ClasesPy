import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation, writers

#https://matplotlib.org/stable/gallery/lines_bars_and_markers/hat_graph.html#sphx-glr-gallery-lines-bars-and-markers-hat-graph-py
#https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html#matplotlib.axes.Axes.annotate

dir='/home/oem/datosFits/Data_2023/MicrochipTest_Marzo/datos/vFiles_2/'
listDir=os.listdir(dir)

def outputStageTiming(vFile='/home/oem/datosFits/Data_2023/MicrochipTest_Marzo/datos/vFiles/voltage_skp_lta_v2_microchip.sh',saveFig=False):
    voltageDict={}
    swingDict={}
    with open(vFile, 'r') as VoltageFile:
        varlist=[]
        for line in VoltageFile:
            #print(line, end='\r')
            if '='in line:
                varlist.append(float(line.split('=')[1]))
            if len(varlist)==2:
                voltageDict[line.split('=')[0][0]]=varlist
                varlist=[]
    vr=-7
    voltageDict['r']=[vr,vr-0.1] #voltaje del nodo de lectura, esta en la seccion del Bias como vr (aumento .01 solo para el plot)
    print(str(voltageDict))

    cell_text = []
    fig, ax = plt.subplots()
    

    for key in voltageDict:
        x=0
        if key.startswith('v') or key.startswith('t') or key.startswith('h') or key.startswith('s') or key.startswith('o') or key.startswith('d') or key.startswith('r'):
            
            #   'key' : [high=0, low=1]
            #high states
            x+=1
            if voltageDict[key][0]>0:   
                ax.annotate(voltageDict[key][0],(key, float(voltageDict[key][0]+0.05)))
                ax.bar(key,voltageDict[key][0],color='black',label=key)
                if voltageDict[key][1]>0:
                    ax.bar(key,voltageDict[key][1], color='white',label=key)
                else:
                    ax.bar(key,voltageDict[key][1], color='black',label=key)          
            elif voltageDict[key][1]<0:
                ax.annotate(voltageDict[key][0],(key, float(voltageDict[key][0]+0.05)))
                ax.bar(key,voltageDict[key][1],color='black',label=key)
                if voltageDict[key][0]<0:
                    ax.bar(key,voltageDict[key][0], color='white',label=key)
                else:
                    ax.bar(key,voltageDict[key][1], color='black',label=key)
            #Lowstates
            # if voltageDict[key][1]>0:
            #     ax.bar(key,voltageDict[key][0],color='black',label=key)
            #     ax.bar(key,voltageDict[key][1], color='white',label=key)
                
            # else:
            #     ax.bar(key,voltageDict[key][1], color='black',label=key)
            #     ax.bar(key,voltageDict[key][0],color='green',label=key)
            if key != 'r':
                ax.annotate(voltageDict[key][1],(key, float(voltageDict[key][1]-.5))) 
            
    if saveFig==True:
        fig.savefig(vFile+'.jpeg',dpi=150)
    
   
# outputStageTiming()

# input()

fig = plt.figure(figsize = (7,5))
def animation_fun(i):
    print(i)
    #input()
    image=dir+listDir[i]
    outputStageTiming(image, saveFig=False)
        

animation=FuncAnimation(fig,animation_fun,frames=10,interval=1000)

plt.show()

