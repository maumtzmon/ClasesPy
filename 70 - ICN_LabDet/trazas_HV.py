import pandas as pd
from matplotlib import pyplot as plt
import os


def twoTraces(df_1, df_2):
    fig, (ch_1, ch_2)=plt.subplots(2, sharex=True,sharey=True)
    ch_1.plot(df_1)
    ch_2.plot(df_2,color='C2')
    #plt.show()
    return fig

filesinDir=os.listdir('/home/oem/Software/SSocial/Daniel/FototuboPruebas/Waveforms/')
emcoTraces={}
fuenteTraces={}

for file in filesinDir:
    if file.endswith('csv'):
        if file.startswith('EMCO'):
            id=file.split('_')[1]
            emcoTraces[id]=file
            
        elif file.startswith('isegHV'):
            id=file.split('_')[1]
            fuenteTraces[id]=file



for key in sorted(emcoTraces):
    df_emco  =pd.read_csv("/home/oem/Software/SSocial/Daniel/FototuboPruebas/Waveforms/"+emcoTraces[key])
    df_fuente=pd.read_csv("/home/oem/Software/SSocial/Daniel/FototuboPruebas/Waveforms/"+fuenteTraces[key])
    keysFuente=df_fuente.keys()
    keysEmco=df_emco.keys()
    twoTraces(df_fuente[keysFuente[-1]],df_emco[keysEmco[-1]])
    plt.show()