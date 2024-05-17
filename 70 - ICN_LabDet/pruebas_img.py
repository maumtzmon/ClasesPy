from astropy.io import fits 
import matplotlib.pyplot as plt
import numpy as np 
import sys
# import os


def main(argObj):
    color='gray' ##Barra de colores de imágenes
    files = argObj
    hdul = fits.open(files[1])
    header = hdul[0].header
    overscan_mask = np.s_[56:638,:530] ## El área activa de la extensión

    ## Figura Plana##
    ## Con Overscan ##
    fig = [[0,0], [0,1], [1,0], [1,1] ]
    figt = 0
    orign = ['lower', 'lower', 'lower','lower']

    fig_all, axs_all = plt.subplots(2, 2, figsize=(20,10))
    max_min=7500

    for i in range(0, len(hdul)):
        data = hdul[i].data
        oScan=hdul[i].data[638:,530:]

        #Aplanar la imágen
        hist , bins_edges = np.histogram(oScan.flatten(), bins = 1000000)
        offset = bins_edges[np.argmax(hist)]
        data = data-offset

        axs_all[fig[i][0],fig[i][1]].imshow(data,cmap=color, vmin = -max_min, vmax = max_min, origin=orign[i])
        axs_all[fig[i][0],fig[i][1]].set_title('ext'+ str(figt+1))
        figt=figt+1

    plt.show()

    ## Sin Overscan ##
    fig_all, axs_all = plt.subplots(2, 2, figsize=(20,10))
    for i in range(0, len(hdul)):
        data = hdul[i].data[56:638,:530] #imagen de 700 x 700 aprox
        #data = hdul[i].data[:]
        oScan=hdul[i].data[638:,530:]
        #oScan = hdul[i].data[:]
        hist , bins_edges = np.histogram(oScan.flatten(), bins = 1000000)
        offset = bins_edges[np.argmax(hist)]
        data = data-offset

        axs_all[fig[i][0],fig[i][1]].imshow(data, cmap=color,vmin = -max_min, vmax = max_min, origin=orign[i])
        axs_all[fig[i][0],fig[i][1]].set_title('ext'+ str(figt+1))
        figt=figt+1
        
    plt.show()

    ### Extensiones juntas###
    img_Planas=[]
    for i in range(0, len(hdul)):
        #data = hdul[i].data[56:638,:530]
        data = hdul[i].data[:]
        #oScan=hdul[i].data[638:,530:]
        oScan = hdul[i].data[:]

        hist , bins_edges = np.histogram(oScan.flatten(), bins = 1000000)
        offset = bins_edges[np.argmax(hist)]
        data = data-offset
        img_Planas.append(data)

    ## Orienta bien las imágenes
    data1 = np.flip(img_Planas[0],0) #extensión 1
    data2 = img_Planas[1] #extensión 2
    data3 = np.flip(img_Planas[2],1) #extensión 3
    data4 = np.flip(img_Planas[3],1) #extensión 4
    data4 = np.flip(data4,0)

    mitad1 = np.concatenate((data2,data1), axis=0)
    mitad2 = np.concatenate((data3,data4), axis=0)

    todo = np.concatenate((mitad1,mitad2), axis=1)

    ## Figura Completa
    fig_all  = plt.imshow(todo, cmap=color ,vmin = -max_min, vmax = max_min, origin='lower')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
   argObj=sys.argv
   exitcode=main(argObj)
   exit(code=exitcode)