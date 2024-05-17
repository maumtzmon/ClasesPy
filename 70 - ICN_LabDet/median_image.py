from os import listdir  as listdir
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits

data_list=[]
flag=True
path='/home/oem/datosFits/serialRegEvents/400x700/setNuevo/'
n=0
files = listdir(path)
validFiles=[]
medianImage=[]

for file in files:
    if file.endswith(".fits"):
        validFiles.append(file)
        print(file+'....OK')
    else:
        print(file+'Not Fits')

for image_name in validFiles:
    dataFromImage=fits.open(path+image_name) #imagen en curso
    if flag:
        (x,y)=np.shape(dataFromImage[0])
        flag=False
        emptyArray=np.empty((x,y,len(validFiles)),dtype=float)
        for ext in range(0,4):
            data_list.append(emptyArray)

    for ext in range(0,4):
        #data_list.append(dataFromImage[ext].data)
        data_list[ext][:,:,n]=dataFromImage[ext].data
    n+=1
    
print("holaMundo")

for ext in range(0,4):
    medianImage.append(np.median(data_list[ext],axis=2))

plt.imshow(medianImage[0])
plt.show()
print("holaMundo")
        #guardar sus datos en una lista
