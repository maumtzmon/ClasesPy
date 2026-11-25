import os
from datosVideo_24MAY23 import sinitPlot
print(os.listdir(path='/home/oem/datosFits/MicrochipTest_Marzo/datos/28JUL23/csvFiles/'))

path='/home/oem/datosFits/MicrochipTest_Marzo/datos/28JUL23/csvFiles/'

for archivo in os.listdir(path):
    print(path+archivo)
    sinitPlot(csvFile=path+archivo, saveFig=True)