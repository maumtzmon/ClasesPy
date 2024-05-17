import pickle   
from matplotlib import pyplot as plt

fileObj = open('/home/oem/Software/cursoInstrumentacion_2022/ClasesPy/data.obj', 'rb')
exampleObj = pickle.load(fileObj)
fileObj.close()
plt.show()