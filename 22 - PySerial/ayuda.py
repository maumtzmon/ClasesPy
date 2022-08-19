file=open('ConfigFile_M335','r')
diccionario={}#diccionario={'clave':'valor'}
#Port: /dev/ttyUSB0
#lista=['port','/dev/ttyUSB0']
for linea in file:
    if ':' in linea:
        #lista=linea.split(':')  #nombre_de_lista[elemento]
        #diccionario[lista[0]]=lista[1]
        diccionario[linea.split(':')[0]]=linea.split(':')[1]

print(diccionario)