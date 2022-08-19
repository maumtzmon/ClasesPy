#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:01:54 2018

@author: mauricio
"""

"""
Proyecto final, curso de Python Cientifico, presenta, Mauricio Martinez Montero

Convertidor de unidades.
Forma de Uso
Cada categoria tendrá tres diferentes tipos de unidades.
Por ejemplo: Temperatura: Celsius, Farehrenheit, Kelvin
Forma de uso:

El usuario selecciona la categoría.
El usuario selecciona la unidad inicial a convertir.
El usuario ingresa el valor a convertir.
El programa despliega el valor equivalente en las tres unidades en una tabla.
El programa pregunta si se desea hacer otra conversión.
El programa muestra el tiempo que usaste el convertidor en [s]
El programa escribe una bitácora de todas las conversiones realizadas.
Cuando el usuario introduce valores inválidos el programa los detecta y le pide 
al usuario corregir.


"""



from misConversiones import *
from definiciones import *



unidades_permitidas = {'TEMPERATURA' : ['KELVIN',
                                        'FARENHEIT', 
                                        'CELSIUS'],
                       'LONGITUD' : ['METRO', 
                                     'PIE',
                                     'PARSECS']}

bitacora = open('conversiones.txt','w')


print(iniciaConvertidor(bitacora, unidades_permitidas))


bitacora.close()