#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:17:15 2018

@author: mauricio
"""

from termcolor import colored

def seleccionaCategoria(unidades_permitidas):
    """
    Imprime las categorías de unidades que se pueden usar y solicita al usuario
    elegir entre dichas categorías.
    
    Args:
        unidades_permitidas: diccionario que contiene como claves las categorías.
    Return:
        categoria: es la categoría seleccionada por el usuario. 
    """
    while True:
        try:
            print('\n' + colored('Categoría de unidades:','blue',attrs=['bold','underline']))
            [print('{:>25}'.format(key.title())) for key in unidades_permitidas.keys()]
            categoria = input('Tu elección: ').upper()
            unidades_permitidas[categoria]
            break
        except KeyError:
            print(colored('La opción ' + categoria + ' es inválida. Intenta de nuevo ...','red',attrs=['bold']))
            
    return categoria

def seleccionaUnidad(unidades):
    """
    Imprime las unidades que se pueden transformar y solicita al usuario
    elegir entre estas unidades
    
    Args:
        unidades: lista de las unidades que son válidas
    """
    while True:
        try:
            print('\n' + colored('Unidades:','green',attrs=['bold','underline']))
            [print('{:>25}'.format(x)) for x in unidades]
            unidad_1 = input('Tu elección: ')
            unidades.index(unidad_1)
            break
        except ValueError:
            print(colored('La opción ' + unidad_1 + ' es inválida. Intenta de nuevo ...','red',attrs=['bold']))

    return unidad_1

def introduceDato():
    """
    Solicita un valor a convertir y lo regresa.
    """
    while True:
        try:
            return float(input('Valor a convertir:'))
        except ValueError:
            print('Opcion inválida, ingresa un número')

def despliegaResultado(unidades, valores):
    """
    Imprime el resultado final de la conversión.
    
    Args:
        unidades: lista de las unidades que son válidas
        valores: valores transformados en las diferentes unidades
    """
    num = len(unidades)
    lines = '-' * 20
    print('.',end='')
    [print(lines,end='.') for x in range(num)]
    print('\n|',end='')
    [print('{:^20}'.format(x),end='|') for x in unidades]
    print('\n+',end='')
    [print(lines,end='+') for x in range(num)]
    print('\n|',end='')
    [print('{:20.10e}'.format(x),end='|') for x in valores]    
    print('\n+',end='')
    [print(lines,end='+') for x in range(num)]

def otraConversion():
    """
    Pregunta si se desea realizar otra conversión y regresa la respuesta.
    """
    return input('Deseas hacer otra conversión? (Si / No): ').lower().startswith('s')
###########
    
from IPython.display import clear_output

def iniciaConvertidor(bitacora):
    """"""
    while True:
        clear_output()
    
        categoria = seleccionaCategoria(unidades_permitidas)
        unidades = []
        [unidades.append(x) for x in unidades_permitidas[categoria]]

        u1 = seleccionaUnidad(unidades)
        v1 = introduceDato()
        
        if categoria == 'TEMPERATURA':
            valores = [x for x in convierteTemperatura(u1,v1)]
        elif categoria == 'LONGITUD':
            valores = [x for x in convierteLongitud(u1,v1)]
        
        despliegaResultado(unidades, valores)

        bitacora.write(categoria + '\n')
        [bitacora.write(x + '\t' + str(y) + '\n') for x,y in zip(unidades_permitidas[categoria],valores)]
    
        if not otraConversion():
            clear_output()
            print('Happy Finish!')
            break
#################################3

from misConversiones import *

unidades_permitidas = {'TEMPERATURA' : ['Kelvin',
                                        'Fahrenheit', 
                                        'Celsius'],
                       'LONGITUD' : ['Metro', 
                                     'Pie',
                                     'Parsecs']}

bitacora = open('conversiones','w')

iniciaConvertidor(bitacora)
        
bitacora.close()