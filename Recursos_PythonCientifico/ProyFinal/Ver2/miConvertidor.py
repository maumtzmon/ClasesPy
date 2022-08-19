#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:23:56 2018

@author: mauricio
"""

def KTC(k):
    return k - 273.15

def CTK(c):
    return c + 273.15

def FTC(f):
    return (f - 32) * 5 / 9

def CTF(c):
    return c * 9 / 5 + 32

def KTF(k):
    return(k * 9 / 5 - 459.67)

def FTK(f):
    return (f + 459.67) * 5 / 9

def convierteTemperatura(unidad, valor):
    """
    Realiza las conversiones de temperaturas correspondientes.
    
    Args:
        unidad: La unidad original.
        valor: El valor que se desea convertir.
    """
    if unidad == 'Kelvin':
        return (valor,
                KTF(valor),
                KTC(valor))
    elif unidad == 'Fahrenheit':
        return (FTK(valor),
                valor,
                FTC(valor))
    elif unidad == 'Celsius':
        return (CTK(valor),
                CTF(valor),
                valor)
#########################

def convertMetroToPie(m):
    return m*3.28

def convertMetroToParSec(m):
    return m*1/(30857*10**16)

def convertPieToMetro(p):
    return p*.3048

def convertPieToParSec(p):
    return p*.3048*1/(30857*(10**16))

def convertParSecToMetro(ps):
    return ps*(30857*(10**16))

def convertParSecToPie(ps):
    return ps*.3048*(30857*(10**16))



def convierteLongitud(unidad, valor):
    print('{:>10.6f}'.format(valor),'{:>30}'.format(unidad))
    
    if unidad == 'Metro':
        return (valor,
                convertMetroToPie(valor),
                convertMetroToParSec(valor))
    elif unidad == 'Pie':
        return (convertPieToMetro(valor),
                valor,
                convertPieToParSec(valor))
        
    elif unidad == 'Parsecs':
        return (convertParSecToMetro(valor),
                convertParSecToPie(valor),
                valor)
