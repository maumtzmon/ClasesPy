#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:01:55 2018

@author: mauricio
"""

"""

Mi Convertidor

Aqui se van colocar todas las operaciones de conversion de las diferentes 
categorias

"""
"""
Conversion de Temperaturas
"""
#CelsiusToKelvin 
KTC = lambda k: k - 273.15
#CelsiusToFarenheit
CTK = lambda c: c + 273.15
#FarenheitToCelsius
FTC = lambda f: (f - 32)* 5 / 9
#CelsiusToFarenheit
CTF = lambda c: (c * 9 / 5) +32
#KelvinToFarenheit
KTF = lambda k: (k * 9 / 5) - 459.67
#FarenheitToKelvin
FTK = lambda f: (f + 459.67) * 5 / 9

#ConvierteTemperatura
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
"""
Conversion de Longitudes
"""
#metroTopie
mtp = lambda m: m*3.28
#metroToparsec
mtps =lambda m: m*1/(30857*10**16)
#pieTometro
ptm = lambda p: p*.3048
#pieToparsec
ptps = lambda p: p*.3048*1/(30857*(10**16))
#pieToparsec
pstm = lambda ps: ps*(30857*(10**16))
#parsecTopie
pstp = lambda ps: ps*.3048*(30857*(10**16))



def convierteLongitud(unidad, valor):
    print('{:>10.6f}'.format(valor),'{:>30}'.format(unidad))
    
    if unidad == 'Metro':
        return (valor,
                mtp(valor),
                mtps(valor))
    elif unidad == 'Pie':
        return (ptm(valor),
                valor,
                ptps(valor))
        
    elif unidad == 'Parsecs':
        return (pstm(valor),
                pstp(valor),
                valor)



"""
Conversion de Presiones
"""