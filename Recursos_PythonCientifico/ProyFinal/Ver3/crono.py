#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:16:28 2018

@author: mauricio
"""

import time
from time import sleep

def crono(f):
    """
    Regresa el tiempo que toma en ejecutarse la funcion.
    """
    def tiempo(*args, **kargs):
        t1 = time.time()
        f(*args, **kargs)
        t2 = time.time()
        return 'Elapsed time: ' + str((t2 - t1)) + "\n"
    return tiempo

@crono
def miFuncion(valor):    
    sleep(2)
    return valor


    
miFuncion(1)
print(miFuncion(1))
