from math import sqrt as sqrt
from math import pow as pow
from math import pi as Pi
from math import exp as exp
from numpy import random as npRandom
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def genRandData():
    data=[]
    for i in range(0,100000,1):  # La altura del pico depende del numero de eventos, no de los parametros de la curva
        data.append(npRandom.normal(loc=0,scale=60))

    for i in range(0,10000,1):
        data.append(npRandom.normal(loc=250,scale=60))

    for i in range(0,1000,1):
        data.append(npRandom.normal(loc=500,scale=60))

    return data


def gauss(x,x0,s,A): # x=data, x0=mean, s=StdDev, A=amplitud
    return A*np.exp(-(x-x0)**2/(2*s**2))/np.sqrt(np.pi*2*s**2)

def gauss2(x,x0,s,A,B,gain): # x=data, x0=mean, s=StdDev, A=amplitud, B=amplitud 2, gain
    return A*np.exp(-(x-x0)**2/(2*s**2))/np.sqrt(np.pi*2*s**2)+B*np.exp(-(x-x0-gain)**2/(2*s**2))/np.sqrt(np.pi*2*s**2)

q=genRandData()
yVals, xVals, _=plt.hist(q, bins=500)#, range=(-200,500))
xVals=(xVals[1:]+xVals[:-1])/2 #bins= (bins_X[1:]+bins_X[:-1])/2
plt.yscale("log")  
plt.show() 

popt, pcov= curve_fit(gauss2,xVals,yVals, p0=(0,60,1300,130,250))

yVals, xVals, _=plt.hist(q, bins=500)
# plt.plot(xVals[75:330],gauss2(xVals[75:330],*popt))
plt.plot(xVals,gauss2(xVals,*popt))
plt.show()