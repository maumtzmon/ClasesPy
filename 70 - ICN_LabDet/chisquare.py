#! /usr/bin/python3

import numpy as np
from scipy.optimize import curve_fit

###########################################a                                    
def f(t,N0,tau):
    """The model function"""
    return N0*np.exp(-t/tau)

###########################################a                                    
def main():

    # Model parameters
    T = 500
    dt = 10
    N0 = 1000
    tau = 100
    stdev = 20

    # Create the artificial dataset
    nobs = int(T/dt + 1.5)
    t = dt*np.arange(nobs)
    N = f(t,N0,tau)
    Nfluct = stdev*np.random.normal(size=nobs)
    N = N + Nfluct
    sig = np.zeros(nobs) + stdev

    # Fit the curve
    start = (1100, 90)
    popt, pcov = curve_fit(f,t,N,sigma = sig,p0 = start,absolute_sigma=True)
    print(popt)
    print(pcov)

    # Compute chi square
    Nexp = f(t, *popt)
    r = N - Nexp
    chisq = np.sum((r/stdev)**2)
    df = nobs - 2
    print("chisq =",chisq,"df =",df)

    # Plot the data with error bars along with the fit result
    import matplotlib.pyplot as plt
    plt.errorbar(t, N, yerr=sig, fmt = 'o', label='"data"')
    plt.plot(t, Nexp, label='fit')
    plt.legend()
    plt.show()

###########################################a                                    
main()