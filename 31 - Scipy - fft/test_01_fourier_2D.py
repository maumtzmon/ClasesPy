import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifftn

time = np.arange(1000)/1000
signalSinus= np.sin(50*2.0*np.pi*time)

fftSin=np.fft.fft(signalSinus)

n=signalSinus.size
timestep=.001

# fftFreqSin=np.fft.fftfreq(n,d=timestep)

# fig, (ax1,ax2)=plt.subplots(2,1)
# ax1.plot(signalSinus)
# ax2.plot(fftFreqSin,np.abs(fftSin))
# plt.show()

Sin_50= 0.75*np.sin(50*2.0*np.pi*time)
Sin_100=0.5*np.sin(100*2.0*np.pi*time)
Sin_250=0.25*np.sin(250*2.0*np.pi*time)
Sin_300=0.125*np.sin(300*2.0*np.pi*time)
sum=Sin_50+Sin_100+Sin_250+Sin_300
# plt.plot(sum)
# plt.show()


n=sum.size                       #magnitud del arreglo
timestep=.001                            #intervalo en segundos de cada muestra

fft_Sum=np.fft.fft(sum)           #Amplitud de la espiga
fftFreqSin=np.fft.fftfreq(n,d=timestep)  #frecuencia de la espiga

fig, (ax1,ax2)=plt.subplots(2,1)
fig.canvas.manager.set_window_title('machupikachu experiment <3')
ax1.plot(sum)
ax2.plot(fftFreqSin,(2*np.abs(fft_Sum))/n)
plt.show()


# >>> timestep = 0.1
# >>> freq = np.fft.fftfreq(n, d=timestep)
# >>> freq