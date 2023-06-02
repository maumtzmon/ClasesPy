from matplotlib import pyplot as plt
from os import fork, getpid
import time 
import sys


plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')

Val = fork()



if Val==0: #si es el proceso hijo
    print('el valod del PID del proceso hijo es ' + str(getpid()))
    plt.show()
    sys.exit() #termina aqui el proceso, de lo contrario hara lo que siga 
               #desṕues del IF-ELSE como el proceso padre
    
else:
    time.sleep(1) #este tiempo solo es para evitar que ambos usen la linea 
                  #de comando para escribir al mismo tiempo
    PID_padre=str(getpid())
    print('el valor del PID del proceso padre es: ' + PID_padre)
   
print('solo imprimir si eres el padre '+PID_padre)
    
