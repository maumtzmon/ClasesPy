import serial as ser
from time import sleep as sleep


port = ser.Serial('/dev/ttyACM0',baudrate=9600, timeout=1)
print("start")
print(port.readline().decode())
sleep(2)
if port.write('abrir'.encode()) == 5:
    print("abierto")

sleep(5)

if port.write('cerrar'.encode()) == 6:
    print("Cerrado")
    sleep(2)
print(port.readline().decode())


port.close()

