from termcolor import colored

def seleccionaCategoria(unidades_permitidas):
    """
    Imprime las categorías de unidades que se pueden usar y solicita 
    al usuario elegir entre dichas categorías.
    
    Args:
        unidades_permitidas: diccionario que contiene como claves las categorías.
    Return:
        categoria: es la categoría seleccionada por el usuario. 
    """
    print('\n' + 
          colored('Categoría de unidades:',
                  'blue',attrs=['bold','underline']))
    [print('{:>25}'.format(key.title())) for key in unidades_permitidas.keys()]
    category = input('Tu elección: ').upper()
    return category

def seleccionaUnidad(unidades):
    """
    Imprime las unidades que se pueden transformar y solicita al usuario
    elegir entre estas unidades
    
    Args:
        unidades: lista de las unidades que son válidas
    """
    print('\n' + colored('Unidades:','red',attrs=['bold','underline']))
    [print('{:>25}'.format(x)) for x in unidades]
    unidad_1 = input('Tu elección: ')
    return unidad_1

def introduceDato():
    """
    Solicita un valor a convertir y lo regresa.
    """
    return float(input('Valor a convertir:'))

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

##############
    
def convertKelvinToCelsius(k):
    return k - 273.15

def convertCelsiusToKelvin(c):
    return c + 273.15

def convertFahrenheitToCelsius(f):
    return (f - 32) * 5 / 9

def convertCelsiusToFahrenheit(c):
    return c * 9 / 5 + 32

def convertKelvinToFahrenheit(k):
    return(k * 9 / 5 - 459.67)

def convertFahrenheitToKelvin(f):
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
                convertKelvinToFahrenheit(valor),
                convertKelvinToCelsius(valor))
    elif unidad == 'Fahrenheit':
        return (convertFahrenheitToKelvin(valor),
                valor,
                convertFahrenheitToCelsius(valor))
    elif unidad == 'Celsius':
        return (convertCelsiusToKelvin(valor),
                convertCelsiusToFahrenheit(valor),
                valor)
        
############################
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

#####################3333
    
from IPython.display import clear_output

unidades_permitidas =  {'TEMPERATURA' : ['Kelvin',
                                        'Fahrenheit', 
                                        'Celsius'],
                       'LONGITUD' : ['Metro', 
                                     'Pie',
                                     'Parsecs']}

bitacora = open('conversiones.txt','w')

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
        
bitacora.close()