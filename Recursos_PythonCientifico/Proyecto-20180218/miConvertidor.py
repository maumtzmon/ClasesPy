
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

def convierteLongitud(unidades, valores):
    """
    Realiza las conversiones de longitudes correspondientes.
    
    Args:
        unidad: La unidad original.
        valor: El valor que se desea convertir.
    """
    print('{:>10.6f}'.format(valores),'{:>30}'.format(unidades))
    return (1,2,3)