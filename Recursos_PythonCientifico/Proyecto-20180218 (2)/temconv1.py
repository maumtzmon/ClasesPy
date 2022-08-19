
units_allowed = {'TEMPERATURA' : ['Kelvin','Fahrenheit', 'Celsius'],
                'LONGITUD' : ['Metro', 'Pie','Parsecs']}

def convert_Kelvin_to_Celsius(k):
    return k - 273.15


# In[195]:

def convert_Celsius_to_Kelvin(c):
    return c + 273.15


# In[196]:

def convert_Fahrenheit_to_Celsius(f):
    return (f - 32) * 5 / 9


# In[197]:

def convert_Celsius_to_Fahrenheit(c):
    return c * 9 / 5 + 32


# In[198]:

def convert_Kelvin_to_Fahrenheit(k):
    return(k * 9 / 5 - 459.67)    


# In[199]:

def convert_Fahrenheit_to_Kelvin(f):
    return (f + 459.67) * 5 / 9


# In[200]:

def temperature_conversion(unit, value):
    if unit == 'Kelvin':
        return (value,
                convert_Kelvin_to_Fahrenheit(value),
                convert_Kelvin_to_Celsius(value))
    elif unit == 'Fahrenheit':
        return (convert_Fahrenheit_to_Kelvin(value),
                value,
                convert_Fahrenheit_to_Celsius(value))
    elif unit == 'Celsius':
        return (convert_Celsius_to_Kelvin(value),
                convert_Celsius_to_Fahrenheit(value),
                value)


