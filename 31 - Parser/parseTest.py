import argparse

def funcion(x,m,b):
    return m*x+b

def parser():
    parser = argparse.ArgumentParser(prog='Temp Control', description='Programa de vizualizacion y control de temperatura para el banco de pruebas ICN-UNAM')

    parser.add_argument('--sp',type=int,nargs=3,help="Devuelte la temperatura actual del sistema. Ingresa la nueva temperatura para cambiarla")
    parser.add_argument()
    
    argObj = parser.parse_args()
    return argObj


def main(argObj):
    y=funcion(argObj[0],argObj[1],argObj[2])
    return y

if __name__ == "__main__":
    argObj = parser()
    exitcode = main(argObj.sp)
    exit(code=exitcode) 