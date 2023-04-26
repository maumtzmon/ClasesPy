import argparse
import sys


def parser():
    
    parser = argparse.ArgumentParser(prog='Parser Examples', description='Programa para evaluar ejemplos de opciones y sub opciones en linea de comandos')
    parser.add_argument('path',type=str,nargs=1,help="captura el nombre de la imagen")

    parser.add_argument("-C", "--const",action="store_true")
    parser.add_argument("-O", "--offset",action="store_true")
    parser.add_argument("-N", "--noise",action="store_true")
    parser.add_argument("-G", "--gain",action="store_true")
    parser.add_argument("-S", "--ser",action="store_true")
    
    if len(sys.argv)==1:
        parser.parse_args(['-h'])

    argObj=parser.parse_args()

    return argObj


def main(argObj):
    input("pon un breakpoint en esta linea y revisa el objeto argObj")

if __name__ == "__main__":
    argObj = parser()
    exitcode = main(argObj)
    exit(code=exitcode) 