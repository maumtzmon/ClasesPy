//
// este es un ejemplo que almacena los argumentos, ya despues tu sabras para que
// pero no se como usarlo con ROOT
//
// para compilarlo usa esto en la linea de comandos
// $g++ -o f fichero.cpp
//
// y para correr lo solo es
// ./f argumento

#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
	for (int i=0; i < argc; i++)
	{
		cout << "argv[" << i << "] = ";
		cout << argv[i] << endl;
	}
	return 0;	
}
