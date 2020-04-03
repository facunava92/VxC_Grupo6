#!/usr/bin/python

#Crear una función adivina que permita adivinar un número secreto generado en forma aleatoria, según las siguientes consignas:
#   El número secreto debe estar entre 0 y 100, y debe ser generado dentro de la función.
#   La función adivina debe recibir un parámetro que indique la cantidad de intentos permitidos.

import random

#Entrada
intentos = int(input('Ingrese la cantidad de intentos permitidos: '))
numero = random.randint(0, 100)
i = 0
flag = False


#Procesos
while(i < intentos):
    guess = int(input('\tIngrese su adivinanza: '))
    if(guess == numero):
        flag = True
        break
    else:
        i += 1

#Salida
if(flag):
    print('\t\t Felicitaciones, número correcto!!')
else:
    print('\t\t Vuelva a intentarlo!!')
