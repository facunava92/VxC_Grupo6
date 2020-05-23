#!/usr/bin/python

#Crear una función adivina que permita adivinar un número secreto generado en forma aleatoria, según las siguientes consignas:
#   El número secreto debe estar entre 0 y 100, y debe ser generado dentro de la función.
#   La función adivina debe recibir un parámetro que indique la cantidad de intentos permitidos.

import random

def adivinar(intentos):

    numero = random.randint(0, 100)
    flag = False
    i=0

    print('\t\t Hint: ', numero)
    while(i < intentos):
        guess = int(input('\tIngrese su adivinanza: '))
        if(guess == numero):
            flag = True
            break
        i += 1
    return flag, i+1

__version__ = '0.1'
