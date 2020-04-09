#!/usr/bin/python
#Crear una función adivina que permita adivinar un número secreto generado en forma aleatoria, según las siguientes consignas:
#   El número secreto debe estar entre 0 y 100, y debe ser generado dentro de la función.
#   La función adivina debe recibir un parámetro que indique la cantidad de intentos permitidos.

from fn_adivina import adivinar
import random

print('\t\t\tPráctico 1 Visión por Computadora\n')
print('\t#Programa para adivinar un número aleatorio dentro del rango 0-100')
intentos = int(input('\tIngrese la cantidad de intentos permitidos: '))
flag, i = adivinar(intentos)

if(flag):
    print('\n')
    print('\tFelicitaciones, ha encontrado número correcto en el intento {}!!'.format(i))
else:
    print('\n')
    print('\tHa alcanzado la cantidad máxima de intentos, vuelva a intentarlo!!')
