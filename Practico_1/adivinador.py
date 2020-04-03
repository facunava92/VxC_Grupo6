#!/usr/bin/python

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

print('\t\t\tPráctico 1 Visión por Computadora\n')
print('\t#Programa para adivinar un número aleatorio dentro del rango 0-100')
intentos = int(input('\tIngrese la cantidad de intentos permitidos: '))
flag, i = adivinar(intentos)

if(flag):
    print('\n')
    print('\tFelicitaciones, ha encontrado número correcto en el intento {}!!'.format(i))
else:
    print('\n')
    print('\n\tHa alcanzado la cantidad máxima de intentos, vuelva a intentarlo!!')
