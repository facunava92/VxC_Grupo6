#!/usr/bin/python

#Crear una función adivina que permita adivinar un número secreto generado en forma aleatoria, según las siguientes consignas:
#   El número secreto debe estar entre 0 y 100, y debe ser generado dentro de la función.
#   La función adivina debe recibir un parámetro que indique la cantidad de intentos permitidos.
import random

def adivinar(intentos):   #recibe la cantidad de intentos

    numero = random.randint(0, 100) # numero aleatorio del 1 al 100  y lo guarda en la variabl " numero"
    flag = False   # asigna la variable "flag" en FALSE
    i=0               #asigna la variable i en cero    

    print('\t\t Hint: ', numero)
    while(i < intentos):   # cuando i sea menor a intentos   
        guess = int(input('\tIngrese su adivinanza: '))  # asigno un  numero adivino el numero 
        if(guess == numero):  # adivinansa es gual al numero aletaroi 
            flag = True 
            break
        i += 1
    return flag, i+1   #le devuelvo los intentos y la varaible flag

__version__ = '0.1'

