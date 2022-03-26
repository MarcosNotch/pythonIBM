import numpy as np
from pandas import array

# Sumar matrices lineales SIN NUMPY
u = [1, 0]
v = [0 , 1]

z = []

for u, v in zip(u,v):
    z.append(u+v)

print(z)

# sumar matricos CON NUMPY

u = np.array([1, 0])

v = np.array([0, 1])

z = u + v

print(z)

# Multiplicacion de matrices con NUMPY por un escalar

y = np.array([1, 2])

z = y*2

print(z)

# Multiplicacion entre matrices

a = np.array([1, 2])
b = np.array([3, 4])

z = a*b

print(z)

# producto dot (para saber que tan parecidos son dos matrices)

a = np.array([1, 2])
b = np.array([3, 4])

z = np.dot(a ,b)

print(z)


# Dos dimensiones

g = np.array([[1, 2, 3], [3, 2, 1,], [4, 5, 6]])

print(g.ndim)


X=np.array([[1,0],[0,1]])
Y=np.array([[2,1],[1,2]]) 
Z=np.dot(X,Y)

print(Z)