
# si usamos with no hace falta cerrar el archivo
# la funcion open puede recibir 3 tipos de segundos argumentos w (write), r (read), a (append)
# fuera del with no se puede leer nada mas 
with open("semana4/pruebaTexto.txt", "r") as archivo1:
    print(archivo1.name)
    print(archivo1.readline())
    print(archivo1.readline())
    print(archivo1.readline())
    print(archivo1.mode)


with open("semana4/pruebaTexto2.txt", "w") as archivo2:
    archivo2.write("Aguante messi carajo\n")


with open("semana4/pruebaTexto2.txt", "r") as archivo2:
    print(archivo2.readline())


# a de append no te crea un nuevo archivo como lo hace w , si no te que lo concatena al que ya tenias


with open("semana4/pruebaTexto2.txt", "a") as archivo2:
    archivo2.write("Aguante Belgrano viejaaa\n")
