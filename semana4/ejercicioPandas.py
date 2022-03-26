import pandas as pd


#Leer archivos csv con pandas

x = {'Name': ['Rose','John', 'Jane', 'Mary'],
     'ID': [1, 2, 3, 4], 
     'Department': ['Architect Group', 'Software Group', 'Design Team', 'Infrastructure'], 
      'Salary':[100000, 80000, 50000, 60000]}

#casting the dictionary to a DataFrame
df = pd.DataFrame(x)

#display the result df

print(df)

# Para selecccionar una columna en especifico
# si no le agrego doble [[]] me duvuelve solo los resultados, si  le agrego tambien me devulve el nombre de la columna

y = df['Name']

print(y)

# loc() --> se accede con los nombres de las filas/columnas e iloc() --> Se accede con los indices Funciones sirven para entrar una fila y columna en especifico
# fila / Columna
z = df.iloc[1, 0]

print(z)


# Encontrar elementos unicos de una columna
lista = {'numero': [1, 2, 3,2,1,3,2,4,5]}

df2 = pd.DataFrame(lista)
print("ahora")
print(df2.loc[0:3, 'numero'])

# encontrar valores segun condiciones

df3 = df2[df2['numero']>1]
