
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# OBJETIVO 

data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\automobileEDA.csv")


df = pd.DataFrame(data, columns=data.columns.values)


# Una sola variable

x = df[["highway-mpg"]]
y = df[["price"]]

lm1 = LinearRegression()

lm1.fit(x, y)

print("Imprimiendo pendiente: ", lm1.coef_)

print("Imprimiendo ordenada: ", lm1.intercept_)


## PARA COMPARAR MODELOS DE DISTINTOS TIPOS

# EL MODELO CON R2 MAS ALTO ES MEJOR 
# EL MODELO CON MSE MAS BAJO ES MEJOR

# Sirve para saber que tan precioso es nuestro modelo valores cercanos a 1 y -1 son buenos
print("Imprimiendo R2 Regresion lineal Simple" , lm1.score(x, y))
print(df[["highway-mpg", "price"]].corr())

# Calculando MSE
# Sirve para saber que tan precioso es nuestro modelo valores cercanos a 1 y -1 son buenos
Yhat=lm1.predict(x)
mse = mean_squared_error(df['price'], Yhat)
print('Imprimiendo MSE para regresion lineal simple: ', mse)



person, p_value = stats.pearsonr(df["highway-mpg"], df["price"])

print("Person: ", person)
print("pvalue: ", p_value)



# Prediccion usando MULTIPLES VARIABLES

z = df[["normalized-losses", "highway-mpg"]]

lm2 = LinearRegression()

# lo entreno
lm2.fit(z, df["price"])


print(lm2.coef_)

print(lm2.intercept_)

array = ([2, 3], [3, 2])

# lo uso para predecir
print(lm2.predict(array))

# Calculando R2 para multiples Variables
print('The R-square is para regresion Lineal Multiple: ', lm2.score(z, df['price']))

# Calculando MSE para multiples variables
Y_predict_multifit = lm2.predict(z)
print('el MSE para regresion lineal multiple: ', \
      mean_squared_error(df['price'], Y_predict_multifit))


## VISUALIZANDO MODELOS ##

# para modelos de regresion lineal simple "regression plots" son los mejores

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

# Con este metodo podemos ver cual tiene mejor correlancion y vemos que highway con price esta mas cerca de -1 
# que peak con price entonces el otro es mas fuerte
print(df[["peak-rpm","highway-mpg","price"]].corr())


# Viendo un grafico residual podemos deducir lo siguiente
# Si los puntos del grafico estan esparcido aleatoriamente en el eje X entonces un buen modelo
# seria uno de regresion lineal
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()


# Para modelos de multiple linear Regression

Y_hat = lm2.predict(z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


## REGRESION POLINOMICA y Pipelines

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']

# Esta funcion Dato un cierto tipo de valores te crea una funcion que le pegue a esos valores
f = np.polyfit(x, y, 11)
# a en teoria es la version mas nueva de la misma funcion
a = np.polynomial.polynomial.Polynomial.fit(x, y, 11)
p = np.poly1d(f)

# p es la funcion

PlotPolly(p, x, y, 'highway-mpg')
PlotPolly(a, x, y, 'highway-mpg')


# Calculando R2 para regresion POLINOMICA (Se importa from sklearn.metrics import r2_score)
r_squared = r2_score(y, p(x))
print('The R-square value is para POLINOMICA: ', r_squared)

# Calculando MSE para regresion POLINOMICA
print("el MSE para polinomica es: ", mean_squared_error(df['price'], p(x)))

# Prediccion y toma de decisiones


new_input=np.arange(1, 100, 1).reshape(-1, 1)

x = df[["highway-mpg"]]
y = df[["price"]]

lm1 = LinearRegression()

lm1.fit(x, y)
 
yhat = lm1.predict(new_input)

print(type(yhat))

print(type(new_input))

print(yhat[0:5])

plt.plot(new_input, yhat)
plt.show()