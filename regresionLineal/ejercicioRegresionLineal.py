import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\automobileEDA.csv")

# VARIABLES NUMERICAS
# Relacion lineal , engine-size y price es un posible predictor de precios

df = pd.DataFrame(data, columns=data.columns.values)

x = df[["highway-mpg"]]
y = df["price"]

lm = LinearRegression()

lm.fit(x, y)



yhat = lm.predict(np.arange(1, 201, 1).reshape(-1, 1))

print(yhat)

grafico = sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


print(lm.intercept_)

print(lm.coef_)


pr = PolynomialFeatures(degree=2, include_bias=False)

x_polly = pr.fit_transform([[1, 2]])

print(x_polly)

# Para usar PIPELINE
# Se crean una lista de tuplas
input = [('Scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2))
, ('mode', LinearRegression())]

pipe = Pipeline(input)

pipe.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y)

yhat = pipe.predict(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

print(yhat)