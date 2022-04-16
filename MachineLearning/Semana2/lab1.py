import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score 



data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\MachineLearning\Semana2\FuelConsumptionCo2.csv")

viz = data[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(data.CYLINDERS, data.CO2EMISSIONS, color='red')
plt.xlabel("Cylindier")
plt.ylabel("Co2Emissions")
plt.show()


x_train , x_test, y_train, y_test = train_test_split(data[['ENGINESIZE']], data[['CO2EMISSIONS']], test_size=0.2, random_state=0)

plt.scatter(x_train, y_train, color='red')
plt.xlabel("ENGINESIZE")
plt.ylabel("Co2Emissions")
plt.show()


lm = LinearRegression()

lm.fit(x_train, y_train)
          
print("Coheficientes: ", lm.coef_)

print("Interecept: ", lm.intercept_)

# De esta forma imprimo todos los datos de train y la funcion que me genero
plt.scatter(x_train.ENGINESIZE, y_train.CO2EMISSIONS,  color='blue')
plt.plot(x_train.ENGINESIZE, lm.coef_[0][0]*x_train.ENGINESIZE + lm.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt.close()

#+++EVALUACION DEL MODELO+++

y_predict = lm.predict(x_test )

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predict - y_test)))
# Distintas formas de evaluar el MSE
print("Residual sum of squares (MSE) MANUAL: %.2f" % np.mean((y_predict - y_test) ** 2))

print("Residual sum of squares (MSE) PRO: ", mean_squared_error(y_test, y_predict)) 

print("R2-score noob: %.2f" % r2_score(y_test , y_predict) )

print("r2 pro: ", lm.score(x_test, y_test))