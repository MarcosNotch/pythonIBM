from statistics import LinearRegression
from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score


# Leemos los datos
data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\MachineLearning\Semana2\FuelConsumptionCo2.csv")


plt.scatter(data[["ENGINESIZE"]], data[["CO2EMISSIONS"]], color="blue")
plt.xlabel("Engine size")
plt.ylabel("co2 emissions")
plt.show()

# Separamos los datos en datos de prueba y datos de test
x_train, x_test, y_train, y_test = train_test_split(data[["ENGINESIZE"]], data[["CO2EMISSIONS"]], test_size=0.2, random_state=1)

# Aca convertimos nuestra variable en polinomica
poly = PolynomialFeatures(degree=2)
x_train_train = poly.fit_transform(x_train)

# Luego podemos continuear como si fuere una regresion linear
lm = linear_model.LinearRegression()

lm.fit(x_train_train, y_train)

# Mostramos los coheficientes y la pendiente

print("Coheficiente: ", lm.coef_)
print("pendiente: ", lm.intercept_)


# Dibujamos la funcion
plt.scatter(x_train, y_train,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = lm.intercept_[0]+ lm.coef_[0][1]*XX+ lm.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

# Evaluamos que el modelo este piola



test_x_poly = poly.transform(x_test)
test_y_ = lm.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test,test_y_ ) )