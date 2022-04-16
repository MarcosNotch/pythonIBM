from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from tqdm import tqdm

data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\regresionLineal\EvaluacionDeModelos\module_5_auto.csv")

# VARIABLES NUMERICAS
# Relacion lineal , engine-size y price es un posible predictor de precios

df = pd.DataFrame(data, columns=data.columns.values)


# primero solo agarramos los datos numericos

dfNum = df=df._get_numeric_data()


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction,color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    

    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)

   # print("xtrain: " , xtrain , " type", type(xtrain) )
   # print("ytrain: " , y_train , " type", type(y_train) )

    plt.plot(xtrain , y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

    plt.show()
    plt.close()



def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)

# -- Primero empezamos separando nuestros datos en datos de entrenamiento y de prueba --



# Guardamos la columna precio
y_data = dfNum["price"]

# guardamos todo menos la columna 'price'
x_data = dfNum.drop("price", axis=1)

print("***************Separacion de datos en train y test***************")

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lre = LinearRegression()

lre.fit(x_train[["horsepower"]], y_train)

r2_test = lre.score(x_test[['horsepower']], y_test)

r2_train = lre.score(x_train[['horsepower']], y_train)

print("r2_test: ", r2_test, " t2_train: ", r2_train)

# aca podemos ver que el r2_test es mucho menor al r2_train por lo tanto
# el modelo es una cagada y que el r2 debe estar cerca de 1 para que sea bueno


# -- validacion cruzada -- 

print("***************VALIDACION CRUZADA***************")

scores = cross_val_score(lre, x_data[["horsepower"]], y_data, cv=4)

print("The mean of the folds are", scores.mean(), "and the standard deviation is" , scores.std())


print("**************OVERFITTING - UNDERFITTING***************")

# se va a demostrar usando multi linear regresion y polinomica

mlr = LinearRegression()

# entrenamos el modelo
mlr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# hacemos predicciones con los datos de entrenamiento

prediction_train = mlr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

prediction_train[0:5]

# hacemos predicciones con los datos de test

prediction_test = mlr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

prediction_test[0:5]

# con este grafico podemos que predecir con los datos de prueba lo hizo bastante bien
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, prediction_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


# Aca podemos ver que con nuevos datos distintos a los de entrenamiento funciona medio como el orto
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,prediction_test,"Actual Values (Test)","Predicted Values (Test)",Title)

# por lo tanto un modelo multi linear no es el apropiado


# probamos ahora con un MODELO POLINOMICO

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)


pModel = PolynomialFeatures(degree=3)

x_train_pModel = pModel.fit_transform(x_train[["horsepower"]])
x_test_pModel = pModel.fit_transform(x_test[["horsepower"]])



poly = LinearRegression()

poly.fit(x_train_pModel, y_train)

predict = poly.predict(x_test_pModel)



print("Predicted values:", predict[0:4])
print("True values:", y_test[0:4].values)

PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pModel)


# Calculamos el r2 del modelo con datos de entrenamiento

print("R2 de la funcion polinomica con datos train", poly.score(x_train_pModel, y_train))


# Calculamos el r2 del modelo con datos de entrenamiento

print("R2 de la funcion polinomica con datos test", poly.score(x_test_pModel, y_test))

# se puede ver que el r2 con los datos de test anda como el ojete

# con esto de aca vemos cual seria el mejor grado de polinomio posible

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    poly.fit(x_train_pr, y_train)
    
    Rsqu_test.append(poly.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')    
plt.show()
plt.close()


# Ejemplo POLINOMICO  con MULTIPLES VARIABLES      

pr1 = PolynomialFeatures(degree=2)

x_train_pr1 = pr1.fit_transform(x_train[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])

x_test_pr1 = pr1.fit_transform(x_test[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])

poly1 = LinearRegression()

poly1.fit(x_train_pr1, y_train)

print("score: ", poly1.score(x_test_pr1, y_test))

prediction = poly1.predict(x_test_pr1)

DistributionPlot(y_test,prediction,"Actual Values (Test)","Predicted Values (Test)",Title)

# Como podemos ver en la grafica en el precio de 100000 y entre 30000 y 400000 anda medio como
# el ocote 

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    poly1.fit(x_train_pr, y_train)
    
    Rsqu_test.append(poly1.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')  
plt.show()
plt.close()


# REGRESION RIDGE con ridge podemos corregir errores de los polinomicos

print("********RIDGE REGRESSION*********")

pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

RigeModel=Ridge(alpha=1)

RigeModel.fit(x_train_pr, y_train)


yhat = RigeModel.predict(x_test_pr)

print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()