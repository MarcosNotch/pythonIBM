import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\MachineLearning\Semana3\teleCust1000t.csv")


x = data[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)


y = data['custcat'].values

# Con esto normalizamos los datos que basicamente es una buena pratica para esos tipos de algoritmos
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

# Dividimos nuestros datos en datos de prueba y test 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Creamos y entrenamos al modelo
k =4
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
neigh 

# Usamos el modelo para predecir
yhat = neigh.predict(x_test)

# Comprobamos con el algoritmo de jaccard que tan bueno es nuestro modelo
print("YTRAIN")
print(y_train.shape)
print("ytest")
print(yhat.shape)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# Con esta funcion podemos ver en un plano mas general cual es el mejor K
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc