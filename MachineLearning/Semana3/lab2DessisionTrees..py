import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\MachineLearning\Semana3\drug200.csv")

data.shape



# No puedo usar el dummies debido a q ue el dummies te hace varais columnas segun cada valor categorico

#sexo = pd.get_dummies(data[['Sex']])

#data["Sex_F"] = sexo["Sex_F"]






# Sacamos la variable objetivo
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]. values


# Convertimos variables categoricas a variables numericas pero usando la misma columna sin crear nuevas como lo haria dummies
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

# Definimos la variable Y
y = data["Drug"]


# Separamos los datos en datos de prueba y datos test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)


# Creamos el modelo
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
 # it shows the default parameters

# Entrenamos el modelo
drugTree.fit(x_train,y_train)

# Hacemos la prediccion
prediction = drugTree.predict(x_test)

print (prediction[0:5])
print (y_test[0:5])

# Aca podemos ver como usa el algoritmo de jaccard para saber que tan preciso es el modelo 1 mejor
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test,prediction))

# Con esto podemos ver el arbol de decisiones que nos armo
tree.plot_tree(drugTree)
plt.show()