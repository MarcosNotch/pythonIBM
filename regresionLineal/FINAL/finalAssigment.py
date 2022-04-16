import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression


data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\regresionLineal\FINAL\kc_house_data_NaN.csv")

# VARIABLES NUMERICAS
# Relacion lineal , engine-size y price es un posible predictor de precios

df = pd.DataFrame(data, columns=data.columns.values)


width = 12
height = 10
plt.figure(figsize=(width, height))
ax1 = sns.boxplot(x=df["waterfront"], y=df["price"])
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()




width = 12
height = 10
plt.figure(figsize=(width, height))
ax1 = sns.regplot(x=df["sqft_above"], y=df["price"])
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()