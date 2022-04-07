import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\automobileEDA.csv")

# VARIABLES NUMERICAS
# Relacion lineal , engine-size y price es un posible predictor de precios

df = pd.DataFrame(data, columns=data.columns.values)



print(df[["engine-size", "price"]].corr())


sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

# VARIABLES CATEGORICAS
# una buena forma de visualizarlas es usando boxplots   

sns.boxplot(x="body-style", y="price", data=df)


# AGRUPANDO

# agrupando por una sola variable

df["drive-wheels"].unique()

df_group_one = df[['drive-wheels','body-style','price']]

df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()

print(df_group_one)


# Agrupando por varias variables
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)
grouped_pivot


# Agrupando por bodyStyle

agrupacionBodyStyle = df[["body-style", "price"]]
agrupado  = agrupacionBodyStyle.groupby(["body-style"], as_index=False).mean()
agrupado


# VEMOS LA CORRELACION de pearson

df.corr()

# VEMOS LA CORRELACION de pearson
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  