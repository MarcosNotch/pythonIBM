from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib as mpl

data = pd.read_excel(r"C:\Users\Usuario\Desktop\CursoIBM\dataVisualization\Canada.xlsx" , skiprows=range(20), skipfooter=2, sheet_name='Canada by Citizenship')


print(data.shape)

data.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

data.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

# Aca preguntamos si todas las columnas son str
print(all(isinstance(column, str) for column in data.columns))

data['Total'] = data.sum(axis=1)

# Con esto cambio el indice por una columna
data.set_index('Country', inplace=True)

print(data.shape)

years = list(map(int, range(1980, 2014)))

df_island = data.loc["Iceland", years]

df_island.plot(kind="bar",)
plt.show()


# Ordeno Los datos
data.sort_values(['Total'], ascending=False, axis=0, inplace=True)


data5 = data.head()

data5 = data5[years].transpose()

# Paso el indice a int
data5.index = data5.index.map(int)

data5.plot(kind="area", alpha=1,stacked=False, figsize=(20, 10))
plt.show()