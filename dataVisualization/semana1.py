from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Usando el artis layer (que en teoria es mas facil de codear)

fig = Figure()
canvas = FigureCanvas(fig)

x = np.random.randn(10000)

ax = fig.add_subplot(111)

ax.hist(x, 100)
ax.set_title("messi")

#fig.savefig("asda.png")


# Usando el pyplot que hace lo mismo pero mas liviano los procesos

#y = np.random.randn(10000)
#plt.hist(x, 100)
#plt.title("messi2")
#plt.show()
#plt.close()


data = pd.read_excel(r"C:\Users\Usuario\Desktop\CursoIBM\dataVisualization\Canada.xlsx" , skiprows=range(20), skipfooter=2, sheet_name='Canada by Citizenship')


data.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

data.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

data['Total'] = data.sum(axis=1)

# Con esto cambio el indice por una columna
data.set_index('Country', inplace=True)

# Con esto obtengo un arrelgo de tales fechas
years = list(map(int, range(1980, 2014)))

print(data.loc["Argentina", years])

#con esto obtengo la fila argentina con la lista de columnas
argentina = data.loc["Argentina", years]

argentina.plot(kind='line') 
plt.title("argentisads")
plt.xlabel("argentinos")
plt.ylabel("years")