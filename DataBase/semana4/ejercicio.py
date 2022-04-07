import pandas as pd
from pyparsing import col

from sqlalchemy import create_engine




## ASI SE CREA UNA TABLA A PARTIR DE UN SQL CON PANDAS

# read_csv Asume que la primera linea del archivo son los nombres de las columnas
data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\DataBase\semana4\datos.csv")

# poniendo header=None decimos que la primera fila no son los nombres de las columnas
#data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\DataBase\semana4\datos.csv", header=None)

# Con eso nos dice que tipo de dato se guardo cada columna del excel
print(data.dtypes)

# si lo pongo asi nomas solo me trae las columnas que som de tipo numerico
print(data.describe())

# de esta me tiene en cuenta todas las columnas incluyendo las que son string
print(data.describe(include="all"))

df = pd.DataFrame(data, columns=data.columns.values)

alchemyEngine = create_engine("mysql+pymysql://root:Pirataceleste1@localhost/data_science", pool_recycle=3600)

mySqlConnection = alchemyEngine.connect()


table = "NombreDeLaTabla"

try:
    df.to_sql(table, mySqlConnection, schema="data_science",if_exists="replace")
except:
    print("Como el orto...")


