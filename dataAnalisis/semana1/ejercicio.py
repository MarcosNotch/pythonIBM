import pandas as pd
from pyparsing import col

from sqlalchemy import create_engine




## ASI SE CREA UNA TABLA A PARTIR DE UN SQL CON PANDAS

# read_csv Asume que la primera linea del archivo son los nombres de las columnas
data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\DataBase\peers\ChicagoCensusData.csv")


# de esta me tiene en cuenta todas las columnas incluyendo las que son string
print(data.describe(include="all"))

df = pd.DataFrame(data, columns=data.columns.values)


pd.get_dummies(df["COMMUNITY_AREA_NAME"])



