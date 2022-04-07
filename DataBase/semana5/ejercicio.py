import pandas as pd
from pyparsing import col

from sqlalchemy import create_engine




## ASI SE CREA UNA TABLA A PARTIR DE UN SQL CON PANDAS

data = pd.read_csv(r"C:\Users\Usuario\Desktop\CursoIBM\DataBase\semana5\ChicagoPublicSchools.csv")


df = pd.DataFrame(data, columns=data.columns.values)

alchemyEngine = create_engine("mysql+pymysql://root:Pirataceleste1@localhost/data_science", pool_recycle=3600)

mySqlConnection = alchemyEngine.connect()

table = "ChicagoPublicSchools"

try:
    df.to_sql(table, mySqlConnection, schema="data_science",if_exists="replace")
except:
    print("Como el orto...")


