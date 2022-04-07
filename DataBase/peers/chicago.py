import pandas as pd
from pyparsing import col

from sqlalchemy import create_engine

import mysql.connector



# Al parecer puedo usar las dos tipos de coneciones a las bases de datos    




alchemyEngine = create_engine("mysql+pymysql://root:Pirataceleste1@localhost/data_science", pool_recycle=3600)


mydb = alchemyEngine.connect()



table = "mcdonalds"

query = "select * from data_science.census_data where PER_CAPITA_INCOME < 11000"

df = pd.read_sql(query, mydb)




print(df)
