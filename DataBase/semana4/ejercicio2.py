from xml.etree.ElementInclude import include
import pandas as pd
from pyparsing import col

from sqlalchemy import create_engine

import matplotlib.pyplot as plt

import seaborn as sns

import mysql.connector



mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Pirataceleste1"
)

# Al parecer puedo usar las dos tipos de coneciones a las bases de datos    

#alchemyEngine = create_engine("mysql+pymysql://root:Pirataceleste1@localhost/data_science", pool_recycle=3600)
#mydb = alchemyEngine.connect()



table = "mcdonalds"

query = "SELECT count(*) FROM data_science.mcdonalds "

df = pd.read_sql(query, mydb)




print(df)



#plot = sns.jointplot(x='per_capita_income_',y='hardship_index', data=df)