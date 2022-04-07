import requests
import pandas as pd
from bs4 import BeautifulSoup


url = "https://www.ibm.com/"

r = requests.get(url)



header = r.headers



# WEBSCRAPING sacar automaticamente informacion de sitios web  usamos bs4

# Guardamos el html de una pagina a la que queremos sacarle informacion


# Ejemplo de como obtener todoo el html de una pagina 


page = requests.get("https://www.binance.com/es").text


soup = BeautifulSoup(page, "html.parser")

print(soup)