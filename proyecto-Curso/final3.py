from random import betavariate
from urllib import request
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots


html_data = requests.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html').text

soup = BeautifulSoup(html_data, 'html5lib')

gme_revenue = pd.DataFrame(columns=['Date', 'Revenue'])


for row in soup.find("tbody").find_all("tr"):
    col = row.find_all("td")
    date = col[0]
    revenue = col[1].getText().replace('$',"").replace(',',"")

    gme_revenue = pd.concat([gme_revenue ,pd.DataFrame([{"Date":date, "Revenue":revenue}])], ignore_index=True)


gme_revenue.tail()  


make_graph(tesla_data, tesla_revenue, 'Tesla').