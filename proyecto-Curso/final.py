from numpy import empty
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
   # print(stock_data.Date)
   # print(revenue_data.Date)
    
    stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()


gme = yf.Ticker("GME")

gme_data = gme.info

gme_data = gme.history(period='max')

gme_data.reset_index(inplace=True)


html_data = requests.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html').text

soup = BeautifulSoup(html_data, 'html5lib')

gme_revenue = pd.DataFrame(columns=['Date', 'Revenue'])


div = soup.find_all("table", {"class": "historical_data_table table"})



for row in div:
    tabla = row
    match =  tabla.find(text='GameStop Quarterly Revenue')
    if (match is not None):
        tablaFinal = tabla


for row in tablaFinal.find_all("tr"):
    col = row.find_all("td")
    if(len(col) != 0):
        date = col[0]
        revenue = col[1].getText().replace('$',"").replace(',',"")

        gme_revenue = pd.concat([gme_revenue ,pd.DataFrame([{"Date":date.getText(), "Revenue":revenue}])], ignore_index=True)


make_graph(gme_data, gme_revenue, 'GAMESTOP')


