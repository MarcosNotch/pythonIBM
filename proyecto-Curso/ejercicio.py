import yfinance as yf
import pandas as pd
import matplotlib as plot

apple = yf.Ticker("AMD")

appleInfo = apple.info


appleHistoryPrice = apple.history(period='max')

volumen = appleHistoryPrice['Volume']

print(volumen)

print(volumen[0])


