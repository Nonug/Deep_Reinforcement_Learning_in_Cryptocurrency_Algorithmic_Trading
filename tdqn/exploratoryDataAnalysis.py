import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller

class ExploratoryDataAnalysis:
	def __init__(self, data, symbol='btc'):
		self.data = data
		self.timeSeries = data['Close']
		self.stock_to_flow = data['s2f']
		self.symbol = symbol.upper()


	# from tdqn
	def plotTimeSeries(self):
		# Generation of the plot
		pd.plotting.register_matplotlib_converters()
		plt.figure(figsize=(10, 4))
		plt.plot(self.timeSeries.index, self.timeSeries.values, color='blue')
		plt.xlabel("Time")
		plt.ylabel("Price")
		plt.title(self.symbol+" Price 2014-2021")
		plt.show()


	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/one-step-prediction.ipynb
	def augmentedDickeyFullerTest(self):
		print("Augmented Dickey-Fuller test (ADF):")
		results = adfuller(self.timeSeries)
		print('ADF Statistic: %f' % results[0])
		print('p-value: %f' % results[1])


	# bollingerBandStrategy
	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/Cryptocurrencies_EDA.ipynb
	def get_bollinger_bands(self, rm, rstd):
		"""Return upper and lower Bollinger Bands."""
		upper_band = rm + 2*rstd
		lower_band = rm - 2*rstd
		return upper_band, lower_band
	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/Cryptocurrencies_EDA.ipynb
	def bollingerBandStrategy(self):
		rm = self.timeSeries.rolling(window=20).mean()
		rstd = self.timeSeries.rolling(window=20).std()
		upper_band, lower_band = self.get_bollinger_bands(rm, rstd)

		plt.figure(figsize=(10,6))
		ax = self.timeSeries.plot(title = self.symbol+" Bollinger Band", lw=1)
		rm.plot(label='Rolling Mean', ax=ax, lw=1)
		upper_band.plot(label='Upper Band', ax=ax, lw=1)
		lower_band.plot(label='Lower Band', ax=ax, lw=1)
		plt.legend(loc='upper left')
		plt.title(self.symbol+" Bollinger Band")


	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/Cryptocurrencies_EDA.ipynb
	def plot_daily_returns(self):
		daily_returns = (self.timeSeries / self.timeSeries.shift(1)) - 1
		daily_returns.iloc[0,:] = 0
		ax = daily_returns.plot(title=self.symbol+" Daily returns", fontsize=12)
		ax.set_xlabel("Date")
		ax.set_ylabel("Price")


	def plot_scarcity(self):
		# scatter plot
		plt.figure(figsize=(10, 4))
		ax = plt.gca()
		ax.scatter(self.stock_to_flow.values ,self.timeSeries.values, s=1)
		plt.xlabel("stock-to-flow (scarcity)")
		plt.ylabel("price")
		plt.title(self.symbol+" Stock To Flow Against Price")


	def check_empty_duplicate(self):
		print("Number of empty elements: ")
		print(self.data[["Open","High","Low","Close","s2f"]].isna().sum())
		print("Number of duplicates: ")
		print(len(self.data)-len(self.data.drop_duplicates()))