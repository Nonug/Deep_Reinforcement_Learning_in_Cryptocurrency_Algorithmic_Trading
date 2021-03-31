import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller

class ExploratoryDataAnalysis:
	def __init__(self, timeSeries, stock_to_flow, symbol='btc'):
		self.timeSeries = timeSeries
		self.symbol = symbol
		self.stock_to_flow = stock_to_flow


	# tdqn
	def plotTimeSeries(self):
		# Generation of the plot
		pd.plotting.register_matplotlib_converters()
		plt.figure(figsize=(10, 4))
		plt.plot(self.timeSeries.index, self.timeSeries.values, color='blue')
		plt.xlabel("Time")
		plt.ylabel("Price")
		plt.show()


	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/one-step-prediction.ipynb
	def augmentedDickeyFullerTest(self):
		# Augmented Dickey-Fuller test
		print("Augmented Dickey-Fuller test (ADF):")
		results = adfuller(self.timeSeries)
		print('ADF Statistic: %f' % results[0])
		print('p-value: %f' % results[1])


	# bollingerBandStrategy1
	# https://randerson112358.medium.com/algorithmic-trading-using-bollinger-bands-python-e5081cbd7b4a
	# Create a function to get the buy and sell signals
	def get_signal(self, data):
		buy_signal = [] #buy list
		sell_signal = [] #sell list
		for i in range(len(data['Close'])):
			if data['Close'][i] > data['Upper'][i]: #Then you should sell 
				buy_signal.append(np.nan)
				sell_signal.append(data['Close'][i])
			elif data['Close'][i] < data['Lower'][i]: #Then you should buy
				sell_signal.append(np.nan)
				buy_signal.append(data['Close'][i])
			else:
				buy_signal.append(np.nan)
				sell_signal.append(np.nan)
		return (buy_signal, sell_signal)
	# https://randerson112358.medium.com/algorithmic-trading-using-bollinger-bands-python-e5081cbd7b4a
	def bollingerBandStrategy1(self):
		plt.style.use('fivethirtyeight')
		df = self.timeSeries.to_frame()
		df.columns = ['Close']
		#Get the time period (20 days)
		period = 20
		# Calculate the 20 Day Simple Moving Average, Std Deviation, Upper Band and Lower Band
		# Calculating the Simple Moving Average
		df['SMA'] = df['Close'].rolling(window=period).mean()
		# Get the standard deviation
		df['STD'] = df['Close'].rolling(window=period).std() 
		# Calculate the Upper Bollinger Band
		df['Upper'] = df['SMA'] + (df['STD'] * 2)
		# Calculate the Lower Bollinger Band
		df['Lower'] = df['SMA'] - (df['STD'] * 2)
		# Create a list of columns to keep
		column_list = ['Close', 'SMA', 'Upper', 'Lower']

		#Create a new data frame
		new_df = df[period-1:]
		#Create new columns for the buy and sell signals
		new_df['Buy'] =  self.get_signal(new_df)[0]
		new_df['Sell'] =  self.get_signal(new_df)[1]

		#Plot all of the data
		#Get the figure and the figure size
		fig = plt.figure(figsize=(12.2,6.4)) #width = 12.2 inches and height = 6.4 inches
		#Add the subplot
		ax = fig.add_subplot(1,1,1) #Number of rows, cols, & index
		# Get the index values of the DataFrame
		x_axis = new_df.index
		# Plot and shade the area between the upper band and the lower band Grey
		ax.fill_between(x_axis, new_df['Upper'], new_df['Lower'], color='grey')
		# Plot the Closing Price and Moving Average
		ax.plot(x_axis, new_df['Close'], color='gold', lw=3, label = 'Close Price',alpha = 0.5)
		ax.plot(x_axis, new_df['SMA'], color='blue', lw=3, label = 'Moving Average',alpha = 0.5)
		ax.scatter(x_axis, new_df['Buy'] , color='green', lw=3, label = 'Buy',marker = '^', alpha = 1)
		ax.scatter(x_axis, new_df['Sell'] , color='red', lw=3, label = 'Sell',marker = 'v', alpha = 1)
		# Set the Title & Show the Image
		ax.set_title('Bollinger Band')
		ax.set_xlabel('Date')
		ax.set_ylabel('USD Price ($)')
		plt.xticks(rotation = 45)
		ax.legend()
		plt.show();


	# bollingerBandStrategy2
	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/Cryptocurrencies_EDA.ipynb
	def get_bollinger_bands(self, rm, rstd):
		"""Return upper and lower Bollinger Bands."""
		upper_band = rm + 2*rstd
		lower_band = rm - 2*rstd
		return upper_band, lower_band
	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/Cryptocurrencies_EDA.ipynb
	def bollingerBandStrategy2(self):
		rm = self.timeSeries.rolling(window=20).mean()
		rstd = self.timeSeries.rolling(window=20).std()
		upper_band, lower_band = self.get_bollinger_bands(rm, rstd)

		plt.figure(figsize=(10,6))
		ax = self.timeSeries.plot(title = self.symbol+" Bollinger Band", lw=1)
		rm.plot(label='Rolling Mean', ax=ax, lw=1)
		upper_band.plot(label='Upper Band', ax=ax, lw=1)
		lower_band.plot(label='Lower Band', ax=ax, lw=1)
		plt.legend(loc='upper left')


	# https://github.com/guangzhixie/cryptocurrency-time-series/blob/master/Cryptocurrencies_EDA.ipynb
	def plot_daily_returns(self):
		daily_returns = (self.timeSeries / self.timeSeries.shift(1)) - 1
		daily_returns.iloc[0,:] = 0
		ax = daily_returns.plot(title="Daily returns", fontsize=12)
		ax.set_xlabel("Date")
		ax.set_ylabel("Price")


	def plot_scarcity(self):
		plt.figure(figsize=(10, 4))
		# plt.plot(self.stock_to_flow.values, self.timeSeries.values, color='blue')
		plt.loglog(self.stock_to_flow.values, self.timeSeries.values)
		plt.xlabel("stock-to-flow (scarcity)")
		plt.ylabel("market value")