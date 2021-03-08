# Deep Reinforcement Learning in Cryptocurrency Algorithmic Trading
I will upload my code to this github repo, The things inside it is still uncommented, I will explain what I've done later may be on 9/3/2021

## folders and files
The dataset is scraped using scripts in scraping_datas
The dataset scraped is stored in datas
The scripts for the algorithms is/ will be stored in scripts
Please ignore the test directory, files in it is for me to test the scripts

## things I've done
I’ve followed https://github.com/guptarohit/cryptoCMD to scrape crypto OHLCV data of btc from https://coinmarketcap.com and store the data in btc_processed.csv.
The scraping process is shown in crypto_scraper.ipynb
The data are from 28/4/2013 - 7/3/2021
Training set from 1/1/2014 - 31/12/2018
Testing set from 1/1/2019 - 31/12/2020

I may use the glassnode api to request data from glassnode later
The process will be shown in glassnode_api.ipynb

A library backtrader can be used to carry out backtesting on OHLC data.
Csv files obtained above can be feeded to the model
Info of the library can be found in https://www.backtrader.com/ 
I have tried to use the library to backtest the data in btc_processed.csv using a strategy copied from the official documentation
