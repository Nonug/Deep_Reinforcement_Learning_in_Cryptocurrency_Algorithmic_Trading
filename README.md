# Deep Reinforcement Learning in Cryptocurrency Algorithmic Trading


## Getting Started
### Prerequisites
* cryptoCMD
	* `pip install git+git://github.com/guptarohit/cryptoCMD.git`
* backtrader
	* `pip install backtrader`
	* `pip install backtrader[plotting]`
* others
	* numpy, pandas, Matplotlib...

### Folders
#### scraping_datas
storing scripts for scraping datas
#### datas
storing datas scraped from the scripts in scraping_datas
#### scripts
storing the scripts for ml model or other things
#### test
Please ignore it, files in it is for me to test the scripts

### Usage
1. Scrape the data by running the jupyter notebooks in `scraping_datas`
2. Move the data scraped to `datas`
3. Run the jupyter notebooks in `scripts`

## things I need to do
- [x] scrape cryptocurrency OHLCV data of btc
- [] scrape blockchain data from glassnode 
- [x] try using backtrader to backtest the data  

## things I need help
- [] tell me what data from glassnode I need to scrape
- [] check whether the data I scraped can be used
- [] look at the backtest result to check whether the backtest is done correctly

## Authors
* [leehiulong](https://github.com/leehiulong)
* [e](https://github.com/Nonug)
* [johnnycls](https://github.com/johnnycls)

## Acknowledgements
[cryptoCMD](https://github.com/guptarohit/cryptoCMD)
[glassnode](https://glassnode.com/)
[backtrader](https://www.backtrader.com/)