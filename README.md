# Deep Reinforcement Learning in Cryptocurrency Algorithmic Trading


## Getting Started
### Prerequisites
* **cryptoCMD**<br/>
	`pip install git+git://github.com/guptarohit/cryptoCMD.git`
* **backtrader**<br/>
	`pip install backtrader`<br/>
	`pip install backtrader[plotting]`
* **others**<br/>
	numpy, pandas, Matplotlib...

## Folders
* **scraping_data**<br/>
	storing scripts for scraping data
* **data**<br/>
	storing data scraped from the scripts in `scripts/scraping_data`
* **scripts**<br/>
	* scraping_data<br/>
		storing the scripts for scraping data
	* backtesting<br/>
		storing the scripts for backtesting
* **test**<br/>
	Please ignore it, files in it is for me to test the scripts

## Usage
1. Scrape the data by running the jupyter notebooks in `scripts/scraping_data`
2. Move the data scraped to `data`
3. Run the jupyter notebooks in `scripts`

## things I need to do
- [x] scrape cryptocurrency OHLCV data of btc
- [x] scrape blockchain data from glassnode 
- [x] try using backtrader to backtest the data <br/>
The following data are not able to be scraped due to different reasons shown in the `scraping_data/glassnode_api.ipynb`<br/>
difficulty_ribbon, difficulty_latest, hash_rate_mean, hard, volume_median, utxo_count

## Authors
* [leehiulong](https://github.com/leehiulong)
* [e](https://github.com/Nonug)
* [johnnycls](https://github.com/johnnycls)

## Acknowledgements
* [cryptoCMD](https://github.com/guptarohit/cryptoCMD)
* [glassnode](https://glassnode.com/)
* [backtrader](https://www.backtrader.com/)