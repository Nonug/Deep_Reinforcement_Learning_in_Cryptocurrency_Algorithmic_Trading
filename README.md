# Deep Reinforcement Learning in Cryptocurrency Algorithmic Trading


## Getting Started
### Prerequisites
* **cryptoCMD (may not be used)**<br/>
	`pip install git+git://github.com/guptarohit/cryptoCMD.git`
* **backtrader**<br/>
	`pip install backtrader`<br/>
	`pip install backtrader[plotting]`
* **others**<br/>
	numpy<br/>
	pandas<br/>
	Matplotlib<br/>
	gym<br/>
	pickle<br/>
	tqdm<br/>
	tabulate<br/>
	torch<br/>
	pandas.plotting<br/>

## Usage
1. Scrape the data by running the jupyter notebooks in `scraping_data`
2. Run the jupyter notebooks in `preprocess` to preprocess and format the data
3. Run the jupyter notebooks in `tdqn` to construct the tdqn model

## things I have/haven't done
- [x] scrape cryptocurrency OHLCV data of btc
- [x] scrape blockchain data from glassnode 
- [x] try using backtrader to backtest the data 
- [ ] modified the tdqn for our data (half-finished)<br/>
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
* [tdqn](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)