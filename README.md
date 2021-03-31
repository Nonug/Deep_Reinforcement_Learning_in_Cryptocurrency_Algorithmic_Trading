# Deep Reinforcement Learning in Cryptocurrency Algorithmic Trading


## Getting Started
### Prerequisites
numpy<br/>
pandas<br/>
Matplotlib<br/>
gym<br/>
tqdm<br/>
tabulate<br/>
torch<br/>

## Usage
1. Run the jupyter notebooks in `tdqn` to construct the tdqn model

## things I have/haven't done
- [x] add stock to flow ratio to data
- [x] eda: plotTimeSeries, daily_returns, adf test, 2 versions for bollingerBand
- [ ] eda: Scarcity plot

## problems
* glassnode does not provide stock to flow of eth
* is daysTillHalving necessary for stock to flow?
* is this split appropriate: training: 2014-2016, validation: 2017-2018, testing: 2019-2020
* which bollingerband graph is better?


## Authors
* [leehiulong](https://github.com/leehiulong)
* [e](https://github.com/Nonug)
* [johnnycls](https://github.com/johnnycls)

## Acknowledgements
* [glassnode](https://glassnode.com/)
* [tdqn](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)