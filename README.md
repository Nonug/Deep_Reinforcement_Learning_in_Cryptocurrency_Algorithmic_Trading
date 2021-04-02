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
etherscan<br/>

## Usage
1. Run the jupyter notebooks in `tdqn` to construct the tdqn model

## version
version for interim prototype
* action space = 2
* state = ['Close',"Low","High","Volume","s2f"]
* basic TDQN 
* experiments on different network can be carry out by including network in the param
* no eth test because haven't scraped eth s2f


## problems
* BiLSTM and ConvDuelingDQN do not work, I believe it's easier to debug for BiLSTM


## Authors
* [leehiulong](https://github.com/leehiulong)
* [e](https://github.com/Nonug)
* [johnnycls](https://github.com/johnnycls)

## Acknowledgements
* [glassnode](https://glassnode.com/)
* [tdqn](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)
* [Cryptocurrency Time Series by GuangZhiXie](https://github.com/guangzhixie/cryptocurrency-time-series)