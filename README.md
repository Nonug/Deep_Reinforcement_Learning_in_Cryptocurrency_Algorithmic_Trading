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
version for final work
* action space = 2
* state = ['Close',"Low","High","Volume","s2f"]
* network = {"DQN", "LSTM", "BiLSTM", "DuelingDQN"}
* cross validation is implemented

## todos
* ConvDuelingDQN (idk how to modify it properly)
* reducing state space by sequential backward selection
* regularisation, dropout layer, early stopping
* continuous action space
* modify Sharpe ratio reward
* optimization


## Authors
* [leehiulong](https://github.com/leehiulong)
* [e](https://github.com/Nonug)
* [johnnycls](https://github.com/johnnycls)

## Acknowledgements
* [glassnode](https://glassnode.com/)
* [tdqn](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)
* [Cryptocurrency Time Series by GuangZhiXie](https://github.com/guangzhixie/cryptocurrency-time-series)