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
tensorboard<br />
etherscan<br/>

## Usage
1. Run the jupyter notebooks in different `tdqn` to construct tdqn models in different versions

## This version
* action space = 2
* default state = ['Close',"Low","High","Volume","s2f"]
* network = {"DQN", "LSTM", "BiLSTM", "DuelingDQN", "ConvDuelingDQN"}
* cross validation is implemented
* different folders for reducing state space

## todos
* reducing state space by sequential backward selection, and record the results in `useful_results`
* regularisation, dropout layer, early stopping
* continuous action space
* modify Sharpe ratio reward
* optimization

## problems

## Authors
* [leehiulong](https://github.com/leehiulong)
* [e](https://github.com/Nonug)
* [johnnycls](https://github.com/johnnycls)

## Acknowledgements
* [glassnode](https://glassnode.com/)
* [tdqn](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)
* [Cryptocurrency Time Series by GuangZhiXie](https://github.com/guangzhixie/cryptocurrency-time-series)