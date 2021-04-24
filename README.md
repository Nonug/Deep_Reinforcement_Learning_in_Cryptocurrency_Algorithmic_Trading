# Deep Reinforcement Learning in Cryptocurrency Algorithmic Trading :chart_with_upwards_trend:

## Getting Started

### Requirements

matplotlib==3.3.2
tabulate==0.8.9<br/>
tqdm==4.50.0<br/>
statsmodels==0.12.1<br/>
numpy==1.19.5<br/>
pandas==1.1.2<br/>
torch==1.7.1+cu110<br/>
gym==0.17.1<br/>


## Usage

### Testing existing trained model

1. Run cells in `test_prototype_and_classics.ipynb` to show EDA, test the performance of the non-AI strategies and DQN agent trained from interim prototype.
2. Run cells in `testStrategy_finalworks.ipynb` to test the performance of the 3 RL agents trained from our final work.
3. The evaluation metrics will be shown in the jupyter notebook and the figures will be stored in `tdqn_final/Figures`and `tdqn_prototype/Figures`

### Training new models and optimisation (not recommended)

1. Run the notebooks in `tdqn_final` or `tdqn_prototype` to train the models, the trained models will be stored in their `tdqn/Strategies`
2. The instructions are stated in the jupyter notebook
3. It is not recommended as hours are required to retrain the model

## Authors

- [leehiulong](https://github.com/leehiulong)
- [e](https://github.com/Nonug) ( Chan Tsz Hei)
- [johnnycls](https://github.com/johnnycls)

## Acknowledgements

- [glassnode](https://glassnode.com/)
- [tdqn](https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading)
- [Cryptocurrency Time Series by GuangZhiXie](https://github.com/guangzhixie/cryptocurrency-time-series)
