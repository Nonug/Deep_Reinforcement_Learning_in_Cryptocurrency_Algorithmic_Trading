# import the files
from tradingSimulator import TradingSimulator
simulator = TradingSimulator()

# set parameters after optimization
percentageCosts = [0, 0.1, 0.2]
transactionCosts = percentageCosts[1]/100
stateLength = 30
strategy = "TDQN"
trainCryptocurrency = "Bitcoin"
testCryptocurrency = "Bitcoin"
AI_TRAINING_PARAM = {
    'startingDate': '2014-01-01',
    'endingDate': '2017-01-01',
    'money': 100000,
    'percentageCosts': percentageCosts,
    'transactionCosts': transactionCosts,
    'name': 'training',
    'network': '',
    # Parameters related to the DQN algorithm
    'stateLength': stateLength,
    'numberOfEpisodes': 1,
    'gamma': 0.4,
    'learningRate': 0.0001,
    'targetNetworkUpdate': 1000,
    'learningUpdatePeriod': 1,
    # Default parameters related to the Experience Replay mechanism
    'capacity': 100000,
    'batchSize': 32,
    'experiencesRequired': 1000,
    # Default parameters related to the Deep Neural Network
    'numberOfNeurons': 512,
    'dropout': 0.2,
    # Default parameters related to the Epsilon-Greedy exploration technique
    'epsilonStart': 1.0,
    'epsilonEnd': 0.01,
    'epsilonDecay': 10000,
    # Default parameters regarding the sticky actions RL generalization technique
    'alpha': 0.1,
    # Default parameters related to preprocessing
    'filterOrder': 5,
    # Default paramters related to the clipping of both the gradient and the RL rewards
    'gradientClipping': 1,
    'rewardClipping': 1,
    # Default parameter related to the L2 Regularization
    'L2Factor': 0.000001,
}

NON_AI_TRAINING_PARAM = {
    'startingDate': '2014-01-01',
    'endingDate': '2019-01-01',
    'money': 100000,
    'stateLength': stateLength,
    'percentageCosts': percentageCosts,
    'transactionCosts': transactionCosts,
    'name': 'training',
    'network': ''
}

VALIDATION_PARAM = {
    'money': 100000,
    'stateLength': stateLength,
    'transactionCosts': transactionCosts,
    'name': 'validation',
    'network': '',
}

BACKTEST_PARAM = {
    'startingDate': '2014-01-01',
    'endingDate': '2021-01-01',
    'money': 100000,
    'stateLength': stateLength,
    'transactionCosts': transactionCosts,
    'name': 'backtest',
    'network': '',
}

TEST_PARAM = {
    'startingDate': '2019-01-01',
    'endingDate': '2021-01-01',
    'money': 100000,
    'stateLength': stateLength,
    'transactionCosts': transactionCosts,
    'name': 'test',
    'network': '',
}

# ETH_TEST_PARAM = {
#     'startingDate': '2019-01-01',
#     'endingDate': '2021-01-01',
#     'money': 100000,
#     'stateLength': stateLength,
#     'transactionCosts': transactionCosts,
#     'name': 'test',
#     'network': '',
# }

AI_TRAINING_PARAM['network'] = 'DQN'
AI_TRAINING_PARAM['name'] = 'btc_tdqn_DQN_training'
VALIDATION_PARAM['network'] = 'DQN'
VALIDATION_PARAM['name'] = 'btc_tdqn_DQN_validation'
BACKTEST_PARAM['name'] = 'btc_tdqn_DQN_backtest'
BACKTEST_PARAM['network'] = 'DQN'

classical = ['Buy and Hold', 'Sell and Hold',
'Trend Following Moving Averages', 'Mean Reversion Moving Averages']

classicalStrats = {
    'Buy and Hold' : 'b&h',
    'Sell and Hold' : 's&h',
    'Trend Following Moving Averages' : 'tfma',
    'Mean Reversion Moving Averages' : 'mrma'
}
AI = ['DQN', 'LSTM', 'ConvDuelingDQN']

# def backtestAI(network,name = "", animation = False):
#     AI_TRAINING_PARAM['network'] = network
#     AI_TRAINING_PARAM['name'] = 'btc_tdqn_' + network + '_' + name + '_training'
#
#     BACKTEST_PARAM['name'] = 'btc_tdqn_' + network + '_' + name + '_backtest'
#     BACKTEST_PARAM['network'] = network
#
#     strategy = "TDQN"
#
#     simulator.test(strategy, trainCryptocurrency, testCryptocurrency, AI_TRAINING_PARAM, BACKTEST_PARAM)
#     print()

def testAI(network,name = "", animation = False, mode = 'test'):
    AI_TRAINING_PARAM['network'] = network
    AI_TRAINING_PARAM['name'] = 'btc_tdqn_' + network + '_' + name + '_training'

    TEST_PARAM['name'] = 'btc_tdqn_' + network + '_' + name + '_test'
    TEST_PARAM['network'] = network

    BACKTEST_PARAM['name'] = 'btc_tdqn_' + network + '_' + name + '_backtest'
    BACKTEST_PARAM['network'] = network

    strategy = "TDQN"

    if mode == 'test':
        simulator.test(strategy, trainCryptocurrency, testCryptocurrency, TEST_PARAM, TEST_PARAM)
    elif mode == 'train':
        simulator.test(strategy, trainCryptocurrency, testCryptocurrency, AI_TRAINING_PARAM, TEST_PARAM)
    else:
        simulator.test(strategy, trainCryptocurrency, testCryptocurrency, BACKTEST_PARAM, TEST_PARAM)
    print()

def train(strategy, animation = False):

    if strategy not in classical:
        print('strategy given is not valid')
        return

    NON_AI_TRAINING_PARAM['name'] = 'btc_' + classicalStrats[strategy] + '_training'
    NON_AI_TRAINING_PARAM['network'] = ''
    simulator.nonAiTrain(strategy, trainCryptocurrency, NON_AI_TRAINING_PARAM)
    print()

# def test(strategy, animation = False):
#
#     if strategy not in classical:
#         print('strategy given is not valid')
#         return
#
#     NON_AI_TRAINING_PARAM['name'] = 'btc_' + classicalStrats[strategy] + '_training'
#     NON_AI_TRAINING_PARAM['network'] = ''
#     simulator.test(strategy, trainCryptocurrency, NON_AI_TRAINING_PARAM)
#     print()

def backtest(strategy, animation = False):

    if strategy not in classical:
        print('strategy given is not valid')
        return

    NON_AI_TRAINING_PARAM['name'] = 'btc_' + classicalStrats[strategy] + '_training'
    NON_AI_TRAINING_PARAM['network'] = ''

    BACKTEST_PARAM['name'] = 'btc_' + classicalStrats[strategy] + '_backtest'
    BACKTEST_PARAM['network'] = ''
    simulator.nonAiTrain(strategy, trainCryptocurrency, NON_AI_TRAINING_PARAM)
    print()
