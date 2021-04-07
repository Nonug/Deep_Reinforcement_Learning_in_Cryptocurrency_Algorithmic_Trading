import os
import importlib
import pickle
import numpy as np
import pandas as pd

from tradingEnv import TradingEnv
from exploratoryDataAnalysis import ExploratoryDataAnalysis


# define default variables
numOfFeatures = 4 #reduce state
actionSpace = 2
bounds = [1, 30]
step = 1
cryptocurrencies = {
    'Bitcoin': 'btc',
    'Ethereum': 'eth',
}
strategies = {
    'Buy and Hold' : 'BuyAndHold',
    'Sell and Hold' : 'SellAndHold',
    'Trend Following Moving Averages' : 'MovingAveragesTF',
    'Mean Reversion Moving Averages' : 'MovingAveragesMR'
}
strategiesAI = {
    'TDQN' : 'TDQN'
}


class TradingSimulator:
    """
    METHODS:    -exploratoryDataAnalysis
                -aiTrain
                -nonAiTrain
                -test
    """


    def exploratoryDataAnalysis(self, cryptocurrencyName, startingDate, endingDate):
        # Retrieve the trading cryptocurrencies information
        if(cryptocurrencyName in cryptocurrencies):
            cryptocurrency = cryptocurrencies[cryptocurrencyName]   
        else:
            raise SystemError("Please check the cryptocurrency specified.")
        
        tradingEnv = TradingEnv(cryptocurrency, startingDate, endingDate)
        analyser = ExploratoryDataAnalysis(tradingEnv.data, cryptocurrency)
        analyser.plotTimeSeries()
        analyser.plot_daily_returns()
        analyser.bollingerBandStrategy()
        analyser.plot_scarcity()
        analyser.check_empty_duplicate()


    def aiTrain(self, strategyName, cryptocurrencyName, PARAM, 
        verbose=True, plotTraining=True, rendering=True, showPerformance=True, saveStrategy=True):

        # 1. INITIALIZATION PHASE
        if(not (strategyName in strategiesAI)):
            raise SystemError("Please check the trading strategy specified.")
        # Retrieve the trading cryptocurrencies information
        if(cryptocurrencyName in cryptocurrencies):
            cryptocurrency = cryptocurrencies[cryptocurrencyName]   
        else:
            raise SystemError("Please check the cryptocurrency specified.")

        # initialize variables
        strategy = strategiesAI[strategyName]
        startingDate = PARAM['startingDate']
        endingDate = PARAM['endingDate']
        money = PARAM['money']
        transactionCosts=PARAM['transactionCosts']
        name = PARAM['name']
        network = PARAM['network']
        stateLength = PARAM['stateLength']
        observationSpace = 1 + (stateLength-1)*numOfFeatures
        numberOfEpisodes = PARAM['numberOfEpisodes']
        trainingParameters = [numberOfEpisodes]
        gamma = PARAM['gamma']
        learningRate = PARAM['learningRate']
        targetNetworkUpdate = PARAM['targetNetworkUpdate']
        learningUpdatePeriod = PARAM['learningUpdatePeriod']
        capacity = PARAM['capacity']
        batchSize = PARAM['batchSize']
        experiencesRequired = PARAM['experiencesRequired']
        numberOfNeurons = PARAM['numberOfNeurons']
        dropout = PARAM['dropout']
        epsilonStart = PARAM['epsilonStart']
        epsilonEnd = PARAM['epsilonEnd']
        epsilonDecay = PARAM['epsilonDecay']
        alpha = PARAM['alpha']
        filterOrder = PARAM['filterOrder']
        gradientClipping = PARAM['gradientClipping']
        rewardClipping = PARAM['rewardClipping']
        L2Factor = PARAM['L2Factor']


        # 2. TRAINING PHASE
        # Initialize the trading environment associated with the training phase
        trainingEnv = TradingEnv(cryptocurrency, startingDate, endingDate, money, name, stateLength, 
                                transactionCosts)
        # Instanciate the strategy classes
        strategyModule = importlib.import_module(str(strategy))
        className = getattr(strategyModule, strategy)
        tradingStrategy = className(observationSpace, actionSpace, network, stateLength, numOfFeatures, 
                                    numberOfNeurons, gamma, learningRate, targetNetworkUpdate, epsilonStart, 
                                    epsilonEnd, epsilonDecay, capacity, batchSize, alpha, filterOrder,
                                    gradientClipping, rewardClipping, L2Factor)
        # Training of the trading strategy
        trainingEnv = tradingStrategy.training(trainingEnv, name, trainingParameters, verbose, rendering,
                                               plotTraining, showPerformance)

        # 3. TERMINATION PHASE
        if(saveStrategy):
            fileName = os.path.join("Strategies", strategy+'_'+network+'_'+cryptocurrency)
            tradingStrategy.saveModel(fileName)
        return tradingStrategy, trainingEnv

    
    def nonAiTrain(self, strategyName, cryptocurrencyName, PARAM, 
        verbose=True, plotTraining=True, rendering=True, showPerformance=True, saveStrategy=True):

        # 1. INITIALIZATION PHASE
        # Retrieve the trading strategy information
        if(not (strategyName in strategies)):
            raise SystemError("Please check the trading strategy specified.")
        # Retrieve the trading cryptocurrencies information
        if(cryptocurrencyName in cryptocurrencies):
            cryptocurrency = cryptocurrencies[cryptocurrencyName]   
        else:
            raise SystemError("Please check the cryptocurrency specified.")  
        # initialize variables      
        startingDate = PARAM['startingDate']
        endingDate = PARAM['endingDate']
        money = PARAM['money']
        strategy = strategies[strategyName]
        trainingParameters = [bounds, step]
        stateLength = PARAM['stateLength']
        observationSpace = 1 + (stateLength-1)*numOfFeatures
        transactionCosts=PARAM['transactionCosts']
        name = PARAM['name']

        # 2. TRAINING PHASE
        # Initialize the trading environment associated with the training phase
        trainingEnv = TradingEnv(cryptocurrency, startingDate, endingDate, money, name, stateLength, 
                                transactionCosts)
        # Instanciate the strategy classes
        strategyModule = importlib.import_module('classicalStrategy')
        className = getattr(strategyModule, strategy)
        tradingStrategy = className()
        # Training of the trading strategy
        trainingEnv = tradingStrategy.training(trainingEnv, name, trainingParameters=trainingParameters,
                                               verbose=verbose, rendering=rendering,
                                               plotTraining=plotTraining, showPerformance=showPerformance)
        
        # 3. TERMINATION PHASE
        if(saveStrategy):
            fileName = os.path.join("Strategies", strategy+'_'+cryptocurrency)
            fileHandler = open(fileName, 'wb') 
            pickle.dump(tradingStrategy, fileHandler)
        return tradingStrategy, trainingEnv


    def aiTrainWithCrossValidation(self, strategyName, cryptocurrencyName, TRAIN_PARAM, VALIDATION_PARAM, 
        verbose=True, plotTraining=True, rendering=True, showPerformance=True, saveStrategy=True):
        validationName = VALIDATION_PARAM['name']
        TRAIN_PARAM['startingDate'] = "2014-01-01"
        TRAIN_PARAM['endingDate'] = "2015-01-01"
        print("Training1")
        self.aiTrain(strategyName, cryptocurrencyName, TRAIN_PARAM)
        VALIDATION_PARAM['name'] = validationName + '1'
        VALIDATION_PARAM['startingDate'] = "2015-01-01"
        VALIDATION_PARAM['endingDate'] = "2019-01-01"
        print("Validation1")
        self.test(strategyName, cryptocurrencyName, cryptocurrencyName, TRAIN_PARAM, VALIDATION_PARAM)

        TRAIN_PARAM['startingDate'] = "2014-01-01"
        TRAIN_PARAM['endingDate'] = "2016-01-01"
        print("Training2")
        self.aiTrain(strategyName, cryptocurrencyName, TRAIN_PARAM)
        VALIDATION_PARAM['name'] = validationName + '2'
        VALIDATION_PARAM['startingDate'] = "2016-01-01"
        VALIDATION_PARAM['endingDate'] = "2019-01-01"
        print("Validation2")
        self.test(strategyName, cryptocurrencyName, cryptocurrencyName, TRAIN_PARAM, VALIDATION_PARAM)

        TRAIN_PARAM['startingDate'] = "2014-01-01"
        TRAIN_PARAM['endingDate'] = "2017-01-01"
        print("Training3")
        self.aiTrain(strategyName, cryptocurrencyName, TRAIN_PARAM)
        VALIDATION_PARAM['name'] = validationName + '3'
        VALIDATION_PARAM['startingDate'] = "2017-01-01"
        VALIDATION_PARAM['endingDate'] = "2019-01-01"
        print("Validation3")
        self.test(strategyName, cryptocurrencyName, cryptocurrencyName, TRAIN_PARAM, VALIDATION_PARAM)

        TRAIN_PARAM['startingDate'] = "2014-01-01"
        TRAIN_PARAM['endingDate'] = "2018-01-01"
        print("Training4")
        self.aiTrain(strategyName, cryptocurrencyName, TRAIN_PARAM)
        VALIDATION_PARAM['name'] = validationName + '4'
        VALIDATION_PARAM['startingDate'] = "2018-01-01"
        VALIDATION_PARAM['endingDate'] = "2019-01-01"
        print("Validation4")
        self.test(strategyName, cryptocurrencyName, cryptocurrencyName, TRAIN_PARAM, VALIDATION_PARAM)

        TRAIN_PARAM['startingDate'] = "2014-01-01"
        TRAIN_PARAM['endingDate'] = "2019-01-01"
        print("Training5")
        self.aiTrain(strategyName, cryptocurrencyName, TRAIN_PARAM)


    def test(self, strategyName, trainCryptocurrencyName, testCryptocurrencyName, TRAIN_PARAM, TEST_PARAM, 
            rendering=True, showPerformance=True):

        # 1. INITIALIZATION PHASE
        if(strategyName in strategies):
            strategy = strategies[strategyName]
            ai = False
        elif(strategyName in strategiesAI):
            strategy = strategiesAI[strategyName]
            ai = True
        else:
            raise SystemError("Please check the trading strategy specified.")
        if(testCryptocurrencyName in cryptocurrencies and trainCryptocurrencyName in cryptocurrencies):
            testCryptocurrency = cryptocurrencies[testCryptocurrencyName]
            trainCryptocurrency = cryptocurrencies[trainCryptocurrencyName]
        else:
            raise SystemError("Please check the cryptocurrency specified.")

        # initialize variables
        trainStartingDate = TRAIN_PARAM['startingDate']
        trainEndingDate = TRAIN_PARAM['endingDate']
        trainMoney = TRAIN_PARAM['money']
        trainTransactionCosts = TRAIN_PARAM['transactionCosts']
        trainName = TRAIN_PARAM['name']
        trainStateLength = TRAIN_PARAM['stateLength']

        startingDate = TEST_PARAM['startingDate']
        endingDate = TEST_PARAM['endingDate']
        money = TEST_PARAM['money']
        transactionCosts = TEST_PARAM['transactionCosts']
        name = TEST_PARAM['name']
        stateLength = TEST_PARAM['stateLength']
        observationSpace = 1 + (stateLength-1)*numOfFeatures
        network = TEST_PARAM['network']


        # 2. LOADING PHASE    
        # Check that the strategy to load exists in the strategy dataset
        if ai:
            fileName = os.path.join("Strategies", strategy+'_'+network+'_'+trainCryptocurrency)
        else:
            fileName = os.path.join("Strategies", strategy+'_'+trainCryptocurrency)
        exists = os.path.isfile(fileName)
        if exists:
            if ai:
                strategyModule = importlib.import_module(strategy)
                className = getattr(strategyModule, strategy)
                tradingStrategy = className(observationSpace, actionSpace, network, stateLength, numOfFeatures)
                tradingStrategy.loadModel(fileName)
            else:
                fileHandler = open(fileName, 'rb') 
                tradingStrategy = pickle.load(fileHandler)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")


        # 3. TESTING PHASE
        # Initialize the trading environments associated with the testing phase
        trainingEnv = TradingEnv(trainCryptocurrency, trainStartingDate, trainEndingDate, trainMoney, 
                                trainName, trainStateLength, trainTransactionCosts)
        testingEnv = TradingEnv(testCryptocurrency, startingDate, endingDate, money, name, stateLength, 
                                transactionCosts)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, name, rendering=rendering, 
                                            showPerformance=showPerformance)
        
        # log training result
        path = os.path.join('log', testCryptocurrency+'_'+startingDate+'_'+endingDate+'_'+name+'_result.csv')
        logDF = testingEnv.data
        logDF.to_csv(path)

        return tradingStrategy, testingEnv