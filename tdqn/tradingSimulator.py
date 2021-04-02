import os
import importlib
import pickle
import numpy as np
import pandas as pd

from tradingEnv import TradingEnv
from exploratoryDataAnalysis import ExploratoryDataAnalysis


# define default variables
numOfFeatures = 5
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
        startingDate = PARAM['startingDate']
        endingDate = PARAM['endingDate']
        money = PARAM['money']
        strategy = strategiesAI[strategyName]
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
        epsilonStart = PARAM['epsilonStart']
        epsilonEnd = PARAM['epsilonEnd']
        epsilonDecay = PARAM['epsilonDecay']
        transactionCosts=PARAM['transactionCosts']
        name = PARAM['name']
        network = PARAM['network']


        # 2. TRAINING PHASE
        # Initialize the trading environment associated with the training phase
        trainingEnv = TradingEnv(cryptocurrency, startingDate, endingDate, money, name, stateLength, transactionCosts)
        # Instanciate the strategy classes
        strategyModule = importlib.import_module(str(strategy))
        className = getattr(strategyModule, strategy)
        tradingStrategy = className(observationSpace, actionSpace, network, gamma=gamma, learningRate=learningRate,
                                    targetNetworkUpdate=targetNetworkUpdate, epsilonStart=epsilonStart, 
                                    epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay, capacity=capacity, 
                                    batchSize=batchSize)
        # Training of the trading strategy
        trainingEnv = tradingStrategy.training(trainingEnv, name, trainingParameters=trainingParameters,
                                               verbose=verbose, rendering=rendering,
                                               plotTraining=plotTraining, showPerformance=showPerformance)

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
        trainingEnv = TradingEnv(cryptocurrency, startingDate, endingDate, money, name, stateLength, transactionCosts)
        # Instanciate the strategy classes
        strategyModule = importlib.import_module('classicalStrategy')
        className = getattr(strategyModule, strategy)
        tradingStrategy = className()
        # Training of the trading strategy
        trainingEnv = tradingStrategy.training(trainingEnv, name, trainingParameters=trainingParameters,
                                               endingDate=endingDate, verbose=verbose, rendering=rendering,
                                               plotTraining=plotTraining, showPerformance=showPerformance)
        
        # 3. TERMINATION PHASE
        if(saveStrategy):
            fileName = os.path.join("Strategies", strategy+'_'+cryptocurrency)
            fileHandler = open(fileName, 'wb') 
            pickle.dump(tradingStrategy, fileHandler)
        return tradingStrategy, trainingEnv


    def test(self, strategyName, trainCryptocurrencyName, testCryptocurrencyName, PARAM, 
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
        startingDate = PARAM['startingDate']
        endingDate = PARAM['endingDate']
        money = PARAM['money']
        transactionCosts = PARAM['transactionCosts']
        name = PARAM['name']
        stateLength = PARAM['stateLength']
        observationSpace = 1 + (stateLength-1)*numOfFeatures
        network = PARAM['network']


        # 2. LOADING PHASE    
        # Check that the strategy to load exists in the strategy dataset
        fileName = os.path.join("Strategies", strategy+'_'+network+'_'+trainCryptocurrency)
        exists = os.path.isfile(fileName)
        if exists:
            if ai:
                strategyModule = importlib.import_module(strategy)
                className = getattr(strategyModule, strategy)
                tradingStrategy = className(observationSpace, actionSpace, network)
                tradingStrategy.loadModel(fileName)
            else:
                fileHandler = open(fileName, 'rb') 
                tradingStrategy = pickle.load(fileHandler)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")


        # 3. TESTING PHASE
        # Initialize the trading environments associated with the testing phase
        testingEnv = TradingEnv(testCryptocurrency, startingDate, endingDate, money, name, stateLength, transactionCosts)
        testingEnv = tradingStrategy.testing(testingEnv, testingEnv, name, rendering=rendering, showPerformance=showPerformance)
        # log training result
        path = os.path.join('log', testCryptocurrency+'_'+startingDate+'_'+endingDate+'_'+name+'_result.csv')
        logDF = testingEnv.data
        logDF.to_csv(path)

        return tradingStrategy, testingEnv