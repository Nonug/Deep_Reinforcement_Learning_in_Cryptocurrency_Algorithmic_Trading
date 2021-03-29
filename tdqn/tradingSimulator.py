import os
import sys
import importlib
import pickle
import itertools

import numpy as np
import pandas as pd

from tabulate import tabulate
from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from tradingEnv import TradingEnv
from timeSeriesAnalyser import TimeSeriesAnalyser
from tradingPerformance import PerformanceEstimator
from TDQN import TDQN

# Variables defining the default observation and state spaces
stateLength = 30
observationSpace = 1 + (stateLength-1)*4
actionSpace = 2

# Variables setting up the default transaction costs
percentageCosts = [0, 0.1, 0.2]
transactionCosts = percentageCosts[1]/100

# Variables specifying the default capital at the disposal of the trader
money = 100000

# Variables specifying the default general training parameters
bounds = [1, 30]
step = 1
numberOfEpisodes = 50

# Dictionary listing the cryptocurrencies supported
cryptocurrencies = {
    'Bitcoin': 'btc',
    'Google': 'GOOGL',
    'Apple': 'AAPL'
}

# Dictionary listing the classical trading strategies supported
strategies = {
    'Buy and Hold' : 'BuyAndHold',
    'Sell and Hold' : 'SellAndHold',
    'Trend Following Moving Averages' : 'MovingAveragesTF',
    'Mean Reversion Moving Averages' : 'MovingAveragesMR'
}

# Dictionary listing the AI trading strategies supported
strategiesAI = {
    'TDQN' : 'TDQN'
}

class TradingSimulator:
    """          
    METHODS:   - simulateNewStrategy: Simulate a new trading strategy on a certain cryptocurrency of the testbench.
               - simulateExistingStrategy: Simulate an already existing trading strategy on a certain cryptocurrency 
                                           of the testbench.
   """
    def analyseTimeSeries(self, cryptocurrencyName, startingDate, endingDate):           
        """
        GOAL: Perform a detailled analysis of the stock market
              price time series.
        
        INPUTS: - cryptocurrencyName: Name of the cryptocurrency (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
        
        OUTPUTS: /
        """

        # Retrieve the trading cryptocurrencies information
        if(cryptocurrencyName in cryptocurrencies):
            cryptocurrency = cryptocurrencies[cryptocurrencyName]   
        # Error message if the cryptocurrency specified is not valid or not supported
        else:
            print("The cryptocurrency specified is not valid, only the following cryptocurrencies are supported:")
            for cryptocurrency in cryptocurrencies:
                print("".join(['- ', cryptocurrency]))
            raise SystemError("Please check the cryptocurrency specified.")
        
        # ENTIRE TRADING DATA
        print("\n\n\nAnalysis of the entire time series (both training and testing phases)")
        print("---------------------------------------------------------------------\n")
        tradingEnv = TradingEnv(cryptocurrency, startingDate, endingDate, 0)
        timeSeries = tradingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

    def plotEntireTrading(self, trainingEnv, validationEnv, name, splitingDate):
        """
        GOAL: Plot the entire trading activity, with both the training
              and testing/validation phases rendered on the same graph for
              comparison purposes.
        
        INPUTS: - trainingEnv: Trading environment for training.
                - testing/validationEnv: Trading environment for validation.
        
        OUTPUTS: /
        """
        # Artificial trick to assert the continuity of the Money curve
        ratio = trainingEnv.data['Money'][-1]/validationEnv.data['Money'][0]
        validationEnv.data['Money'] = ratio * validationEnv.data['Money']

        # Concatenation of the training and testing trading dataframes
        dataframes = [trainingEnv.data, validationEnv.data]
        data = pd.concat(dataframes)

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the cryptocurrency market price
        trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
        validationEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_') 
        ax1.plot(data.loc[data['Action'] == 1.0].index, 
                 data['Close'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax1.plot(data.loc[data['Action'] == -1.0].index, 
                 data['Close'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Plot the second graph -> Evolution of the trading capital
        trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
        validationEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_') 
        ax2.plot(data.loc[data['Action'] == 1.0].index, 
                 data['Money'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax2.plot(data.loc[data['Action'] == -1.0].index, 
                 data['Money'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the vertical line seperating the training and testing datasets
        ax1.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
        ax2.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short", "Train/"+name+" separation"])
        ax2.legend(["Capital", "Long", "Short", "Train/"+name+" separation"])
        plt.savefig(os.path.join('Figures',str(trainingEnv.marketSymbol)+'_Training'+name+'Rendering.png')) 
        #plt.show()



    def trainAndValidate(self, strategyName, cryptocurrencyName, PARAM, observationSpace=observationSpace, 
        actionSpace=actionSpace, stateLength=stateLength, transactionCosts=transactionCosts, bounds=bounds, 
        step=step, verbose=True, plotTraining=True, rendering=True, showPerformance=True, saveStrategy=True):
        """
        GOAL: Simulate a new trading strategy on a certain cryptocurrency included in the
              testbench, with both learning and validation phases.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - cryptocurrencyName: Name of the cryptocurrency (in the testbench).
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - verbose: Enable the printing of a simulation feedback.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - validationEnv
        """

        # 1. INITIALIZATION PHASE
        startingDate = PARAM['startingDate']
        endingDate = PARAM['endingDate']
        splitingDate = PARAM['splitingDate']
        money = PARAM['money']
        # Retrieve the trading strategy information
        if(strategyName in strategies):
            strategy = strategies[strategyName]
            trainingParameters = [bounds, step]
            ai = False
        elif(strategyName in strategiesAI):
            strategy = strategiesAI[strategyName]
            ai = True
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
        # Error message if the strategy specified is not valid or not supported
        else:
            print("The strategy specified is not valid, only the following strategies are supported:")
            for strategy in strategies:
                print("".join(['- ', strategy]))
            for strategy in strategiesAI:
                print("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        # Retrieve the trading cryptocurrencies information
        if(cryptocurrencyName in cryptocurrencies):
            cryptocurrency = cryptocurrencies[cryptocurrencyName]   
        # Error message if the cryptocurrency specified is not valid or not supported
        else:
            print("The cryptocurrency specified is not valid, only the following cryptocurrencies are supported:")
            for cryptocurrency in cryptocurrencies:
                print("".join(['- ', cryptocurrency]))
            raise SystemError("Please check the cryptocurrency specified.")


        # 2. TRAINING PHASE
        # Initialize the trading environment associated with the training phase
        trainingEnv = TradingEnv(cryptocurrency, startingDate, splitingDate, money, stateLength, transactionCosts)

        # Instanciate the strategy classes
        if ai:
            strategyModule = importlib.import_module(str(strategy))
            className = getattr(strategyModule, strategy)
            tradingStrategy = className(observationSpace, actionSpace, gamma=gamma, learningRate=learningRate,
                                        targetNetworkUpdate=targetNetworkUpdate, epsilonStart=epsilonStart, 
                                        epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay, capacity=capacity, 
                                        batchSize=batchSize)
        else:
            strategyModule = importlib.import_module('classicalStrategy')
            className = getattr(strategyModule, strategy)
            tradingStrategy = className()

        # Training of the trading strategy
        trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters,
                                               endingDate=endingDate, verbose=verbose, rendering=rendering,
                                               plotTraining=plotTraining, showPerformance=showPerformance)
        
        # 3. VALIDATION PHASE

        # Initialize the trading environment associated with the validation phase
        validationEnv = TradingEnv(cryptocurrency, splitingDate, endingDate, money, stateLength, transactionCosts)

        # Validation of the trading strategy
        validationEnv = tradingStrategy.testing(trainingEnv, validationEnv, rendering=rendering, showPerformance=showPerformance)
            
        # Show the entire unified rendering of the training and validation phases
        if rendering:
            self.plotEntireTrading(trainingEnv, validationEnv, 'Validation', splitingDate)

        # 4. TERMINATION PHASE
        # If required, save the trading strategy with Pickle
        if(saveStrategy):
            fileName = os.path.join("Strategies", strategy+'_'+cryptocurrency+'_'+startingDate+'_'+endingDate)
            if ai:
                tradingStrategy.saveModel(fileName)
            else:
                fileHandler = open(fileName, 'wb') 
                pickle.dump(tradingStrategy, fileHandler)

        # Return of the trading strategy simulated and of the trading environments backtested
        return tradingStrategy, trainingEnv, validationEnv

    
    def test(self, strategyName, cryptocurrencyName, PARAM, observationSpace=observationSpace, 
            actionSpace=actionSpace, stateLength=stateLength, transactionCosts=transactionCosts,
            rendering=True, showPerformance=True):
        """
        GOAL: Simulate an already existing trading strategy on a certain
              cryptocurrency of the testbench, the strategy being loaded from the
              strategy dataset. There is no training phase, only a testing
              phase.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - cryptocurrencyName: Name of the cryptocurrency (in the testbench).
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """
        startingDate = PARAM['startingDate']
        endingDate = PARAM['endingDate']
        splitingDate = PARAM['splitingDate']
        money = PARAM['money']
        # 1. INITIALIZATION PHASE

        # Retrieve the trading strategy information
        if(strategyName in strategies):
            strategy = strategies[strategyName]
            ai = False
        elif(strategyName in strategiesAI):
            strategy = strategiesAI[strategyName]
            ai = True
        # Error message if the strategy specified is not valid or not supported
        else:
            print("The strategy specified is not valid, only the following strategies are supported:")
            for strategy in strategies:
                print("".join(['- ', strategy]))
            for strategy in strategiesAI:
                print("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        # Retrieve the trading cryptocurrency information
        if(cryptocurrencyName in cryptocurrencies):
            cryptocurrency = cryptocurrencies[cryptocurrencyName]
        # Error message if the cryptocurrency specified is not valid or not supported
        else:
            print("The cryptocurrency specified is not valid, only the following cryptocurrencies are supported:")
            for cryptocurrency in cryptocurrencies:
                print("".join(['- ', cryptocurrency]))
            raise SystemError("Please check the cryptocurrency specified.")

        # 2. LOADING PHASE    
        # Check that the strategy to load exists in the strategy dataset
        fileName = os.path.join("Strategies", strategy+'_'+cryptocurrency+'_'+startingDate+'_'+splitingDate)
        exists = os.path.isfile(fileName)
        # If affirmative, load the trading strategy
        if exists:
            if ai:
                strategyModule = importlib.import_module(strategy)
                className = getattr(strategyModule, strategy)
                tradingStrategy = className(observationSpace, actionSpace)
                tradingStrategy.loadModel(fileName)
            else:
                fileHandler = open(fileName, 'rb') 
                tradingStrategy = pickle.load(fileHandler)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")


        # 3. TESTING PHASE
        # Initialize the trading environments associated with the testing phase
        trainingEnv = TradingEnv(cryptocurrency, startingDate, splitingDate, money, stateLength, transactionCosts)
        testingEnv = TradingEnv(cryptocurrency, splitingDate, endingDate, money, stateLength, transactionCosts)

        # Testing of the trading strategy
        trainingEnv = tradingStrategy.testing(trainingEnv, trainingEnv, rendering=rendering, showPerformance=showPerformance)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=rendering, showPerformance=showPerformance)

        # Show the entire unified rendering of the training and testing phases
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv, 'Testing', splitingDate)
            
        # log training result
        path = os.path.join('log', 'test.csv')
        logDF = testingEnv.data
        logDF.to_csv(path)

        return tradingStrategy, trainingEnv, testingEnv