#Implementing trading environment
import os
import gym
import math
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation

class TradingEnv(gym.Env):
    # GOAL: Implement a custom trading environment compatible with OpenAI Gym.

    def __init__(self, marketSymbol, startingDate, endingDate, money=0, name="", stateLength=30, transactionCosts=0):
        # GOAL: Object constructor initializing the trading environment by setting up
        #       the trading activity dataframe as well as other important variables.

        # Load the cryptocurrency market data from the database
        csvPath = os.path.join('..','data',marketSymbol+'_'+startingDate+'_'+endingDate+'.csv')
        exists = os.path.isfile(csvPath)
        if(exists):
            self.data = pd.read_csv(csvPath, header=0, index_col='Timestamp', parse_dates=True)
        else:
            print(csvPath, "does not exist")

        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      self.data['s2f'][0:stateLength].tolist(),  # reduce state
                      [0]]
        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1
        self.name = name

    def reset(self):
        # GOAL: Perform a soft reset of the trading environment.

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        self.data

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist(),
                      self.data['s2f'][0:self.stateLength].tolist(),  # reduce state
                      [0]]
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    def computeLowerBound(self, cash, numberOfShares, price):
        # GOAL: Compute the lower bound of the complete RL action space, i.e. the minimum number of share to trade.

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound

    def rounding(self, shares):
        """
        GOAL: Rounds down the given number of shares to the nearest 0.01.

        INPUTS: - shares to be rounded

        OUTPUTS: rounded share
        """
        # return math.floor(shares) * 100) / 100
        return math.floor(shares)

    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).

        INPUTS: - action: Trading decision (1 = long, 0 = short).

        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Stting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False

        price = self.data['Close'][t]

        # Uniformly distributed entry price (for testing consistency, it is commented)
        # np.random.seed = 3359
        # price = np.random.uniform(self.data['Low'][t], self.data['High'][t])


        # CASE 1: LONG POSITION
        if(action == 1):
            self.data['Position'][t] = 1
            # Case a: Long -> Long
            if(self.data['Position'][t - 1] == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * price
            # Case b: No position -> Long
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = self.rounding(self.data['Cash'][t - 1]/(price * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * price * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * price
                self.data['Action'][t] = 1
            # Case c: Short -> Long
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * price * (1 + self.transactionCosts)
                self.numberOfShares = self.rounding(self.data['Cash'][t]/(price * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * price * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * price
                self.data['Action'][t] = 1

        # CASE 2: SHORT POSITION
        elif(action == 0):
            self.data['Position'][t] = -1
            # Case a: Short -> Short
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * price
                else:
                    numberOfSharesToBuy = min(self.rounding(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * price * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * price
                    customReward = True
            # Case b: No position -> Short
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = self.rounding(self.data['Cash'][t - 1]/(price * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * price * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * price
                self.data['Action'][t] = -1
            # Case c: Long -> Short
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * price * (1 - self.transactionCosts)
                self.numberOfShares = self.rounding(self.data['Cash'][t]/(price * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * price * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * price
                self.data['Action'][t] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['Returns'][t]
        else:
            self.reward = (self.data['Close'][t-1] - price)/self.data['Close'][t-1]

        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      self.data['s2f'][self.t - self.stateLength : self.t].tolist(), # reduce state
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1

        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        customReward = False
        if(otherAction == 1):
            otherPosition = 1
            if(self.data['Position'][t - 1] == 1):
                otherCash = self.data['Cash'][t - 1]
                otherHoldings = numberOfShares * price
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = self.rounding(self.data['Cash'][t - 1]/(price * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] - numberOfShares * price * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * price
            else:
                otherCash = self.data['Cash'][t - 1] - numberOfShares * price * (1 + self.transactionCosts)
                numberOfShares = self.rounding(otherCash/(price * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * price * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * price
        else:
            otherPosition = -1
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'][t - 1]
                    otherHoldings =  - numberOfShares * price
                else:
                    numberOfSharesToBuy = min(self.rounding(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * price * (1 + self.transactionCosts)
                    otherHoldings =  - numberOfShares * price
                    customReward = True
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = self.rounding(self.data['Cash'][t - 1]/(price * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] + numberOfShares * price * (1 - self.transactionCosts)
                otherHoldings = - numberOfShares * price
            else:
                otherCash = self.data['Cash'][t - 1] + numberOfShares * price * (1 - self.transactionCosts)
                numberOfShares = self.rounding(otherCash/(price * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * price * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * price
        otherMoney = otherHoldings + otherCash
        if not customReward:
            otherReward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
        else:
            otherReward = (self.data['Close'][t-1] - price)/self.data['Close'][t-1]
        otherState = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      self.data['s2f'][self.t - self.stateLength : self.t].tolist(), # reduce state
                      [otherPosition]]
        self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.

              No animation

        INPUTS: /

        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        plt.savefig(os.path.join('Figures', self.name+'_Rendering.png'))

    def animate(self, i):
        self.plotClose.append(self.data['Close'][len(self.plotClose)])
        self.plotAction.append(self.data['Action'][len(self.plotAction)])
        self.plotMoney.append(self.data['Money'][len(self.plotMoney)])
        plotClose = pd.Series(self.plotClose)
        plotAction = pd.Series(self.plotAction)
        plotMoney = pd.Series(self.plotMoney)

        # Plot the first graph -> Evolution of the stock market price
        plotClose.plot(ax=self.ax1, color='blue', lw=2)
        self.ax1.plot(plotAction.loc[plotAction == 1.0].index,
                 plotClose[plotAction == 1.0],
                 '^', markersize=5, color='green')
        self.ax1.plot(plotAction.loc[plotAction == -1.0].index,
                 plotClose[plotAction == -1.0],
                 'v', markersize=5, color='red')

        # Plot the second graph -> Evolution of the trading capital
        plotMoney.plot(ax=self.ax2, color='blue', lw=2)
        self.ax2.plot(plotAction.loc[plotAction == 1.0].index,
                 plotMoney[plotAction == 1.0],
                 '^', markersize=5, color='green')
        self.ax2.plot(plotAction.loc[plotAction == -1.0].index,
                 plotMoney[plotAction == -1.0],
                 'v', markersize=5, color='red')

    def render_and_animate(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.

              With animation.

        INPUTS: /

        OUTPUTS: /
        """
        writergif = animation.PillowWriter(fps=30)

        # Set the Matplotlib figure and subplots
        self.fig = plt.figure(figsize=(10, 8))
        self.plotClose = []
        self.plotAction = []
        self.plotMoney = []
        self.ax1 = self.fig.add_subplot(211, ylabel='Price', xlabel='Time')
        self.ax2 = self.fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=self.ax1)
        # Generation of the two legends and plotting
        self.ax1.legend(["Price", "Long",  "Short"])
        self.ax2.legend(["Capital", "Long", "Short"])

        ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.data)-1, interval=200, repeat_delay=5000)
        ani.save(os.path.join('Figures', self.name+'_Rendering.gif'), writer=writergif)

class TradingEnv_cont(gym.Env):
    # GOAL: Implement a custom trading environment compatible with OpenAI Gym.

    def __init__(self, marketSymbol, startingDate, endingDate, money=0, name="", stateLength=30, transactionCosts=0, rewardMode = 'default'):
        # GOAL: Object constructor initializing the trading environment by setting up
        #       the trading activity dataframe as well as other important variables.

        # Load the cryptocurrency market data from the database
        csvPath = os.path.join('..','data',marketSymbol+'_'+startingDate+'_'+endingDate+'.csv')
        exists = os.path.isfile(csvPath)
        if(exists):
            self.data = pd.read_csv(csvPath, header=0, index_col='Timestamp', parse_dates=True)
        else:
            print(csvPath, "does not exist")

        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['ActionType'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        self.data['AssetPct'] = 0. # AssetPct%: % of net value in asset

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      self.data['s2f'][0:stateLength].tolist(),  # reduce state
                      self.data['AssetPct'][0:stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.rewardMode = rewardMode
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts

        # caps the maximum price variation the agent can endure under short position
        # so that the agent is always able to have enough cash to pay back the loss from shorting.
        self.epsilon = 0.1

        self.name = name


    def reset(self):
        # GOAL: Perform a soft reset of the trading environment.

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['ActionType'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        self.data['AssetPct'] = 0.
        self.data

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist(),
                      self.data['s2f'][0:self.stateLength].tolist(),  # reduce state
                      self.data['AssetPct'][0:self.stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state


    def computeLowerBound(self, cash, numberOfShares, price):
        # GOAL: Compute the lower bound of the complete RL action space, i.e. the minimum number of share to trade.
        # Epsilon: Max. tolerable upward volatility before closing a short trade

        # Computation of the RL action lower bound
        # numberOfShares * price = holdings owed (negative)
        # Ensure (- holdings owed - cash) > 0
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound

    def computeSharpeRatio(self, returnsDf, riskFreeRate=0):
        """
        GOAL: Compute the Sharpe Ratio of the trading activity, which is one of
              the most suited performance indicator as it balances the brute
              performance and the risk associated with a trading activity.

        INPUTS:     - riskFreeRate: Return of an investment with a risk null.

        OUTPUTS:    - sharpeRatio: Sharpe Ratio performance indicator.
        """
        # returnsDf = self.data['Returns']

        # Compute the expected return
        expectedReturn = returnsDf[:self.t].mean()

        # Compute the returns volatility
        volatility = returnsDf[:self.t].std()

        # Compute the Sharpe Ratio (252 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            # Adjust by a sqrt of T for fair comparison across different time intervals
            sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate) /volatility
        else:
            sharpeRatio = 0
        return sharpeRatio

    def computeReward(self, customReward, otheraction = False, otherMoney = 0):
        """
        GOAL: Compute the reward of a step under multiple modes.

        INPUTS:     - self.rewardMode:
                                    'default' daily return as Reward
                                    'delay'   daily return asjusted multiplied with an increasing factor
                                    'sharpe'  Sharpe Ratio as Reward
                    - customReward:
                    - otherMoney:   self.data['Money'][t] or otherMoney

        OUTPUTS:    - reward
        """
        t = self.t

        if self.rewardMode == 'delay':
            delay_coeff = t / self.data.shape[0]  # t elapsed / total t
        else:
            delay_coeff = 1  # default multiplier

        # normal action
        if not otheraction:
            if self.rewardMode == 'sharpe':
                reward = self.computeSharpeRatio(self.data['Returns'])
            else:
                if not customReward:
                    reward = self.data['Returns'][t] * delay_coeff
                else:
                    reward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1] * delay_coeff
        # other action
        else:
            t -= 1  # undo the changes in step locally
            if self.rewardMode == 'sharpe':
                returnsCopy = self.data['Returns'].copy()  # clone returns
                returnsCopy[t] = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
                reward = self.computeSharpeRatio(returnsCopy)
            else:
                if not customReward:
                    reward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1] * delay_coeff
                else:
                    reward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1] * delay_coeff

        return reward

    def takeAction1(self,action, other = False):

        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False
        actionList = [-10,0,10]# [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        actionType = actionList[action]
        amount = abs(actionType) / 10  # the amount to be bought/sold

        if actionType not in actionList:
            raise SystemExit(f"Prohibited action! Action (given {actionType}) not correct.")

        if not other:
            self.data['ActionType'][t] = actionType

        if self.data['Money'][t - 1] <= 0 or self.data['Cash'][t - 1] < 0 :
            actionType = 0
        # CASE 1: LONG POSITION
        if(actionType > 0):
            newPosition = 1
            # Case a: Long -> Long
            if(self.data['Position'][t - 1] == 1):
                actionType = 0
            # Case b: No position -> Long
            elif(self.data['Position'][t - 1] == 0):
                newNumberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1
            # Case c: Short -> Long not allowed
            else:
                self.data['Position'][t] = self.data['Position'][t - 1]
                actionType = 0

        # CASE 2: SHORT POSITION
        elif(actionType < 0):
            self.data['Position'][t] = -1
            # Case a: Short -> Short
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                    customReward = True
            # Case b: No position -> Short
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
            # Case c: Long -> Short not allowed
            else:
                self.data['Position'][t] = self.data['Position'][t - 1]
                actionType = 0

        if (actionType == 0):
            self.data['Action'][t] = 0
            self.data['Cash'][t] = self.data['Cash'][t - 1]
            self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]



        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['AssetPct'][t] = self.data['Holdings'][t] / self.data['Money'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]

        # self.data['Action'][t] = actionType
        # Set the RL reward returned to the trading agent
        self.reward = self.computeReward(customReward, otheraction = False)

        # go bust
        if self.data['Money'][t] <= 0 or self.data['Cash'][t] < 0 :
            self.reward = -1
        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      self.data['s2f'][self.t - self.stateLength : self.t].tolist(), # reduce state
                      self.data['AssetPct'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1
        return

    def takeAction(self, action, other=False):
        # Setting of some local variables
        if not other:
            t = self.t
        else:
            t = self.t - 1
        numberOfShares = self.numberOfShares
        newNumberOfShares = numberOfShares
        customReward = False
        actionList = [-10, 0, 10]  # [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        actionType = actionList[action]
        amount = abs(actionType) / 10  # the amount to be bought/sold

        if not other:
            self.data['ActionType'][t] = actionType
        else:
            actionType = actionList[- 1 - action] # opposite action

        # go bust, no action allowed
        if self.data['Money'][t - 1] <= 0 or self.data['Cash'][t - 1] < 0:
            actionType = 0

        # CASE 1: Buy Action
        if (actionType > 0):
            if(self.data['Position'][t - 1] == -1):
                # recover % of the short position
                newShares = abs(numberOfShares * amount)
            else:
                maxShareAmount = self.data['Cash'][t - 1] / \
                    (self.data['Close'][t] * (1 + self.transactionCosts))
                newShares = abs(maxShareAmount * amount)

            if math.floor(newShares * 100) == 0:  # Nullify trades with size < 0.01
                # actionType = 0
                newAction = 0
                newCash = self.data['Cash'][t - 1]
                newHoldings = newNumberOfShares * self.data['Close'][t]
            else:
                newCash = self.data['Cash'][t - 1] - newShares * \
                    (self.data['Close'][t] * (1 + self.transactionCosts))
                newNumberOfShares += newShares
                newHoldings = newNumberOfShares * self.data['Close'][t]
                newAction = 1

        # CASE 2: Sell Action
        elif (actionType < 0):
            if (self.data['Position'][t - 1] == -1):  # short => short
                lowerBound = self.computeLowerBound(
                    self.data['Cash'][t - 1], numberOfShares, self.data['Close'][t - 1])
                if lowerBound <= 0:  # continue short
                    maxShareAmount = self.data['Cash'][t - 1] / (1 - self.data['AssetPct'][t - 1]) / (
                        self.data['Close'][t] * (1 + self.transactionCosts))
                    newShares = abs(maxShareAmount * amount)
                else:  # Buy back
                    newShares = - min(lowerBound, abs(newNumberOfShares))
                    customReward = True
                # nullify short that goes over x1 leverage
                if self.data['AssetPct'][t - 1] <= -0.9:
                    newShares = 0
            elif (self.data['Position'][t - 1] == 1):
                # recover % of the long position
                newShares = abs(numberOfShares * amount)
            else:
                maxShareAmount = self.data['Cash'][t - 1] / \
                    (self.data['Close'][t] * (1 + self.transactionCosts))
                newShares = abs(maxShareAmount * amount)

            if math.floor(newShares * 100) == 0:  # Nullify trades with size < 0.01
                # actionType = 0
                newAction = 0
                newCash = self.data['Cash'][t - 1]
                newHoldings = newNumberOfShares * self.data['Close'][t]
            else:
                # TODO: add safety measures to prevent overshorting
                newCash = self.data['Cash'][t - 1] + newShares * \
                    (self.data['Close'][t] * (1 + self.transactionCosts))
                newNumberOfShares -= newShares
                newHoldings = newNumberOfShares * self.data['Close'][t]
                newAction = -1
                if customReward:  # forced buy back
                    newAction = 1

        # CASE 3: Skip Action
        elif (actionType == 0):
            newAction = 0
            newCash = self.data['Cash'][t - 1]
            newHoldings = newNumberOfShares * self.data['Close'][t]

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit(
                f"Prohibited action! Action (given {actionType}) invalid.")

        # if (round(newNumberOfShares, 3) == 0):  # treat |no. of share| < 0.01 as 0
        if (math.floor(newNumberOfShares * 100) == 0):  # treat |no. of share| < 0.01 as 0
            newPosition = 0
        elif (newNumberOfShares > 0):
            newPosition = 1
        else:
            newPosition = -1

        # Update the total amount of money owned by the agent, as well as the return generated
        newMoney = newHoldings + newCash
        try:
            newAssetPct = newHoldings / newMoney
        except:
            print(newHoldings, newMoney)
        newReturns = (newMoney - self.data['Money']
                      [t - 1]) / self.data['Money'][t - 1]

        if not other:
            self.numberOfShares = newNumberOfShares
            self.data['Cash'][t] = newCash
            self.data['Action'][t] = newAction
            self.data['Position'][t] = newPosition
            self.data['Holdings'][t] = newHoldings
            self.data['Money'][t] = newMoney
            self.data['AssetPct'][t] = newAssetPct
            self.data['Returns'][t] = newReturns


        # self.data['Action'][t] = actionType
        # Set the RL reward returned to the trading agent
        if not other:
            reward = self.computeReward(customReward, otheraction= False)
        else:
            reward = self.computeReward(customReward, otheraction = True, otherMoney = newMoney)

        # go bust
        if newMoney <= 0 or newCash < 0:
            reward = -1
        # Transition to the next trading time step
        if not other:
            self.t = self.t + 1


        newState = [self.data['Close'][self.t - self.stateLength: self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength: self.t].tolist(),
                      self.data['High'][self.t -
                                        self.stateLength: self.t].tolist(),
                      self.data['Volume'][self.t -
                                          self.stateLength: self.t].tolist(),
                      # reduce state
                      self.data['s2f'][self.t - self.stateLength: self.t].tolist(),
                      self.data['AssetPct'][self.t - \
                                            self.stateLength: self.t].tolist(),
                      [newPosition]]
        if not other:
            if(self.t == self.data.shape[0]):  # reached the end of the time series
                done = 1
            else:
                done = 0
        else:
            done = 0

        return newState, reward, done


    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).

        INPUTS: - action: Trading decision [- 10, 10]

        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        self.state, self.reward, self.done = self.takeAction(action, other = False)
        otherState, otherReward, _ = self.takeAction(action, other = True)
        self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}
        return self.state, self.reward, self.done, self.info


    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.

        INPUTS: /

        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        plt.savefig(os.path.join('Figures', self.name+'_Rendering.png'))
