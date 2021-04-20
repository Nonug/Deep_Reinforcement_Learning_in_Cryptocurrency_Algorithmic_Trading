#Implementing trading environment
import os
import gym
import math
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
pd.options.mode.chained_assignment = None

class TradingEnv(gym.Env):
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
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      self.data['s2f'][0:stateLength].tolist(),  # new feature
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
                      self.data['s2f'][0:self.stateLength].tolist(),  # new features
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

        # Stting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False
        actionList = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        actionType = actionList[action]
        amount = abs(actionType) / 10  # the amount to be bought/sold



        if actionType not in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
            raise SystemExit(f"Prohibited action! Action (given {actionType}) not correct.")

        # CASE 1: Buy Action
        if (actionType > 0):
            if(self.data['Position'][t - 1] == -1):
                newShares = round(numberOfShares * amount, 3) # recover % of the short position
            else:
                maxShareAmount = self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts))
                newShares = round(maxShareAmount * amount, 3)

            if newShares == 0:  # Nullify trades with size < 0.005
                # actionType = 0
                self.data['Action'][t] = 0
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - newShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares += newShares
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1

        # CASE 2: Sell Action
        elif (actionType < 0):
            if (self.data['Position'][t - 1] == 1):
                newShares = round(numberOfShares * amount, 3) # recover % of the long position
            elif (self.data['Position'][t - 1] == -1):
                newShares = 0
            else:
                maxShareAmount = self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts))
                newShares = round(maxShareAmount * amount, 3)

            if newShares == 0:  # Nullify trades with size < 0.005
                # actionType = 0
                self.data['Action'][t] = 0
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            else:
                # TODO: add safety measures to prevent overshorting
                self.data['Cash'][t] = self.data['Cash'][t - 1] + newShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares -= newShares
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1

        # CASE 3: Skip Action
        elif (actionType == 0):
            self.data['Action'][t] = 0
            self.data['Cash'][t] = self.data['Cash'][t - 1]
            self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]


        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit(f"Prohibited action! Action (given {actionType}) not within [-10, 10].")

        # if (round(self.numberOfShares, 3) == 0):  # treat |no. of share| < 0.0005 as 0
        if (self.numberOfShares == 0):  # treat |no. of share| < 0.0005 as 0
            self.data['Position'][t] = 0
        elif (self.numberOfShares > 0):
            self.data['Position'][t] = 1
        else:
            self.data['Position'][t] = -1

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]

        # self.data['Action'][t] = actionType
        # Set the RL reward returned to the trading agent
        self.reward = self.computeReward(customReward, otheraction = False)

        # go bust
        # if self.data['Money'][t] <= 0:
        #     self.done = 1
        #     self.reward = -1000
        # TODO: Add holdings as a state?
        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      self.data['s2f'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1

        # # Same reasoning with the other action (exploration trick)
        # otherAction = int(not bool(action))
        # customReward = False
        # if(otherAction == 1):
        #     otherPosition = 1
        #     if(self.data['Position'][t - 1] == 1):
        #         otherCash = self.data['Cash'][t - 1]
        #         otherHoldings = numberOfShares * self.data['Close'][t]
        #     elif(self.data['Position'][t - 1] == 0):
        #         numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
        #         otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
        #         otherHoldings = numberOfShares * self.data['Close'][t]
        #     else:
        #         otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
        #         numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
        #         otherCash = otherCash - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
        #         otherHoldings = numberOfShares * self.data['Close'][t]
        # else:
        #     otherPosition = -1
        #     if(self.data['Position'][t - 1] == -1):
        #         lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
        #         if lowerBound <= 0:
        #             otherCash = self.data['Cash'][t - 1]
        #             otherHoldings =  - numberOfShares * self.data['Close'][t]
        #         else:
        #             numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
        #             numberOfShares -= numberOfSharesToBuy
        #             otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
        #             otherHoldings =  - numberOfShares * self.data['Close'][t]
        #             customReward = True
        #     elif(self.data['Position'][t - 1] == 0):
        #         numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
        #         otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
        #         otherHoldings = - numberOfShares * self.data['Close'][t]
        #     else:
        #         otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
        #         numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
        #         otherCash = otherCash + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
        #         otherHoldings = - self.numberOfShares * self.data['Close'][t]
        #
        # otherMoney = otherHoldings + otherCash
        #
        # otherReward = self.computeReward(customReward, otheraction = True, otherMoney = otherMoney)
        # # if not customReward:
        # #     otherReward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1] * delay_coeff
        # # else:
        # #     otherReward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1] * delay_coeff
        #
        # otherState = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
        #               self.data['Low'][self.t - self.stateLength : self.t].tolist(),
        #               self.data['High'][self.t - self.stateLength : self.t].tolist(),
        #               self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
        #               self.data['s2f'][self.t - self.stateLength : self.t].tolist(),
        #               [otherPosition]]
        # self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, 0


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
