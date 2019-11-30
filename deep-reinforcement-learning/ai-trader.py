import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
from tqdm import tqdm_notebook, tqdm
from collections import deque


class AI_Trader():
    def __init__(self, state_size, action_space=3, model_name="AITrader"):  # Stay, Buy, Sell
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.model_builder()

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(
            units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(
            units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))

        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay


def sigmoid(x):
    return 1/(1+math.exp(-x))


def stocks_price_format(n):
    if n < 0:
        return "-$ {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))


def dataset_loader(stock_name):
    # Complete the dataset loader function
    dataset = data_reader.DataReader(stock_name, data_source="yahoo")

    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]
    close = dataset['Close']

    return close


def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1

    if starting_id >= 0:
        windowed_data = data[starting_id:timestep+1]
    else:
        windowed_data = -starting_id * [data[0]] + list(data[0:timestep+1])

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))

    return np.array([state])


# Load the dataset
stock_name = "AAPL"
data = dataset_loader(stock_name)

# Set hyper parameters
window_size = 10
episodes = 1000

batch_size = 32
data_samples = len(data) - 1

# Define trader model
trader = AI_Trader(window_size)

# Training loop
for episode in range(1, episodes + 1):
    print("Episode: {}/{}".format(episode, episodes))

    state = state_creator(data, 0, window_size + 1)

    total_profit = 0
    trader.inventory = [] # Clean inventory before we start each episode

    for t in tqdm(range(data_samples)): # tqdm creates progress bar visualization
        action = trader.trade(state) # Use trade funciton to get an action
        next_state = state_creator(data, t+1, window_size + 1) # Next state getter
        reward = 0

        if action == 1:  # Buying
            trader.inventory.append(data[t]) # Put stock in inventory
            print("AI Trader bought: ", stocks_price_format(data[t]))

        elif action == 2 and len(trader.inventory) > 0:  # Selling and we already bought stocks
            buy_price = trader.inventory.pop(0) # Remove stock from inventory in order

            reward = max(data[t] - buy_price, 0) # Either 0 reward or the profit
            total_profit += data[t] - buy_price # Update total profit
            print("AI Trader sold: ", stocks_price_format(
                data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))
            
        if t == data_samples - 1: # If this is the last sample, be done
            done = True
        else:
            done = False
        
        trader.memory.append((state, action, reward, next_state, done)) # Put all data in memory

        state = next_state # Change state to next_state to continue iterating

        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("########################")
        
        if len(trader.memory) > batch_size: # If we have more information in our memory than our specified batch, train on this batch
            trader.batch_train(batch_size)
        
    if episode % 10 == 0: # Save this model every ten episodes
        trader.model.save("ai_trader_{}.h5".format(episode))
