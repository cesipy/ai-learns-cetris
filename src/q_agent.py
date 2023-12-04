import keras
import numpy as np
import random

class Agent:
    def __init__(self, number_layers, epsilon, q_table):
        self.number_layers = number_layers
        self.epsilon       = epsilon
        self.q_table       = q_table
        self.model         = self.init_model()


    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            pass
            # return random state 
        else:
            # return best q value for state
            pass


    def predict(self, state): 
        pass


    def init_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(self.number_layers, activation="relu", input_shape=(5,)))
        model.add(keras.layers.Dense(units=2, activation='linear'))  # Assuming 2 actions
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
