import keras
import numpy as np
import random
from simpleLogger import SimpleLogger
from collections import deque

from metadata import State

logger = SimpleLogger()
MODEL_NAME = "../models/model"
EPSILON_COUNTER_EPOCH = 50
MIN_EPSILON = 0.2

class Agent:
    def __init__(self, 
                 n_neurons, 
                 epsilon, 
                 q_table, 
                 actions, 
                 action_space_string, 
                 load_model: bool = False, 
                 num_actions=None, 
                 board_shape=None):
        self.n_neurons           = n_neurons
        self.epsilon             = epsilon
        self.q_table             = q_table
        self.memory              = deque(maxlen=1000)
        self.actions             = actions
        self.current_action      = None
        self.current_state       = None
        self.discount_factor     = 0.9
        self.action_space_string = action_space_string
        self.counter             = 0
        self.counter_weight_log  = 0
        self.counter_epsilon     = 0
        self.num_actions         = num_actions
        self.board_shape         = board_shape

        if load_model:
            self.model = self._load_model()
        else:
            self.model = self._init_model()


    def train(self, 
              state: State, 
              action, 
              next_state, 
              reward):
        next_state_array = state.convert_to_array()
        state_array = next_state.convert_to_array()
        target = reward + self.discount_factor * np.max(self.model.predict(next_state_array.reshape(1, -1), verbose=0))
        target_q_values = self.model.predict(state_array.reshape(1, -1), verbose=0)
        target_q_values[0, action] = (1 - 0.1) * target_q_values[0, action] + 0.1 * target
        self.model.fit(state_array.reshape(1, -1), target_q_values, epochs=1, verbose=0)
        self.counter += 1
        if self.counter == EPSILON_COUNTER_EPOCH:
            self.counter = 0
            self._save_model()


    def epsilon_greedy_policy(self, 
                              state: State):
        state_array = state.convert_to_array()
        if random.random() <= self.epsilon:
            return_val = np.random.choice(self.actions)
            logger.log(f"randomly chosen return val {return_val}")
            logger.log(f"current state{state}")
            self.counter_epsilon += 1
            if self.counter_epsilon == EPSILON_COUNTER_EPOCH:
                logger.log(f"current epsilon={self.epsilon}, counter={self.counter_epsilon}")
                self.counter_epsilon = 0
                if self.epsilon >= MIN_EPSILON:
                    self.epsilon -= 0.0025
            return return_val
        else:
            q_values = self.model.predict(state_array.reshape(1, -1), verbose=0)[0]
            logger.log(f"q_values: {q_values}")
            return_val = np.argmax(q_values)
            logger.log(f"return val IN Q TABLE {return_val}")
            return return_val


    def predict(self, state):
        state_values = state.get_values()
        q_values = self.model.predict(np.array([state_values]), verbose=0)[0]
        q_table = {action: q_values[i] for i, action in enumerate(self.action_space_string)}
        logger.log(f"in prediction: {q_table}")
        return q_table


    def _init_model(self):
        n_output = len(self.actions)
        logger.log(f"n_output: {n_output}")
        n_input = np.prod(self.board_shape)
        input_shape = (n_input,)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=n_input, activation="relu", input_shape=input_shape))
        model.add(keras.layers.Dense(units=self.n_neurons, activation="relu"))
        model.add(keras.layers.Dense(units=self.n_neurons, activation="relu"))
        model.add(keras.layers.Dense(units=n_output, activation="linear"))
        self._log_model_summary(model, logger)
        model.compile(optimizer='adam', loss='mse')
        return model


    def _save_model(self):
        self.model.save(f"{MODEL_NAME}.keras")
        self.counter_weight_log += 1
        if self.counter_weight_log == 50:
            self.counter_weight_log = 0
            for layer in self.model.layers:
                logger.log(layer.get_weights())


    def _load_model(self):
        return keras.models.load_model(f"{MODEL_NAME}.keras")


    def _log_model_summary(self, model: keras.Sequential, logger):
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        summary_str = "\n".join(summary_str)
        logger.log(summary_str + "\n\n")


    def get_epsilon(self):
        return self.epsilon
