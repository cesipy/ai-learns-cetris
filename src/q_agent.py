import keras
import numpy as np
import random
from simpleLogger import SimpleLogger
from collections import deque

logger = SimpleLogger()
MODEL_NAME = "model.h5"

class Agent:

    def __init__(self, n_neurons, epsilon, q_table, actions, load_model: bool = False):
        self.n_neurons = n_neurons
        self.epsilon = epsilon
        self.q_table = q_table
        self.memory  = deque(maxlen=1000)   # deque of quintuple (x,x,x,x,x)
        self.actions = actions

        if load_model:
            # load keras model from file
            self.model = self._load_model()

        else:
            # build a new keras sequential model
            self.model = self._init_model()


    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            #  choose a random action
            value =random.choice(self.actions)   # choose randomly from 'left', 'right' and 'rotate
            logger.log(f"selected action {value}")
            return value
        else:
            #choose the action with the highest q value
            q_values = self.predict(state)
            value = max(q_values, key=q_values.get)
            logger.log(f"selected action {value}")
            return value


    def predict(self, state):
        state_values = state.get_values()
        q_values = self.model.predict(np.array([state_values]))[0]
        logger.log(q_values)
        q_table = {}

        for i, action in enumerate(["rotate", "left", "right"]):
            q_table[action] = q_values[i]
            
        return q_table



    def _init_model(self):
        # temp: magic numbers
        n_output = 3  # rotate, left, right
        n_input  = 5
        input_shape = (5,)  # holes, lines cleared, bumpiness, piece_type, height

        model = keras.models.Sequential()

        # input layer with 5 nodes
        model.add(keras.layers.Dense(units=n_input, activation="relu", input_shape=input_shape))
        model.add(keras.layers.Dense(units=self.n_neurons, activation="relu"))  # one hidden layer
        model.add(keras.layers.Dense(units=n_output))  # for output (rotate, left, right)

        self._log_model_summary(model, logger)
    
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model


    def _save_model(self):
        """
        saves keras model as a file.
        """
        self.model.save(MODEL_NAME)


    def _load_model(self):
        """
        loads a saved keras model.
        """
        return keras.models.load_model(MODEL_NAME)


    def _log_model_summary(self, model: keras.Sequential, logger):
        """
        logs teras model summary to logger file.
        """
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        summary_str = "\n".join(summary_str)

        logger.log(summary_str+"\n\n")


# ----------------------------------- #

def testing():
    n_neurons = 30
    epsilon = 0.3
    q_table = {}
    actions = ["left", "rotate", "right"]

    agent = Agent(n_neurons, epsilon, q_table, actions)

    # temporary tests
    state = [1, 2, 3, 4, 5]  
    action = agent.epsilon_greedy_policy(state)
    logger.log(f"Selected action: {action}")

    q_values = agent.predict(state)
    logger.log(f"Q-values: {q_values}\n")


def main():
    testing()


if __name__ == '__main__':
    main()
