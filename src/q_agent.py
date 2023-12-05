import keras
import numpy as np
import random
from simpleLogger import SimpleLogger

logger = SimpleLogger()
MODEL_NAME = "model.h5"

class Agent:
    def __init__(self, n_neurons, epsilon, q_table, load_model: bool = False):
        self.n_neurons = n_neurons
        self.epsilon = epsilon
        self.q_table = q_table

        if load_model:
            # load keras model from file
            self.model = self.load_model()

        else:
            # build a new keras sequential model
            self.model = self.init_model()


    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.choice(["rotate", "left", "right"])
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = self.predict(state)
            return max(q_values, key=q_values.get)

    def predict(self, state):
        # Assuming state is a list of features [holes, lines_cleared, bumpiness, piece_type, height]
        q_values = self.model.predict(np.array([state]))[0]
        return {action: q_values[i] for i, action in enumerate(["rotate", "left", "right"])}


    def init_model(self):
        # temp: magic numbers
        n_output = 3  # rotate, left, right
        input_shape = [5]  # holes, lines cleared, bumpiness, piece_type, height

        model = keras.models.Sequential()

        # input layer with 5 nodes
        model.add(keras.layers.Dense(input_shape[0], activation="relu", input_shape=input_shape))
        model.add(keras.layers.Dense(self.n_neurons, activation="relu"))  # one hidden layer
        model.add(keras.layers.Dense(n_output))  # for output (rotate, left, right)

        self.log_model_summary(model, logger)
    
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model


    def save_model(self):
        """
        saves keras model as a file.
        """
        self.model.save(MODEL_NAME)


    def load_model(self):
        """
        loads a saved keras model.
        """
        return keras.models.load_model(MODEL_NAME)


    def log_model_summary(self, model: keras.Sequential, logger):

        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        summary_str = "\n".join(summary_str)

        logger.log(summary_str+"\n\n")



def testing():
    n_neurons = 30
    epsilon = 0.3
    q_table = {}

    agent = Agent(n_neurons, epsilon, q_table)

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
