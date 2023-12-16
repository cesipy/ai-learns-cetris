import keras
import numpy as np
import random
from simpleLogger import SimpleLogger
from collections import deque

logger = SimpleLogger()
MODEL_NAME = "../models/model"

class Agent:

    def __init__(self, n_neurons, epsilon, q_table, actions, load_model: bool = False):
        self.n_neurons = n_neurons
        self.epsilon = epsilon
        self.q_table = q_table
        self.memory  = deque(maxlen=1000)   # deque of quintuple (x,x,x,x,x)
        self.actions = actions
        self.current_action = None
        self.current_state  = None
        self.discount_factor = 0.9 # temp magic number
        
        # only for debugging and print out all weigths od nn
        self.counter = 0

        if load_model:
            # load keras model from file
            self.model = self._load_model()

        else:
            # build a new keras sequential model
            self.model = self._init_model()


    def train(self, state, action, next_state, reward):
        next_state_array = next_state.convert_to_array()
        state_array      = state.convert_to_array()

        target = reward + self.discount_factor * np.max(self.model.predict(next_state_array.reshape(1, -1)))
        target_q_values = self.model.predict(state_array.reshape(1, -1))
        target_q_values[0, action] = target

        self.model.fit(state_array.reshape(1, -1), target_q_values, epochs=1, verbose=0)

        self._save_model()


    def epsilon_greedy_policy(self, state, epsilon=0.5):
        state_array = state.convert_to_array()

        if random.random() <= epsilon:
            return_val = np.random.choice(self.actions)
            logger.log(f"return val {return_val}")
            return return_val
        else:
            q_values = self.predict(state)
            logger.log(f"q_values: {q_values}")
            return_val = np.argmax(list(q_values.values()))
            logger.log(f"return val IN Q TABLE {return_val}")
            return return_val



    def predict(self, state):
        state_values = state.get_values()
        q_values = self.model.predict(np.array([state_values]))[0]
        logger.log(q_values)
        q_table = {}

        action_space = [
            "4right-rotate", "4right", "3right-rotate", "3right", "2right-rotate", "2right", "right-rotate", "right", 
            "nothing", "left-rotate", "left", "2left-rotate", "2left", "3left-rotate", "3left", "4left-rotate", "4left"
            ]


        for i, action in enumerate(action_space):
            q_table[action] = q_values[i]
            
        logger.log(f"in prediction: {q_table}")
        return q_table



    def _init_model(self):
        # temp: magic numbers
        n_output = 17  # rotate, left, right
        n_input  = 5
        input_shape = (5,)  # holes, lines cleared, bumpiness, piece_type, height

        model = keras.models.Sequential()

        # input layer with 5 nodes
        model.add(keras.layers.Dense(units=n_input, activation="relu", input_shape=input_shape))
        model.add(keras.layers.Dense(units=self.n_neurons, activation="relu"))  # one hidden layer
        model.add(keras.layers.Dense(units=n_output))  # for output (rotate, left, right)

        self._log_model_summary(model, logger)
    
        model.compile(optimizer='adam', loss='mse')

        return model


    def _save_model(self):
        """
        saves keras model as a file.
        """
        #self.model.save(MODEL_NAME)
        self.model.save(f"{MODEL_NAME}.keras")

        self.counter += 1
        if self.counter == 50:
            self.counter = 0

            for layer in self.model.layers:
                logger.log(layer.get_weights())


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
    
    agent._save_model()


def main():
    testing()


if __name__ == '__main__':
    main()
