import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import keras

import numpy as np
import random
from simpleLogger import SimpleLogger
from collections import deque

from state import State
from config import LOGGING, LEARNING_RATE, PLACEHOLDER_GAME_BOARD, BATCH_SIZE, COUNTER, EPOCHS, NUM_BATCHES, DISCOUNT

logger = SimpleLogger()
MODEL_NAME = "../models/model"
EPSILON_COUNTER_EPOCH = 50
MIN_EPSILON = 0.003

class Agent:
    def __init__(
        self, 
        n_neurons, 
        epsilon, 
        q_table, 
        actions, 
        action_space_string, 
        load_model: bool = False, 
        num_actions=None, 
        board_shape=None, 
        epsilon_decay: float=0.9995
    ):
        self.n_neurons           = n_neurons
        self.epsilon             = epsilon
        self.q_table             = q_table
        self.memory              = deque(maxlen=500)
        self.actions             = actions
        self.current_action      = None
        self.current_state       = None
        self.discount_factor     = DISCOUNT
        self.action_space_string = action_space_string
        self.counter             = 0
        self.counter_weight_log  = 0
        self.counter_epsilon     = 0
        self.num_actions         = num_actions
        self.board_shape         = board_shape
        self.epsilon_decay       = epsilon_decay

        if load_model:
            self.model = self._load_model()
        else:
            self.model = self._init_model()
            
        logger.log(f"actions in __init__: {self.actions}")
        #self.train_on_basic_scenarios()


    def train(self, state: State, action, next_state: State, reward):
        state_array = state.convert_to_array()
        next_state_array = next_state.convert_to_array()
        
        self.memory.append((state_array, action, reward, next_state_array))
        
        # # Train on this experience
        # self.model.fit(
        #     state_array.reshape(1, -1),
        #     current_q_values,
        #     epochs=1,
        #     verbose=0
        # )
        
        if len(self.memory) >= BATCH_SIZE and self.counter % COUNTER == 0 :
            
            for _ in range(NUM_BATCHES):
                #logger.log(f"processing batch from memory, current len: {len(self.memory)}")
                minibatch = random.sample(self.memory,BATCH_SIZE)
                states, actions, rewards, next_states = zip(*minibatch)
                
                states = np.array(states)
                next_states = np.array(next_states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                    
                # Predict Q-values for current and next states
                current_qs = self.model.predict(states, verbose=0)
                next_qs = self.model.predict(next_states, verbose=0)
                
                targets = rewards + self.discount_factor * np.max(next_qs, axis=1)
                
                # Update targets for actions taken
                for i in range(BATCH_SIZE):
                    current_qs[i][actions[i]] = targets[i]
                
                # Train on batch
                self.model.fit(
                    states, 
                    current_qs,
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    verbose=0)
        
        self.counter += 1
        if self.counter == COUNTER:
            self.counter = 0
            self._save_model()


    def epsilon_greedy_policy(self, 
                              state: State):
        state_array = state.convert_to_array()
        if random.random() <= self.epsilon:
            return_val = np.random.choice(self.actions)
            if LOGGING:
                logger.log(f"randomly chosen return val {return_val}")
                logger.log(f"current state{state}")
            self.counter_epsilon += 1
            if self.counter_epsilon == EPSILON_COUNTER_EPOCH:
                
                if LOGGING:
                    logger.log(f"current epsilon={self.epsilon}, counter={self.counter_epsilon}")
                self.counter_epsilon = 0
                if self.epsilon >= MIN_EPSILON:
                    self.epsilon *= self.epsilon_decay
                    
            logger.log(f"randomly chosen: {return_val}")
            return return_val
        else:
            q_values = self.model.predict(state_array.reshape(1, -1), verbose=0)[0]
            
            valid_q_values = q_values[:(len(self.actions))]
            return_val = np.argmax(valid_q_values) -32    # offset for the line above
            
            if LOGGING:
                logger.log(f"q_values: {q_values}")
                logger.log(f"return val IN Q TABLE {return_val}")
            logger.log(f"q_val: {return_val}")
            return return_val


    def predict(self, state):
        state_values = state.get_values()
        q_values = self.model.predict(np.array([state_values]), verbose=0)[0]
        q_table = {action: q_values[i] for i, action in enumerate(self.action_space_string)}
        logger.log(f"in prediction: {q_table}")
        return q_table
    
    
    def train_on_basic_scenarios(self):
        """Pre-train on ideal scenarios with meaningful transitions"""
        basic_scenarios = [
            # Perfect Tetris clear scenario
            (
                # Current state: well set up for Tetris
                {
                    'lines_cleared': 0,
                    'holes': 0,
                    'height': 15,
                    'bumpiness': 4,
                    'wells': 1,
                    'column_heights': [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 11, 15, 15, 15],
                    'row_transitions': 4,
                    'column_transitions': 4,
                    'landing_height': 15
                },
                # Next state: after Tetris clear
                {
                    'lines_cleared': 4,
                    'holes': 0,
                    'height': 11,
                    'bumpiness': 0,
                    'wells': 0,
                    'column_heights': [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
                    'row_transitions': 0,
                    'column_transitions': 0,
                    'landing_height': 11
                },
                800  # High reward for Tetris
            ),
            
            # Single line clear
            (
                {
                    'lines_cleared': 0,
                    'holes': 1,
                    'height': 8,
                    'bumpiness': 2,
                    'wells': 0,
                    'column_heights': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                    'row_transitions': 2,
                    'column_transitions': 2,
                    'landing_height': 8
                },
                {
                    'lines_cleared': 1,
                    'holes': 0,
                    'height': 7,
                    'bumpiness': 1,
                    'wells': 0,
                    'column_heights': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                    'row_transitions': 1,
                    'column_transitions': 1,
                    'landing_height': 7
                },
                100  # Good reward for line clear
            ),
            
            # Creating well for future Tetris
            (
                {
                    'lines_cleared': 0,
                    'holes': 0,
                    'height': 10,
                    'bumpiness': 8,
                    'wells': 0,
                    'column_heights': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                    'row_transitions': 0,
                    'column_transitions': 0,
                    'landing_height': 10
                },
                {
                    'lines_cleared': 0,
                    'holes': 0,
                    'height': 12,
                    'bumpiness': 4,
                    'wells': 1,
                    'column_heights': [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 8, 12, 12, 12],
                    'row_transitions': 4,
                    'column_transitions': 4,
                    'landing_height': 12
                },
                50  # Moderate reward for setting up Tetris opportunity
            ),
            
            # Bad scenario - holes and high bumpiness
            (
                {
                    'lines_cleared': 0,
                    'holes': 5,
                    'height': 15,
                    'bumpiness': 10,
                    'wells': 2,
                    'column_heights': [15, 12, 15, 10, 15, 11, 15, 13, 15, 14, 15, 12, 15, 13],
                    'row_transitions': 20,
                    'column_transitions': 15,
                    'landing_height': 15
                },
                {
                    'lines_cleared': 0,
                    'holes': 8,
                    'height': 17,
                    'bumpiness': 12,
                    'wells': 3,
                    'column_heights': [17, 13, 17, 11, 17, 12, 17, 14, 17, 15, 17, 13, 17, 14],
                    'row_transitions': 24,
                    'column_transitions': 18,
                    'landing_height': 17
                },
                -50  # Penalty for creating holes and increasing bumpiness
            ),
            
            # Very bad scenario - near death
            (
                {
                    'lines_cleared': 0,
                    'holes': 10,
                    'height': 20,
                    'bumpiness': 15,
                    'wells': 4,
                    'column_heights': [20, 18, 20, 16, 20, 17, 20, 19, 20, 18, 20, 17, 20, 19],
                    'row_transitions': 30,
                    'column_transitions': 25,
                    'landing_height': 20
                },
                {
                    'lines_cleared': 0,
                    'holes': 12,
                    'height': 23,
                    'bumpiness': 18,
                    'wells': 5,
                    'column_heights': [23, 20, 23, 18, 23, 19, 23, 21, 23, 20, 23, 19, 23, 21],
                    'row_transitions': 35,
                    'column_transitions': 30,
                    'landing_height': 23
                },
                -100  # Large penalty for dangerous height
            )
        ]

        def set_state_values(state, values):
            state.board = PLACEHOLDER_GAME_BOARD
            
            """Helper to set all state values"""
            state.lines_cleared = values['lines_cleared']
            state.holes = values['holes']
            state.height = values['height']
            state.bumpiness = values['bumpiness']
            state.wells = values['wells']
            state.column_heights = values['column_heights']
            state.row_transitions = values['row_transitions']
            state.column_transitions = values['column_transitions']
            state.landing_height = values['landing_height']
        
        for _ in range(10):  # Pre-training iterations
            for current_values, next_values, reward in basic_scenarios:
                # Create states with proper transitions
                current_state = State(PLACEHOLDER_GAME_BOARD, 0)
                next_state = State(PLACEHOLDER_GAME_BOARD, 0)
                
                # Set all values for both states
                set_state_values(current_state, current_values)
                set_state_values(next_state, next_values)
                
                # Train on these scenarios
                self.train(current_state, 0, next_state, reward)


    def _init_model(self):
        n_output = 52
        n_input = 5  # Use more input features
        input_shape = (n_input,)
        model = keras.models.Sequential([
            keras.layers.Dense(128, activation="relu", input_shape=input_shape, kernel_initializer='he_uniform'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation="relu", kernel_initializer='he_uniform'),
            keras.layers.Dense(256, activation="relu", kernel_initializer='he_uniform'),
            keras.layers.Dense(n_output, activation="linear", kernel_initializer='glorot_uniform')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='huber'
        )
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
