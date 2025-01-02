import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import keras

import numpy as np
import random
from simpleLogger import SimpleLogger
from collections import deque

from tetris_expert import TetrisExpert

from state import State
from config import *

logger = SimpleLogger()
MODEL_NAME = "../models/model"


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
        self.memory              = deque(maxlen=1000)
        self.actions             = actions
        self.current_action      = None
        self.current_state       = None
        self.discount_factor     = DISCOUNT
        self.action_space_string = action_space_string
        self.counter             = 0
        self.counter_weight_log  = 0
        self.counter_epsilon     = 0
        self.cunter_tetris_expert = 0
        self.starting_tetris_expert_modulo = COUNTER_TETRIS_EXPERT      # when to do the expert for faster reward discovery
        self.num_actions         = num_actions
        self.board_shape         = board_shape
        self.epsilon_decay       = epsilon_decay
        self.tetris_expert       = TetrisExpert(self.actions)

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
        
        if len(self.memory) >=1000 and self.counter % COUNTER == 0 :
            
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
                max_next_qs = np.max(next_qs, axis=1)
                targets = rewards + (self.discount_factor * max_next_qs)
                
                # Update only the Q values for the actions taken
                target_qs = current_qs.copy()
                for i in range(BATCH_SIZE):
                    target_qs[i][actions[i]] = targets[i]
                
                # Train on batch
                self.model.fit(
                    states,
                    target_qs,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=0
                )
        
        self.counter += 1
        if self.counter == COUNTER:
            self.counter = 0
            self._save_model()


    def epsilon_greedy_policy(self, 
                              state: State):
        state_array = state.convert_to_array()
        if random.random() <= self.epsilon:
            
            if self.cunter_tetris_expert % int(round(self.starting_tetris_expert_modulo)) == 0:
                #logger.log(f"tetris expert with current  modulo rounded {int(round(self.starting_tetris_expert_modulo))}")
                self.cunter_tetris_expert = 0
                return_val = self.tetris_expert.get_best_move(state=state)
                if return_val is None:
                    return_val = np.random.choice(self.actions)
                
                self.starting_tetris_expert_modulo +=0.0
                #logger.log(f"current expert modulo: {self.starting_tetris_expert_modulo}")
            else:
                return_val = np.random.choice(self.actions)
            self.cunter_tetris_expert += 1
            
            
            if LOGGING:
                logger.log(f"randomly chosen return val {return_val}")
                logger.log(f"current state{state}")
        else:
            q_values = self.model.predict(state_array.reshape(1, -1), verbose=0)[0]

            action_q_values = {}
            for i, action in enumerate(self.actions):
                action_q_values[action] = q_values[i]
                
            return_val = max(action_q_values.items(), key=lambda x: x[1])[0]

        self.counter_epsilon += 1
        if self.counter_epsilon == EPSILON_COUNTER_EPOCH:
            
            if LOGGING:
                logger.log(f"current epsilon={self.epsilon}, counter={self.counter_epsilon}")
            self.counter_epsilon = 0
            if self.epsilon >= MIN_EPSILON:
                self.epsilon *= self.epsilon_decay
                

        return return_val


    def predict(self, state):
        state_values = state.get_values()
        q_values = self.model.predict(np.array([state_values]), verbose=0)[0]
        q_table = {action: q_values[i] for i, action in enumerate(self.action_space_string)}
        logger.log(f"in prediction: {q_table}")
        return q_table
    


    def _init_model(self):
        n_output = len(self.actions)
        n_input = 9
        input_shape = (n_input,)
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=input_shape, kernel_initializer='he_uniform'),
            keras.layers.Dense(64, activation="relu", kernel_initializer='he_uniform'),
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
