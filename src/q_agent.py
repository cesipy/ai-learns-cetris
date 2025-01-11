import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Empty string to disable CUDA


import torch
#torch.set_num_threads(1)  # This helps prevent multiprocessing issues
from nn_model import CNN, nn

import numpy as np
import random
from simpleLogger import SimpleLogger
from collections import deque

from tetris_expert import TetrisExpert

from state import State
from config import *

logger = SimpleLogger()
MODEL_NAME = "../models/model"

device = torch.device("cpu")
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
        self.memory              = deque(maxlen=70000)
        self.actions             = actions
        self.current_action      = None
        self.current_state       = None
        self.discount_factor     = DISCOUNT
        self.action_space_string = action_space_string
        self.counter             = 0
        self.counter_weight_log  = 0
        self.counter_epsilon     = 0
        self.cunter_tetris_expert = 0
        self.starting_tetris_expert_modulo = COUNTER_TETRIS_EXPERT      # when to use the expert for faster reward discovery
        self.num_actions         = num_actions
        self.board_shape         = board_shape
        self.epsilon_decay       = epsilon_decay
        self.tetris_expert       = TetrisExpert(self.actions)

        self.model = CNN(num_actions=num_actions).to(device)
        self.target_model = CNN(num_actions=num_actions).to(device)
        self.target_update_counter = 0
        self.target_update_frequency = 500
            
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # Copy weights to target model
        self.target_model.load_state_dict(self.model.state_dict())
            
        if load_model:
            self.model = self._load_model()
            
            
        logger.log(f"actions in __init__: {self.actions}")
        #self.train_on_basic_scenarios()


    def train(self, state: State, action, next_state: State, reward):
        state_array = state.convert_to_array()
        next_state_array = next_state.convert_to_array()
        
        # convert to tensors 
        state_array = torch.from_numpy(state.convert_to_array()).float()
        next_state_array = torch.from_numpy(next_state.convert_to_array()).float()
        
        # action space has negative values -> just workaround for this
        norm_action = State.normalize_action(action)
        
        self.memory.append((state_array, norm_action, reward, next_state_array))
        
        if len(self.memory) >=1500 and self.counter % COUNTER == 0 :
            
            for _ in range(NUM_BATCHES):
                #logger.log(f"processing batch from memory, current len: {len(self.memory)}")
                self.train_batch()
                
                
                
        # sync the target and normal models.
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_frequency: 
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
        self.counter += 1
        if self.counter == COUNTER:
            self.counter = 0
            self._save_model()


    def epsilon_greedy_policy(self, state: State):
        state_array = torch.from_numpy(state.convert_to_array()).float()
        
        # exploration
        if random.random() <= self.epsilon:
            if self.cunter_tetris_expert % int(round(self.starting_tetris_expert_modulo)) == 0:
                self.cunter_tetris_expert = 0
                return_val = self.tetris_expert.get_best_move(state=state)
                if return_val is None:
                    return_val = np.random.choice(self.actions)
                self.starting_tetris_expert_modulo += 0.0
            else:
                return_val = np.random.choice(self.actions)
            self.cunter_tetris_expert += 1
            
        # exploitation
        else:
            with torch.no_grad():  # Don't track gradients for prediction
                q_values = self.model(state_array.unsqueeze(0))[0]
                # normalized actions for lookup, denorm for return
                action_q_values = {State.denormalize_action(i): q_value.item() 
                                for i, q_value in enumerate(q_values)}
                return_val = max(action_q_values.items(), key=lambda x: x[1])[0]
                
                #logger.log(f"return_val: {return_val}")

        # Epsilon decay
        self.counter_epsilon += 1
        if self.counter_epsilon == EPSILON_COUNTER_EPOCH:
            if self.epsilon >= MIN_EPSILON:
                self.epsilon *= self.epsilon_decay
            self.counter_epsilon = 0

        return return_val


    def predict(self, state):
        state_values = state.get_values()
        q_values = self.model.predict(np.array([state_values]), verbose=0)[0]
        q_table = {action: q_values[i] for i, action in enumerate(self.action_space_string)}
        logger.log(f"in prediction: {q_table}")
        return q_table
    


    
    def train_batch(self):
        logger.log("starting batch training")
        # get batch from memory
        batch = random.sample(self.memory, BATCH_SIZE)
        
        # handling of all the elements for tensors
        states = []
        next_states = []
        actions = []
        rewards = []
        
        for state, action, reward, next_state in batch:
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
        
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        
        # logger.log(f"States shape: {states.shape}")
        # logger.log(f"Next states shape: {next_states.shape}")
        # logger.log(f"Actions shape: {actions.shape}")
        # logger.log(f"Rewards shape: {rewards.shape}")
        
        current_qs = self.model(states)
        next_qs = self.target_model(next_states)
        
        max_next_q = next_qs.max(1)[0]
        target_qs = rewards + (self.discount_factor * max_next_q)
        
        q_values = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(q_values, target_qs.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.log(f"Batch training completed. Loss: {loss.item():.4f}")
        
        
    def _save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, f"{MODEL_NAME}.pt")

    def _load_model(self):
        checkpoint = torch.load(f"{MODEL_NAME}.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']




    def get_epsilon(self):
        return self.epsilon
