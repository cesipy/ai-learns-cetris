import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Empty string to disable CUDA

import torch
#torch.set_num_threads(1)  # This helps prevent multiprocessing issues
from nn_model import CNN, nn

from typing import Tuple
import numpy as np
import random
from collections import deque

from simpleLogger import SimpleLogger
from tetris_expert import TetrisExpert
from state import State
from config import *
from memory import Memory
os.chdir(SRC_DIR)

logger = SimpleLogger()
MODEL_NAME = "../models/model"
MEMORY_PATH = "../res/precollected-memory/memory.pkl"

ONLY_TRAINING = False           # only training, no pretraining with expert
IMITATION_COLLECTOR = False
IMITATIO_LEARNING_BATCHES = 130

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
        #self.memory              = deque(maxlen=70000)
        #replacing normal deque with priority based model
        self.memory              = Memory(maxlen=150000, bias_recent=False, bias_reward=True)
        self.expert_memory       = Memory(maxlen=50000, bias_recent=False)
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

        self.model        = CNN(num_actions=num_actions, simple_cnn=SIMPLE_CNN).to(device)
        self.target_model = CNN(num_actions=num_actions, simple_cnn=SIMPLE_CNN).to(device)
        self.target_update_counter = 0
        self.target_update_frequency = 1000
        
        self.counter_interlearning_imitation = 0
        self.counter_interlearning_imitation_target = 20 
            
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # Copy weights to target model
        self.target_model.load_state_dict(self.model.state_dict())
            
        if load_model:
            self._load_model()
            self._load_target_model()
            self.epsilon = 0.01     # only small expsilon here
            
        if not ONLY_TRAINING:           # circumvents the imitation collector
            if IMITATION_COLLECTOR:
                if os.path.exists(MEMORY_PATH):
                    self.memory.load_memory(MEMORY_PATH)
                    self.memory.maxlen = 100000
                    logger.log(f"loaded memory with size: {len(self.memory)}")
                else: 
                    logger.log("No existing memory found. Creating new memory for imitation learning collection.")
                    self.memory = Memory(maxlen=100000, bias_recent=False)  
                    
                    memory_dir = os.path.dirname(MEMORY_PATH)
                    if memory_dir and not os.path.exists(memory_dir):
                        os.makedirs(memory_dir, exist_ok=True)
                        logger.log(f"Created directory for storing memory: {memory_dir}")
                    
            else:
                self.imitation_learning_memory = Memory(maxlen=30000)
                self.imitation_learning_memory.load_memory(path=MEMORY_PATH)
                #self.train_imitation_learning(batch_size=1024, epochs_per_batch=1)


        logger.log(f"actions in __init__: {self.actions}")
        #self.train_on_basic_scenarios()
        
        
    def train_imitation_learning(self, batch_size: int, epochs_per_batch: int):
        self.imitation_learning_memory.bias_recent = False
        memory_as_list = self.imitation_learning_memory.memory_list.copy()
        dataset_size = len(self.imitation_learning_memory)
        losses = []

        self.imitation_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001)
        
        for epoch in range(epochs_per_batch):
            
            random.shuffle(memory_as_list)
            epoch_loss = 0
            
            for i in range(0, dataset_size, batch_size):
                batch = memory_as_list[i:i+batch_size]
                if len(batch) == 0: 
                    break
                loss = self.imitation_learning_step(batch)
                epoch_loss += loss
                #logger.log(f"epoch: {epoch+1}/{epochs_per_batch}, batch: {i//batch_size+1}/{dataset_size//batch_size}, loss: {loss:.4f}")
                
            avg_loss_epoch = epoch_loss / (dataset_size // batch_size)
            losses.append(avg_loss_epoch)
            
            logger.log(f"Epoch {epoch+1}/{epochs_per_batch}: loss: {avg_loss_epoch:.4f}")
            
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Create and save the loss plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Average Loss per Batch')
        plt.xlabel('Batch Number')
        plt.ylabel('Average Loss')
        plt.title('Imitation Learning Training Loss')
        plt.grid(True)
        plt.legend()
        
        plot_dir = os.path.dirname(MODEL_NAME)
        plot_path = os.path.join(plot_dir, 'imitation_learning_loss.png')
        
        # Create directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory
        
        logger.log(f"Loss plot saved to: {plot_path}")

        

    def imitation_learning_step(self, batch):
        """
        used to train one batch for imitation learning
        """
        states_game_board = []
        piece_types = []
        actions = []
        
        # reward does not matter here. so i can save good memory once and then use it for different reward implementations^
        for (state_game_board, state_piece_type), action, _, _ in batch:
            states_game_board.append(state_game_board)
            piece_types.append(state_piece_type)
            actions.append(action)
        
        # convert to tensors
        states_game_board = torch.stack(states_game_board).to(device)
        piece_types = torch.stack(piece_types).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)

        # Get model predictions
        q_values = self.model(states_game_board, piece_types)
        
        # For imitation learning, just match the expert's actions
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Create targets - 1 for taken action, 0 for others
        targets = torch.ones_like(q_values)

        loss = nn.MSELoss()(q_values, targets)
        
        self.imitation_optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.imitation_optimizer.step()
        
        return loss.item()

    def train(self, state: State, action, next_state: State, reward, is_expert_move: bool):
        state_array, piece_type = state.convert_to_array()
        next_state_array, next_piece_type = next_state.convert_to_array()
        # state_array = state.convert_to_array()           # this is a tuple of (gameboard and piecetype one-hot)
        # next_state_array = next_state.convert_to_array() # ditto
        
        # convert to tensors 
        
        state_array = torch.from_numpy(state_array).float()
        piece_type = torch.from_numpy(piece_type).float()
        next_state_array = torch.from_numpy(next_state_array).float()
        next_piece_type = torch.from_numpy(next_piece_type).float()
        
        # action space has negative values -> just workaround for this
        norm_action = State.normalize_action(action)
        
        #self.memory.append(((state_array, piece_type), norm_action, reward, (next_state_array, next_piece_type)))
        self.memory.add(((state_array, piece_type), norm_action, reward, (next_state_array, next_piece_type)))
        
        # if the current state, action, reward pair is computed by the greedy expert 
        # => store in separate expert memory. 
        if is_expert_move: 
            self.expert_memory.add(((state_array, piece_type), norm_action, reward, (next_state_array, next_piece_type)))
        
        if len(self.memory) >=1500 and self.counter % COUNTER == 0 :
            
            if  (not ONLY_TRAINING) and  IMITATION_COLLECTOR:
                # save list as pickle (checkpointing)
                logger.log(f"saving memory, current memory size: {len(self.memory)}")
                self._save_memory(MEMORY_PATH)
            
            else:
                for _ in range(NUM_BATCHES):
                    #logger.log(f"processing batch from memory, current len: {len(self.memory)}")
                    self.train_batch(self.memory)
                    if self.counter % 10000 == 0:
                        self.train_batch(self.expert_memory)
                        logger.log(f"Expert memory training done. Current memory size: {len(self.memory)}")
                
                
        # sync the target and normal models.
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_frequency: 
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
        self.counter += 1
        if self.counter % COUNTER == 0:
            self._save_model()


    def epsilon_greedy_policy(self, state: State) -> Tuple[int, bool]:
        """
        Method to get actions. Actions are either from current policy (exploitation),
        random, or expert moves (computed greedily by the tetris expert). 
        Random exploration aswell as tetris expert are set in the config files. 
        
        Whenever an expert move is taken, the boolean flag  `is_expert_move` is set to True.
        
        Args:
            state (State): current state of the game
            
        Returns:
            Tuple[int, bool]: action and a boolean flag indicating whether the action
            was taken by the expert
        """
        state_array, piece_type = state.convert_to_array()
        state_array = torch.from_numpy(state_array).float()
        piece_type  = torch.from_numpy(piece_type).float()
        
        is_expert_move = False
        # exploration
        if random.random() <= self.epsilon:
            # this is the tetris expert for imitation learning
            # if self.counter % 100 in [
            #     0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 25, 26, 27, 28,
            #     30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45 ,46, 47, 48, 50, 52 ,53,
            #     54, 55, 56, 57, 58, 59, 60, 65, 66, 67, 68, 84,85,86,87,88,89,90
            # ]:
            if self.cunter_tetris_expert % int(round(self.starting_tetris_expert_modulo)) == 0:
                self.cunter_tetris_expert = 0
                return_val = self.tetris_expert.get_best_move(state=state)
                if return_val is None:
                    return_val = np.random.choice(self.actions)
                self.starting_tetris_expert_modulo += 0.0
                is_expert_move = True
            else:
                return_val = np.random.choice(self.actions)
            self.cunter_tetris_expert += 1
            
        # exploitation
        else:
            with torch.no_grad():  # Don't track gradients for prediction
                q_values = self.model(state_array.unsqueeze(0), piece_type.unsqueeze(0))[0]
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

        return return_val, is_expert_move


    def predict(self, state):
        state_values = state.get_values()
        q_values = self.model.predict(np.array([state_values]), verbose=0)[0]
        q_table = {action: q_values[i] for i, action in enumerate(self.action_space_string)}
        logger.log(f"in prediction: {q_table}")
        return q_table
    

    def _save_memory(self, path: str): 
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.log(f"Created directory: {directory}")
        
        self.memory.save_memory(path=path)
        logger.log(f"successfully saved memory to file: {path}")
        
    def _load_memory(self, path:str): 
        pass
    
    def train_batch(self, memory):
        logger.log("starting batch training")
        # get batch from memory
        #batch = random.sample(self.memory, BATCH_SIZE)
        batch = memory.sample(BATCH_SIZE)
        
        # handling of all the elements for tensors
        states_game_board = []
        piece_types = []
        next_states_game_board = []
        next_piece_types = []
        actions = []
        rewards = []
        
        for (state_game_board, state_piece_type), action, reward,(next_state_game_board, next_state_piece_type) in batch:
            states_game_board.append(state_game_board)
            piece_types.append(state_piece_type)
            next_states_game_board.append(next_state_game_board)
            next_piece_types.append(next_state_piece_type)
            actions.append(action)
            rewards.append(reward)
        
        states_game_board = torch.stack(states_game_board).to(device)
        piece_types = torch.stack(piece_types).to(device)
        next_states_game_board = torch.stack(next_states_game_board).to(device)
        next_piece_types = torch.stack(next_piece_types).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)

        
        # logger.log(f"States shape: {states.shape}")
        # logger.log(f"Next states shape: {next_states.shape}")
        # logger.log(f"Actions shape: {actions.shape}")
        # logger.log(f"Rewards shape: {rewards.shape}")
        
        current_qs = self.model(states_game_board, piece_types)
        next_qs = self.target_model(next_states_game_board, next_piece_types)
        
        max_next_q = next_qs.max(1)[0]
        target_qs = rewards + (self.discount_factor * max_next_q)
        
        q_values = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.HuberLoss()(q_values, target_qs.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
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
        
    def _load_target_model(self): 
        checkpoint = torch.load(f"{MODEL_NAME}.pt")
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']




    def get_epsilon(self):
        return self.epsilon
