import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Empty string to disable CUDA

import torch
#torch.set_num_threads(1)  # This helps prevent multiprocessing issues
from nn_model import CNN, nn, DB_CNN

from typing import Tuple, Optional
import numpy as np
import random
from collections import deque

from simpleLogger import SimpleLogger
from tetris_expert import TetrisExpert
from state import State
from config import *
from memory import Memory
import utils

os.chdir(SRC_DIR)

logger = SimpleLogger()


logger.log(f"using device: {device}")

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
        
        #replacing normal deque with priority based model
        self.memory              = Memory(maxlen=MEMORY_MAXLEN, bias_recent=USE_RECENCY_BIAS, bias_reward=USE_REWARD_BIAS)
        self.expert_memory       = Memory(maxlen=MEMORY_EXPERT_MAXLEN, bias_recent=False)
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
        self.tetris_expert       = TetrisExpert(self.actions, expert_period=200000, expert_period_len=5)
        self.step                = 0


        self.model        = DB_CNN(num_actions=num_actions, simple_cnn=SIMPLE_CNN).to(device)
        self.target_model = DB_CNN(num_actions=num_actions, simple_cnn=SIMPLE_CNN).to(device)
        

        
        self.target_update_counter = 0
        self.target_update_frequency = 3000
        
        self.counter_interlearning_imitation = 0
        self.counter_interlearning_imitation_target = 20 
            
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
        # Copy weights to target model
        self.target_model.load_state_dict(self.model.state_dict())
        
        
        self.loss_history = []
        self.avg_loss_window = 100  #how many batches to avg loss
            
        if load_model:
            self._load_model()
            self._load_target_model()
            self.epsilon = LOAD_EPSILON     # only small expsilon here

        # compile after loading or else normal code. 
        torch.compile(self.model)
        torch.compile(self.target_model)

        if not load_model: 
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
                    self.imitation_learning_memory = Memory(maxlen=300000)
                    self.imitation_learning_memory.load_memory(path=MEMORY_PATH)
                    self.train_imitation_learning(#
                        batch_size=IMITATION_LEARNING_BATCH_SIZE, 
                        epochs_per_batch=IMITATION_LEARNING_EPOCHS,
                    )
        
        logger.log(f"actions in __init__: {self.actions}")
        #self.train_on_basic_scenarios()
        
        
    def train_imitation_learning(self, batch_size: int, epochs_per_batch: int):
        self.imitation_learning_memory.bias_recent = False
        memory_as_list = self.imitation_learning_memory.memory_list.copy()
        dataset_size = len(self.imitation_learning_memory)
        losses = []

        self.imitation_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=IMITATION_LEARNING_LR)
        
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
        states_column_features = []
        actions = []
        
        # reward does not matter here. so i can save good memory once and then use it for different reward implementations^
        for (state_game_board, state_piece_type, state_column_features), action, _, _ in batch:
            states_game_board.append(state_game_board)
            piece_types.append(state_piece_type)
            if isinstance(state_column_features, np.ndarray):
                state_column_features = torch.from_numpy(state_column_features).float()
            states_column_features.append(state_column_features.reshape(-1))  # Flatten to 1D
            actions.append(action)
        
        # convert to tensors
        states_game_board = torch.stack(states_game_board).to(device)
        piece_types = torch.stack(piece_types).to(device)
        states_column_features = torch.stack(states_column_features).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        
        
        # this reshape is necessary for imitation learning
        state_column_features = states_column_features.view(states_game_board.size(0), -1)

        # preds in logits, to be used for corss entropy loss (classification)
        # tried before with MSE, but had really bad results. 
        logits = self.model(
            x=states_game_board, 
            piece_type=piece_types, 
            column_features=state_column_features
        )
        
        # Use the full logits tensor and let CrossEntropyLoss select the expert action index
        loss = nn.CrossEntropyLoss()(logits, actions)
        
        self.imitation_optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.imitation_optimizer.step()
        
        return loss.item()
    
    def train_independent_batches(self):
        sample_size = NUM_BATCHES * BATCH_SIZE
        batch_unbiased = self.memory.sample_no_bias(k=sample_size)
        batch_biased   = self.memory.sample_with_reward_bias(k=sample_size) #reward bias
        batch_biased_rec = self.memory.sample_with_recent_bias(k=sample_size) #recency bias
        
        if USE_REWARD_BIAS: 
            for idx in range(NUM_BATCHES):
                batch = batch_biased[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                self.train_batch(batch)
            
            logger.log(f"\nstarting unbiased training!")
            for idx in range(NUM_BATCHES//2): 
                batch = batch_unbiased[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                self.train_batch(batch)
                
        elif USE_RECENCY_BIAS: 
            for idx in range(NUM_BATCHES):
                batch = batch_biased_rec[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                self.train_batch(batch)
            
            logger.log(f"\nstarting unbiased training!")
            for idx in range(NUM_BATCHES//2):
                batch = batch_unbiased[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                self.train_batch(batch)
        
        # no bias
        else:
            for idx in range(NUM_BATCHES):
                batch = batch_unbiased[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                self.train_batch(batch)
                
            # train a few iterations on biased data
            logger.log("\ntraining on biased data!")
            for idx in range(10):
                batch = batch_biased[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                self.train_batch(batch)
                
        
        if self.counter % 20000 == 0:
            logger.log("\ntraining on expert memory!")
            expert_batch = self.expert_memory.sample_no_bias(k=sample_size)
            for idx in range(NUM_BATCHES):
                batch = expert_batch[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                if len(batch) == 0:
                    break
                self.train_batch(batch)
        
        # for i in range(NUM_BATCHES):
        #     #logger.log(f"processing batch from memory, current len: {len(self.memory)}")
        #     self.train_batch(self.memory)

        #     if self.counter % 20000 == 0:
        #         self.train_batch(self.expert_memory)
        #         logger.log(f"Expert memory training done. Current memory size: {len(self.memory)}")
        
        # if USE_REWARD_BIAS:
        #     logger.log(f"training on unbiased data!")
        #     for _ in range(NUM_BATCHES//2):
        #         # also some sampling on non biased data to avoid overfitting on only good data
                
        #         self.train_batch(self.memory, explicit_bias=False, unbiased_batch_size=BATCH_SIZE)
        # else:
        # # temp: only very few biased training steps. 
        #     for _ in range(10):
        #         logger.log(f"training on biased data!")
        #         self.train_batch(self.memory, explicit_bias=True)
        
    def train_batch_(self, fraction=2):
        if USE_REWARD_BIAS: 
            for i in range(NUM_BATCHES):
                batch = self.memory.sample_with_reward_bias(k=BATCH_SIZE)
                self.train_batch(batch=batch)
                
        elif USE_RECENCY_BIAS: 
            for i in range(NUM_BATCHES):
                batch = self.memory.sample_with_recent_bias(k=BATCH_SIZE)
                self.train_batch(batch=batch)
                
            
        elif not USE_REWARD_BIAS and not USE_RECENCY_BIAS:
            for i in range(NUM_BATCHES):
                batch = self.memory.sample_no_bias(k=BATCH_SIZE)
                self.train_batch(batch=batch)
                
            # do some biased training
            logger.log("\n\ntraining on biased data!")
            for j in range(7): 
                batch = self.memory.sample_with_reward_bias(k=BATCH_SIZE, temperature=1.0)
                self.train_batch(batch=batch,)
            return
            
        else:   
            raise ValueError("Bias not correctly set in config.py")
        
        # some unbiased training whenever bias is used
        for i in range(NUM_BATCHES//fraction):
            batch = self.memory.sample_no_bias(k=BATCH_SIZE)
            self.train_batch(batch=batch)

            
        

    def train(self, state: State, action, next_state: State, reward, is_expert_move: bool):
        state_array, piece_type = state.convert_to_array()
        next_state_array, next_piece_type = next_state.convert_to_array()

        state_column_features = state.get_column_features()
        next_state_column_features = next_state.get_column_features()
        
        # convert to tensors 
        
        state_array = torch.from_numpy(state_array).float().to(device)
        piece_type = torch.from_numpy(piece_type).float().to(device)
        state_column_features = torch.from_numpy(state_column_features).float().to(device)
        next_state_array = torch.from_numpy(next_state_array).float().to(device)
        next_piece_type = torch.from_numpy(next_piece_type).float().to(device)
        next_state_column_features = torch.from_numpy(next_state_column_features).float().to(device)
        
        # action space has negative values -> just workaround for this
        norm_action = State.normalize_action(action)
        
        #self.memory.append(((state_array, piece_type), norm_action, reward, (next_state_array, next_piece_type)))
        self.memory.add(((state_array, piece_type, state_column_features), norm_action, reward, (next_state_array, next_piece_type, next_state_column_features)))
        
        # if the current state, action, reward pair is computed by the greedy expert 
        # => store in separate expert memory. 
        if is_expert_move: 
            self.expert_memory.add(((state_array, piece_type, state_column_features), norm_action, reward, (next_state_array, next_piece_type, next_state_column_features)))
        
        if len(self.memory) >=12000 and self.counter % COUNTER == 0 :
            
            if  (not ONLY_TRAINING) and  IMITATION_COLLECTOR:
                # save list as pickle (checkpointing)
                logger.log(f"saving memory, current memory size: {len(self.memory)}")
                self._save_memory(MEMORY_PATH)
            
            else:
                #self.train_independent_batches()       # compute single large batch, is then splitted. NO REPLACEMENT
                self.train_batch_(fraction=2)           # compute small batches, replacement allowed
                

                
        # sync the target and normal models.
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_frequency: 
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
        self.counter += 1
        # checkpointing
        if self.counter % COUNTER == 0:
            self._save_model(suffix=self.counter)


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
        state_array = torch.from_numpy(state_array).float().to(device)
        piece_type  = torch.from_numpy(piece_type).float().to(device)

        column_features = state.get_column_features()
        column_features = torch.from_numpy(column_features).float()
        
        is_expert_move = False


        # use expert step to move forward. 
        # we want to have a period (eg 100 pieces) of epxert placement regardless of epsilon
        # to trigger this, use the expert here

        if self.tetris_expert.step() == True: 
            return_val = self.tetris_expert.get_best_move(state=state)
            if return_val is None:
                return_val = np.random.choice(self.actions)
            return return_val, True

        # exploration
        if random.random() <= self.epsilon:
            # this is the tetris expert for imitation learning
            # if self.counter % 100 in [
            #     i for i in range(1,96)
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
            with torch.no_grad():  
                q_values = self.model(
                    state_array.unsqueeze(0), 
                    piece_type.unsqueeze(0), 
                    column_features.unsqueeze(0),
                )[0]
                # normalized actions for lookup, denorm for return
                action_q_values = {State.denormalize_action(i): q_value.item() 
                                for i, q_value in enumerate(q_values)}
                return_val = max(action_q_values.items(), key=lambda x: x[1])[0]
                
                #logger.log(f"return_val: {return_val}")

        # Epsilon decay
        self.counter_epsilon += 1
        if self.counter_epsilon == EPSILON_COUNTER_EPOCH:
            if self.epsilon >= MIN_EPSILON:
                # only temp. really slow decay at first 12000
                self.epsilon *= self.epsilon_decay
                # if self.counter < 15000:
                #     self.epsilon -= 0.000006        #12*0.000006 = 0.72
                # else:
                #     self.epsilon *= self.epsilon_decay
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
    
    def train_batch(self, batch):
        # update learning rate
        if USE_LR_SCHEDULER:
            lr = utils.get_lr(
                iteration=self.step, 
                warmup_steps=WARMUP_STEPS, 
                max_steps=MAX_STEPS, 
                init_lr=LEARNING_RATE, 
                min_lr=MIN_LEARNING_RATE
            )
            for param_group in self.optimizer.param_groups: 
                param_group["lr"] = lr

        
        # handling of all the elements for tensors
        states_game_board = []
        piece_types = []
        next_states_game_board = []
        next_piece_types = []
        actions = []
        rewards = []
        states_column_features = []
        next_states_column_features = []

        for (state_game_board, state_piece_type, state_column_features), action, reward,(next_state_game_board, next_state_piece_type, next_state_column_features) in batch:
            states_game_board.append(state_game_board)
            piece_types.append(state_piece_type)
            states_column_features.append(state_column_features)

            next_states_game_board.append(next_state_game_board)
            next_piece_types.append(next_state_piece_type)
            next_states_column_features.append(next_state_column_features)
            actions.append(action)
            rewards.append(reward)
        
        states_game_board = torch.stack(states_game_board).to(device)
        piece_types = torch.stack(piece_types).to(device)
        states_column_features = torch.stack(states_column_features).to(device)

        next_states_game_board = torch.stack(next_states_game_board).to(device)
        next_piece_types = torch.stack(next_piece_types).to(device)
        next_states_column_features = torch.stack(next_states_column_features).to(device)

        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)

        for epoch in range(EPOCHS):
            current_qs = self.model(
                x=states_game_board, 
                piece_type=piece_types,
                column_features=states_column_features
            )
            
            next_qs = self.target_model(
                x=next_states_game_board, 
                piece_type=next_piece_types,
                column_features=next_states_column_features
            )
            
            max_next_q = next_qs.max(1)[0]
            target_qs = rewards + (self.discount_factor * max_next_q)
            
            q_values = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)

            loss = nn.HuberLoss()(q_values, target_qs.detach())
            
            self.optimizer.zero_grad()
            loss.backward()
            
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            self.loss_history.append(loss.item())

            self.step += 1
            
            logger.log(f"Epoch {epoch+1}: Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            with torch.no_grad():
                q_values = current_qs.gather(1, actions.unsqueeze(1))
                log_string = f"norm: {norm:.4f}, q_mean: {q_values.mean().item():.4f}, q_std: {q_values.std().item():.4f}, max_q: {q_values.max().item():.4f}, min_q: {q_values.min().item():.4f}, td_error: {(target_qs - q_values).abs().mean().item():.4f}"
                logger.log(log_string)

    
        
    def _save_model(self, suffix: Optional[str]=None):
        if USE_LR_SCHEDULER: 
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
            #}, f"{MODEL_NAME}-{suffix}.pt")
            }, f"{MODEL_NAME}-{1}.pt")
        else: 
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
            #}, f"{MODEL_NAME}-{suffix}.pt")
            }, f"{MODEL_NAME}-{1}.pt")
        
    def _load_model(self):
        checkpoint = torch.load(f"{MODEL_NAME}.pt", map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
    def _load_target_model(self): 
        checkpoint = torch.load(f"{MODEL_NAME}.pt", map_location=device)
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']




    def get_epsilon(self):
        return self.epsilon
