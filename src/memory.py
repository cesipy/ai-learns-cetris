from typing import List, Tuple
import random 
import numpy as np
import pickle
import torch
import os

from config import *

os.chdir(SRC_DIR)

from simpleLogger import SimpleLogger

logger = SimpleLogger()

class Memory():
    def __init__(
        self, maxlen:int, 
        bias_recent:bool=False,
        bias_reward:bool=False,
    ): 
        self.maxlen = maxlen
        self.memory_list = []
        self.bias_recent = bias_recent
        self.bias_reward = bias_reward
        
        
        
    def add(self, elem) -> None: 
        if len(self.memory_list) >= self.maxlen: 
            # remove the first element
            self.memory_list = self.memory_list[1:].copy()
            self.memory_list.append(elem)
        else: 
            self.memory_list.append(elem)
            
    def __len__(self,) -> int:
        return len(self.memory_list)
    
    def sample_no_bias(self, k:int): 
        """explicit uniform sampling from memory"""
        if len(self.memory_list) <= k:
            return self.memory_list.copy()
        return random.sample(self.memory_list, k=k)
    
    def sample(self, k) -> List: 
        if len(self.memory_list) <=k: 
            return self.memory_list.copy()
        if self.bias_recent and self.bias_reward: 
            logger.log("biasing both recent and reward not working, defaulting to no bias")
            return random.sample(self.memory_list, k=k)
        
        elif self.bias_recent: 
            return self.sample_with_recent_bias(k=k)
        
        elif self.bias_reward:
            return self.sample_with_reward_bias(k=k)
        
        return random.sample(self.memory_list, k=k)
    
    def sample_with_reward_bias(self, k:int, temperature=REWARD_TEMPERATURE) -> List:
        """
        Sample k elements from the memory buffer with a bias towards higher rewards.
        
        Args:
            k (int): Number of elements to sample
            temperature (float): Temperature parameter for the softmax function used to bias the sampling.
                A value of 1.0 results in a strong bias, while a value of 0.0 results in a uniform distribution."""
        if len(self.memory_list) <= k: 
            return self.memory_list.copy()
        
        rewards = []
        
        for t in self.memory_list: 
            rewards.append(t[2])        # extract rewards fro tuple t
        
        min_reward = min(rewards)
        max_reward = max(rewards)
        reward_range = max_reward - min_reward + 1e-6
        
        normalized_rewards = [ (r - min_reward) / reward_range for r in rewards]  # like partition function, small value
        
        # temperature for the bias
        # temperature = 1: strong bias
        # temperature = 0: uniform distr.
        
        biased_rewards = [r ** temperature for r in normalized_rewards]
        

        
        total_rewards = sum(biased_rewards)
        probs         = [r/total_rewards for r in biased_rewards]
        
        samples_indecies = np.random.choice(
            len(self.memory_list),
            size=k, 
            replace=False, 
            p=probs
        )
        
        samples = [self.memory_list[index] for index in samples_indecies]
        return samples
    
    
    def sample_with_recent_bias(self, k) -> List: 
        
        # not enough samples to choose from
        if len(self.memory_list) <=k: 
            return self.memory_list.copy()
        
        
        if not len(self.memory_list) == self.maxlen: 
            # do normal sampling, as memory is not full -
            #handling properties of probability distribution is harder, I dont have time for this.
            logger.log(f"defaulting to normal choice as memory is still not full. current memory size: {len(self.memory_list)}")
            return random.sample(self.memory_list, k=k)
        prob = self.construct_probability_distr()
        
        # if this is not correct, the whole calculation is not working
        assert(len(self.memory_list) == len(prob))
        
        samples_indices = np.random.choice(len(self.memory_list), size=k, replace=False, p=prob)
        
        samples = [self.memory_list[index] for index in samples_indices]
        return samples
    
    
    def construct_probability_distr(self, ): 
        if not self.bias_recent:
            # return uniform 
            length = len(self.memory_list)
            unif = [1/length for i in self.memory_list]
            return unif
        
        # constrcut own probability distribution for the task. 
        copy_list = self.memory_list.copy()
        probs = []

        recent_bias_fraction = 0.7
        split_index = int( recent_bias_fraction *  
                       len(self.memory_list)     
        )
        
        denom_recent =  (
            2 *         # times two, because i want to norm sum of distribution to 1
            (1 -recent_bias_fraction) * len(self.memory_list)
            )
        denom_earlier = (
            2 * 
            (recent_bias_fraction) * len(self.memory_list)
        )

        for i in copy_list[:split_index]:
            probs.append(1/denom_earlier)
        for i in copy_list[split_index:]:
            probs.append(1/denom_recent)
            
        #print(f"sum of prob: {sum(probs)}")
        
        return probs
    
    def save_memory(self, path: str) -> None:
        """
        Save the memory buffer to a file using pickle.
        The method handles PyTorch tensors by converting them to numpy arrays before saving.
        
        Args:
            path (str): Path where the memory buffer will be saved
        """
        serializable_memory = []

        for exp in self.memory_list: 
            (state, norm_action, next_state) = exp

            serializable_memory.append((
                state.to_dict(), 
                norm_action, 
                next_state.to_dict()	
            ))



        # serializable_memory = []
        
        # for experience in self.memory_list:
        #     (state_array, piece_type, state_column_features), action, reward, (next_state_array, next_piece_type, next_state_column) = experience
            
        #     serializable_memory.append((
        #         (
        #             state_array.cpu().numpy() if torch.is_tensor(state_array) else state_array,
        #             piece_type.cpu().numpy() if torch.is_tensor(piece_type) else piece_type,
        #             state_column_features.cpu().numpy() if torch.is_tensor(state_column_features) else state_column_features
        #         ),
        #         action,
        #         reward,
        #         (
        #             next_state_array.cpu().numpy() if torch.is_tensor(next_state_array) else next_state_array,
        #             next_piece_type.cpu().numpy() if torch.is_tensor(next_piece_type) else next_piece_type, 
        #             next_state_column.cpu().numpy() if torch.is_tensor(next_state_column) else next_state_column
        #         )
        #     ))
        
        with open(path, 'wb') as f:
            pickle.dump({
                'memory_list': serializable_memory,
                'maxlen': self.maxlen,
                'bias': self.bias_recent
            }, f)

    def load_memory(self, path: str) -> None:
        """
        Load the memory buffer from a file.
        The method handles converting numpy arrays back to PyTorch tensors.
        
        Args:
            path (str): Path to the saved memory buffer file
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            

        self.maxlen = data['maxlen']
        self.bias_recent = data['bias']
        
        self.memory_list = []
        for experience in data['memory_list']:
            (state_dict, action, next_state_dict) = experience
            self.memory_list.append((
                state_dict.from_dict(), 
                action, 
                next_state_dict.from_dict()
            ))

            # (state_array, piece_type, state_column_features), action, reward, (next_state_array, next_piece_type, next_state_column_features) = experience
            
            # # Convert numpy arrays to tensors
            # self.memory_list.append((
            #     (
            #         torch.from_numpy(state_array).float() if isinstance(state_array, np.ndarray) else state_array,
            #         torch.from_numpy(piece_type).float() if isinstance(piece_type, np.ndarray) else piece_type,
            #         torch.from_numpy(state_column_features).float() if isinstance(state_column_features, np.ndarray) else state_column_features
            #     ),
            #     action,
            #     reward,
            #     (
            #         torch.from_numpy(next_state_array).float() if isinstance(next_state_array, np.ndarray) else next_state_array,
            #         torch.from_numpy(next_piece_type).float() if isinstance(next_piece_type, np.ndarray) else next_piece_type, 
            #         torch.from_numpy(next_state_column_features).float() if isinstance(next_state_column_features, np.ndarray) else next_state_column_features
            #     ),
                
            # ))
        
# ----------------------------------------------
        
def plot_sampling_distribution(memory: Memory):
    import matplotlib.pyplot as plt
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, )
    
    # Plot 1: Distribution of multiple sampling runs
    all_samples = []
    for _ in range(100):  
        samples = memory.sample_with_recent_bias(k=1000)
        all_samples.extend(samples)
    
    all_samples = [i[1] for i in all_samples]
    ax1.hist(all_samples, bins=50, density=True, alpha=0.7, color='blue')
    ax1.set_title('Distribution of Sampled Values (100 runs)')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    
    # Plot 2: Theoretical Probability Distribution
    x = np.arange(len(memory.memory_list))
    prob_dist = memory.construct_probability_distr()
    
    
    ax2.plot(x, prob_dist, color='red', label='Probability Distribution')
    ax2.set_title('Theoretical Probability Distribution')
    ax2.set_xlabel('Index in Memory')
    ax2.set_ylabel('Probability')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

        
def main():
    maxlen = 30000
    memory = Memory(maxlen=maxlen, bias_recent=True)
    
    for i in range(maxlen): 
        memory.add(((i, i+12), i+1, i,(i-1, i)))
    
    prob = memory.construct_probability_distr()
    print(f"lengths: \n\tprobs:{len(prob)}, \n\tmemory: {len(memory)}")
    
    samples = memory.sample_with_recent_bias(k=100)
    print(samples)
    
    plot_sampling_distribution(memory=memory)
    
    
    path = "memory.pkl"
    memory.save_memory(path=path)
    
    del memory
    memory = Memory(maxlen=maxlen, bias_recent=True)
    memory.load_memory(path=path)
    
    plot_sampling_distribution(memory=memory)
    
if __name__ == "__main__": 
    main() 