from typing import List, Tuple
import random 
import numpy as np

import matplotlib.pyplot as plt

from simpleLogger import SimpleLogger

logger = SimpleLogger()

class Memory():
    def __init__(self, maxlen:int, bias:bool=False): 
        self.maxlen = maxlen
        self.memory_list = []
        self.bias = bias
        
        
        
    def add(self, elem) -> None: 
        if len(self.memory_list) >= self.maxlen: 
            # remove the first element
            self.memory_list = self.memory_list[1:].copy()
            self.memory_list.append(elem)
        else: 
            self.memory_list.append(elem)
            
    def __len__(self,) -> int:
        return len(self.memory_list)
    
    def sample(self, k) -> List: 
        if self.bias: 
            return self.sample_with_recent_bias(k=k)
        
        return random.sample(self.memory_list, k=k)
    
    
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
        
        samples = np.random.choice(self.memory_list, size=k, replace=False, p=prob)
        return samples
    
    
    def construct_probability_distr(self, ): 
        if not self.bias:
            # return uniform 
            length = len(self.memory_list)
            unif = [1/length for i in self.memory_list]
            return unif
        
        # constrcut own probability distribution for the task. 
        copy_list = self.memory_list.copy()
        probs = []

        recent_bias_fraction = 0.3
        split_index = int( recent_bias_fraction *  
                       len(self.memory_list)     
        )
        
        denom_recent =  (
            2 *         # times two, because i want to norm sum of distribution to 1
            recent_bias_fraction * len(self.memory_list)
            )
        denom_earlier = (
            2 * 
            (1-recent_bias_fraction) * len(self.memory_list)
        )

        for i in copy_list[:split_index]:
            probs.append(1/denom_earlier)
        for i in copy_list[split_index:]:
            probs.append(1/denom_recent)
            
        #print(f"sum of prob: {sum(probs)}")
        
        return probs
        
        
def plot_sampling_distribution(memory: Memory):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, )
    
    # Plot 1: Distribution of multiple sampling runs
    all_samples = []
    for _ in range(100):  
        samples = memory.sample_with_recent_bias(k=1000)
        all_samples.extend(samples)
    
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
    memory = Memory(maxlen=maxlen, bias=True)
    
    for i in range(maxlen): 
        memory.add(i)
    
    prob = memory.construct_probability_distr()
    print(f"lengths: \n\tprobs:{len(prob)}, \n\tmemory: {len(memory)}")
    
    samples = memory.sample_with_recent_bias(k=100)
    print(samples)
    
    plot_sampling_distribution(memory=memory)
    
if __name__ == "__main__": 
    main() 