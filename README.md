# AI learns cetris - Reinforement Machine Learning
tail -f ../logs/py_log_2025-02-21.txt


## Demo
This is a demo of the model playing tetris.


### Four pieces
The trained model achieved good performance with about 160 pieces cleared. 

![four pieces](./res/demo-4-pieces.gif)

### Seven (all) pieces
in progress...


## Run the code 

In order to run the code, you need to have the python dependencies installed (torch, etc). The dependencies are stored in the `requirements.txt`. To install: 

```bash
pip install -r requirements.txt
```

As the C-code uses the `ncurses` library, it is necessary to install it. On Ubuntu, you can install it with: 

```bash
sudo apt-get install libncurses5-dev libncursesw5-dev
```

### Run a trained model
To run a trained model, simply adjust the `src/config.py` to your needs. This means copying the code from `src/run_config.py` to  `src/config.py`. Then you can run the model with the `./running_script`(from `src`) or using docker. I recommend using docker, but there you cannot see the game.


Of course you can choose the model you want to run. The models are stored in the `models` directory. 

```python
MODEL_NAME = "../models/trained_4_pieces_140_avg.pt"  # where are models saved? (for e.g. checkpointing )
MEMORY_PATH = "../res/precollected-memory/memory.pkl"   # where to collect mem
```
Note: The models are trained with different numbers of pieces. To adjust the number of pieces please set `NUMBER_OF_PIECES` in `src/config.py` AND in `src/cpp/tetris.h` under `#define AMOUNT_OF_PIECES 4`


### Train a model from scratch 
To train a model from scratch, adjust the `src/config.py` to your needs. This means copying the code from `src/train_config.py` to  `src/config.py`. Then you can run the model with the `./running_script`(from `src`) or using docker. I recommend using docker, but there you cannot see the game.


## run with docker
With docker it is possible to run several different configs for training. Each one uses just one core. 
different containers are available in `docker-compose.yaml`. 
To run one container, simply type: 

```bash
# run all experiments
docker-compose up

# run specific experiments
docker-compose up --build experiment_1
docker-compose up --build experiment_2

```


## Architecture
This project consists of two major parts: 
- Tetris implemented in C++ (heavy C-style), displaying the game using `ncurses`
- Reinforcement learning model implemented in Python, using `pytorch`. 

In order to communicate between the two parts, a FIFO is used. The C++ code writes the game state to a file, and the python code reads it. The python code writes the action to a file, and the C++ code reads it.



## Take-aways
It was really hard to train a model from scratch and then achieve good results. Most of them got stuck or plateaued at a certain point. As there are so many hyperparameters, it is hard to find the right combination.

I imporved my workflow by checkpointing the models when they reached some good point and finetuned them further with a slightly smaller learning rate. With the Docker Setup it was possible to run multiple experiments in parallel.

The hardest parts in learning were: 
- Finding good reward function (hardest)
- how much imitation learning/expert learning?
- hyperparameters ($\gamma, \lambda, \epsilon, lr$, batch_size, etc)


## Working version for four pieces reloaded
Switch to commit "856b9a834909d643897f0f2e2610bc221b455fea" and train the model from scratch. after about 15k episodes the model averaged to 7 lines cleared. Then I saved the model and tuned it with a smaller learning rate and a smaller replay memory. This helped to achieve ... pieces. 

This is a video showcasing the training process: 
![traininge](./res/training_4_pieces.gif)

## working version with four pieces reloaded 2
model is trained from commit: "452fa334dc9c433425a1dde0d0fdecad9e86a21e".
This resulted in an average of 16 pieces cleared. Then finetuned furthermore.  

## Working version for four pieces. 
Switch to commit "insert here". 

The model reached a plateau at about 8k episodes with avg 20 lines cleared. 
Uses the following hyperparams: 
```python
Training:
---------
EPSILON....................... 1.0
EPSILON_DECAY................. 0.996
DISCOUNT...................... 0.85
LEARNING_RATE................. 0.002
BATCH_SIZE.................... 64
COUNTER....................... 2000
EPOCHS........................ 2
NUM_BATCHES................... 70
MIN_EPSILON................... 0.01
EPSILON_COUNTER_EPOCH......... 50

Model:
------
BOARD_HEIGHT.................. 28
BOARD_WIDTH................... 10
FC_HIDDEN_UNIT_SIZE........... 150
NUMBER_OF_PIECES.............. 4

Environment:
------------
BASE_DIR...................... /app
SRC_DIR....................... /app/src
LOG_DIR....................... /app/logs
RES_DIR....................... /app/res
TETRIS_COMMAND................ /app/src/cpp/tetris

Communication:
--------------
FIFO_STATES................... fifo_states
FIFO_CONTROLS................. fifo_controls
COMMUNICATION_TIME_OUT........ 45.0
SLEEPTIME..................... 1e-06
INTER_ROUND_SLEEP_TIME........ 0.2

Experiment:
-----------
LOGGING....................... False
LOAD_MODEL.................... True
ITERATIONS.................... 100000
PLOT_COUNTER.................. 50
MOVING_AVG_WINDOW_SIZE........ 50
COUNTER_TETRIS_EXPERT......... 2

Other Variables:
--------------
ACTIONS....................... [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
IMITATION_COLLECTOR........... False
IMITATION_LEARNING_BATCH_SIZE. 64
IMITATION_LEARNING_EPOCHS..... 15
IMITATION_LEARNING_LR......... 0.002
IMITATIO_LEARNING_BATCHES..... 130
MEMORY_EXPERT_MAXLEN.......... 60000
MEMORY_MAXLEN................. 150000
MEMORY_PATH................... ../res/precollected-memory/memory.pkl
MODEL_NAME.................... ../models/model
ONLY_TRAINING................. False

POSSIBLE_NUMBER_STEPS......... 4
SIMPLE_CNN.................... True
USE_LR_SCHEDULER.............. True
USE_RECENCY_BIAS.............. False
USE_REWARD_BIAS............... True
np............................ <module 'numpy' from '/usr/local/lib/python3.11/site-packages/numpy/__init__.py'>
os............................ <module 'os' (frozen)>
warnings...................... <module 'warnings' from '/usr/local/lib/python3.11/warnings.py'>
08:10:24- fd_states: 8 fd_controls: 7
``` 


## Reward function
The tuning of the reward function was the central part for the learning process. After hundreds of trial and errors I came up with the following reward function. It was working very well for (adjust) x pieces. 

```python
def calculate_reward(next_state: State):
    reward = 0
    
    if next_state.immedeate_lines_cleared > 0:
        fraction = next_state.piece_count / 70
        if fraction > 1: 
            line_cleared_reward = (next_state.immedeate_lines_cleared ** 2) * 200
        else:
            line_cleared_reward = (next_state.immedeate_lines_cleared ** 2) * 200 * fraction*2

        reward += line_cleared_reward
        
    survival_bonus = 1.0* next_state.piece_count 
    reward += survival_bonus
    
    reward -= 0.3 * next_state.get_height_variance() ** 1.5
    reward -= 1.5 * next_state.max_height

    reward -= 0.2 * next_state.bumpiness
    reward -= 0.95 * next_state.holes
    
    if next_state.is_state_game_over():
        game_over_penalty = 500 
        reward -= game_over_penalty

    return reward/500.0
```

Both the expert used and the model optimized this function. 

In the beginning I used really simple rewards such as only lines cleared and number of pieces on the board. But then the agent never reached the reward, as it was highly unlikely to clear lines at random. So I implemented negative rewards or punishments as described [here](https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf). 
But also here the model struggled to learn and discover positive rewards.

In combination with greedily expert moves the model was able to learn the game with the above reward function.



## Expert training
To discover positive rewards like clearing lines, I implemented a tetris expert. This expert is a greedy algorithm that tries all possible moves and calculates the reward of the resulting states. The best move is then chosen.

 


## Working version for two pieces. 
Switch to commit `0ccc4eb0ee345ab8a20dfde3619505e0f51d0e36` and use `models/trained_two_pieces_new.pt`. In the commit `0ccc4eb` everything should work. Note that in order to run it on Docker, you also need to copy it in the `Dockerfile`!


The model improves drastically at about 4k-4.5k epochs. 
Uses the following hyperparams: 


![plot](./res/newplot.png)

