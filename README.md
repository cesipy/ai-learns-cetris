# AI learns cetris - Reinforement Machine Learning
tail -f ../logs/py_log_2025-02-02.txt


- e4: only training, with super simple reward

## TODOs

- [ ] modify memory to include best rewards as bias. 

- [ ] record perfect matches, save to pickle and maybe pretrain on this for several episodes
- [ ] epochs in imitation training, also normal training
- [ ]  imitation training: partial random, partial completely expert
- [ ] unify batch training methods
- [ ] fix game-over funciton

- [x] priority based memory to favour recent experience
- [x] reduce size of C  logs
- [ ] initialize weights from uniform
- [x] include piece type in current state for cnn. i guess it doesnt know it right now
- [ ] normalize loss on batchsize
- [ ] logger queue for mp.Process
- [x] script to fetch logs from docker containers
- [x] deactivate oinly single thread in torch in qagent
- [ ] improve state function for tetris board - encode state to matrix with current position and all the blocks in the field. 
- [ ] in c: see next tetris piece
- [x] use seeds to fix the run: especially for debugging in the begining to know if it even works


- [x] is action space correct? research and write comprehensive documentation
    - [ ] still write documentation
- [x] test with easy tetroids
- [x] fix lines cleared!
- [ ] save outcomes to file where reward is bigger than 0
- [x] add some visualizations
- [x] batch processing only after n steps
- [x] tetrisexpert implementation


- [ ] documentation for piece_types:int  -> what is what?
- [x] add current_piece_count to state so that i can separate reward function from other code. 
- [x] maybe dockerize application so i can run several runs in parallel

- [ ] problem with placeholder action: state s1 received -> get action a1; s2 is direct result (has already new piece inserted!); s3 (=s2 + 1 tick) -> a2, s4, ...
    - remove the extra tick, has to be modified on c++ side. 


- [x] in tetris expert: if several actions ahave same reward, always first one is chosen -> make random to remove this bias
- [x] lines cleared from pipe is only 1 digit - what happens at more?


# run with docker
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