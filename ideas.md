# ideas

use struct lib in python to read struct sent from c++
```c
typedef struct {
    int lines_cleared;      // to be maximized
    int height;             // to be minimized
    int holes;              // to be minimized
    int bumpiness           // to be minimized
}State;
```
then we get a formula that is trained by the q-learning algorithm
`a*Height + b*lines_cleared + c*holes + d*bumpiness`


print content of named pipe.
```bash
cat < named_pipe
```

### handling the game
in each iteration in `main_loop` :
1. current state with current falling piece, holes, bumpiness, etc is shared to python code
2. is analysed and sent in python code
3. is received in c and executed
4. gravity tick

### fixing attributes

- there are 14 cols in a row
- 