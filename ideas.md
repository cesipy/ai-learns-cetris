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
