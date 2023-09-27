#include "tetris.h"


int rxc(void)
{
    printf("hello");
    return 1;
}

void calculate_lines_cleared(Game* g, State* s)
{
    s->lines_cleared = g->score;
}