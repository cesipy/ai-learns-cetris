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


void calculate_height(Game* g, State* s)
{
    int height = 0;
    for(int i=0; i< g->rows; i++)
    {
        for (int j=0; j<g->cols; j++)
        {
            if (g->game_board[i][j].fixed_piece)
            {
                height++;
            }
        }
    }
    s->height = height;
}


void calculate_holes(Game* g, State* s)
{
    int number_holes = 0;
    // for referencing
    bool reached_first_fixed_blocks = false;
    for (int i=0; i<g->rows; i++)
    {
        for (int j=0; j<g->cols; j++)
        {
            Block block = g->game_board[i][j];
            if (block.fixed_piece) { reached_first_fixed_blocks = true; }

            bool condition = reached_first_fixed_blocks
                    && block.value == EMPTY_CELL
                    && g->game_board[i+1][j].fixed_piece;
            if (condition)
            {
                number_holes++;
            }
        }
    }

    s->holes = number_holes;
}


void calculate_bumpiness(Game* g, State* s)
{

}