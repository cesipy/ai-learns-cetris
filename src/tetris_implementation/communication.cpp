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
    int bumpiness;
    int highest_point_in_a, highest_point_in_b;

    // get the highest position in first column
    for(int i=0; i<g->rows;i++)
    {
        if (g->game_board[i][0].fixed_piece)
        {
            highest_point_in_a = i;
            break;
        }
    }

    for (int j=1; j<g->cols; j++)
    {
        for (int i=0; i<g->rows; i++)
        {
            if (g->game_board[i][j].fixed_piece)
            {
                highest_point_in_b = i;

                // calculate difference
                int delta = highest_point_in_a - highest_point_in_b;
                delta = delta < 0 ? -delta : delta;

                //add to overall bumpiness:
                bumpiness += delta;

                highest_point_in_a = highest_point_in_b;
                break;
            }
        }
    }
}