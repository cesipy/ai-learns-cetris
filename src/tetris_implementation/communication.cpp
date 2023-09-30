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


void calculate_holes(Game* g, State* s) {
    int number_holes = 0;
    bool reached_first_fixed_blocks = false;

    for (int i = 0; i < 29 - 1; i++)    //temp: magic numbers because of distortion of game height and width TO BE FIXED
    { // Avoid checking beyond the last row
        for (int j = 0; j < 15; j++)    //temp: magic numbers because of distortion of game height and width TO BE FIXED
        {
            Block block = g->game_board[i][j];
            if (block.fixed_piece)
            {
                reached_first_fixed_blocks = true;
            }

            bool condition = reached_first_fixed_blocks
                             && block.value == EMPTY_CELL
                             && g->game_board[i - 1][j].fixed_piece;

            if (condition) {
                number_holes++;
            }
        }
    }

    s->holes = number_holes;
}



void calculate_bumpiness(Game* g, State* s)
{
    int bumpiness = 0;
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

    for (int j=1; j<15; j++)        // width = 14, so <15
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

    s->bumpiness = bumpiness;
}


void update_state(Game* g, State * s)
{
    calculate_lines_cleared(g, s);
    calculate_height(g, s);
    calculate_bumpiness(g, s);
    calculate_holes(g, s);
    s->piece_type = g->piece_type;
}


char* state_to_string(const State* s) {
    std::ostringstream output;

    // Format the struct values into a string
    output << "Lines Cleared: " << s->lines_cleared
           << ", Height: " << s->height
           << ", Holes: " << s->holes
           << ", Bumpiness: " << s->bumpiness
           << ", Piece Type: " << s->piece_type << "\n";

    // Create a copy of the string and return its c_str()
    std::string str = output.str();
    char* result = new char[str.length() + 1]; // +1 for null-terminator
    strcpy(result, str.c_str());

    return result;
}


const int setup_named_pipe(const char* name)
{
    const mode_t permission = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH; // 644
    if (mkfifo(name, permission) != 0)
    {
        // add error handling
        fprintf(stderr, "error @ mkfifo");
        exit(EXIT_FAILURE);
    }

    const int fd = open(name, O_RDWR);
    if (fd < 0)
    {
        fprintf(stderr, "error @ opening fifo");
        exit(EXIT_FAILURE);
    }

    return fd;

}