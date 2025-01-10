#include "tetris.hpp"

int gravity(Game* g)
{
    // check if the blocks below are free
    bool can_move_down = can_piece_move(down, g);
    if (!can_move_down)
    {
        falling_to_fixed(g);

        return 1;       // ultimate gravity turn
    }

    // save middle point
    int middle_row = g->middle_coordinate.row;
    int middle_col = g->middle_coordinate.col;

    // piece is moved down
    for (int i=g->rows-1; i >= 0; i--)
    {
        for (int j= g->cols-1; j >= 0; j--)
        {
            /* checks for falling pieces/blocks in order to move them every tick */
            bool condition =
                    g->game_board[i][j].value == 1 &&
                    g->game_board[i][j].falling_piece &&
                    !g->game_board[i][j].moved_in_prev_iteration;

            if(condition && i < g->bottom_height)
            {

                // save the color
                short color = g->game_board[i][j].color;

                // update position of the falling piece on the board
                set_block(i, j, EMPTY_CELL, false, false, NO_COLOR, g);

                // update new position and set as falling piece
                set_block(i+1, j, CELL, true, true, color, g);

                if (i == middle_row && j == middle_col)
                {
                    g->middle_coordinate.row++;
                }
            }

            else if (g->game_board[i][j].value == 1 && g->game_board[i][j].moved_in_prev_iteration)
            {
                // update if block was moved
                g->game_board[i][j].moved_in_prev_iteration = false;
            }

            else if (condition && i+1 >= g->bottom_height)
            {
                falling_to_fixed(g);
                g->need_new_piece = true;
                return 1;       // return needed for skip_tick_gravity if down arrow is pressed
            }
        }
    }

    return 0;                    // return needed for skip_tick_gravity if down arrow is pressed
}


void example_fill_board(Game* g)
{
    int rows = g->cols;
    int cols = g->cols;

    for (int i=0; i<rows; i++)
    {
        for (int j=0; j< cols; j++)
        {
            if (i == 20 && j < BOARD_EDGE_RIGHT  && j >= 0)
            {
                set_block(i, j, CELL, false, false, 4, g);
            }

            if ( i == 21 && j < BOARD_EDGE_RIGHT && j >= 0)
            {
                set_block(i, j, CELL, false, false, 5, g);
            }
        }
    }
}


void insert_falling_piece(type type, Game* g)
{
    int mid = 4;
    short color = generate_random_number(0, 7);
    Position pos;

    if (type == J)
    {
        set_block(1, mid - 1, CELL, true, false, color, g);
        set_block(1, mid    , CELL, true, false, color, g);
        set_block(1, mid + 1, CELL, true, false, color, g);
        set_block(2, mid + 1, CELL, true, false, color, g);

        // assign middle position
        pos.row = 1;
        pos.col = mid;

        // assign piece type
        g->piece_type = J;
    }

    else if (type == I)
    {
        set_block(1, mid - 1, CELL, true, false, color, g);
        set_block(1, mid    , CELL, true, false, color, g);
        set_block(1, mid + 1, CELL, true, false, color, g);
        set_block(1, mid + 2, CELL, true, false, color, g);

        // assign middle position
        pos.row = 1;
        pos.col = mid + 1;

        // assign piece type
        g->piece_type = I;
    }

    else if (type == L)
    {
        set_block(1, mid - 1, CELL, true, false, color, g);
        set_block(1, mid    , CELL, true, false, color, g);
        set_block(1, mid + 1, CELL, true, false, color, g);
        set_block(2, mid - 1, CELL, true, false, color, g);

        // assign middle position
        pos.row = 1;
        pos.col = mid ;


        // assign piece type
        g->piece_type = L;
    }

    else if (type == T)
    {
        set_block(1, mid - 1, CELL, true, false, color, g);
        set_block(1, mid    , CELL, true, false, color, g);
        set_block(1, mid + 1, CELL, true, false, color, g);
        set_block(2, mid    , CELL, true, false, color, g);

        // assign middle position
        pos.row = 1;
        pos.col = mid;


        // assign piece type
        g->piece_type = T;
    }

    else if (type == S )
    {
        set_block(1, mid    , CELL, true, false, color, g);
        set_block(1, mid + 1, CELL, true, false, color, g);
        set_block(2, mid    , CELL, true, false, color, g);
        set_block(2, mid - 1, CELL, true, false, color, g);

        // assign middle person
        pos.row = 1;
        pos.col = mid;


        // assign piece type
        g->piece_type = S;
    }

    else if (type == Z )
    {
        set_block(1, mid    , CELL, true, false, color, g);
        set_block(1, mid + 1, CELL, true, false, color, g);
        set_block(2, mid + 1, CELL, true, false, color, g);
        set_block(2, mid + 2, CELL, true, false, color, g);

        // assign middle person
        pos.row = 1;
        pos.col = mid +1;

        // assign piece type
        g->piece_type = Z;
    }

    else if (type == O)
    {
        set_block(1, mid    , CELL, true, false, color, g);
        set_block(1, mid + 1, CELL, true, false, color, g);
        set_block(2, mid    , CELL, true, false, color, g);
        set_block(2, mid + 1, CELL, true, false, color, g);

        // only temporary, should not have a middle;
        pos.row = 1;
        pos.col = mid;

        // assign piece type
        g->piece_type = O;
    }

    g->middle_coordinate = pos;
    g->piece_counter++;
}


void move_piece(direction dir, Game* g) {
    bool can_move = can_piece_move(dir, g);

    if (!can_move)
    {
        Logger("cannot move");
        return;
    }

    // traverse the g board in the direction based on the dir parameter
    if (dir == right)
    {
        for (int i = g->rows - 1; i >= 0; i--)
        {
            for (int j = g->cols - 1; j >= 0; j--)
            {
                bool condition = g->game_board[i][j].value == CELL &&
                                 g->game_board[i][j].falling_piece;

                if (condition)
                {
                    // calculate the new column index
                    int new_j = j + 1;

                    if (is_valid_block(i, new_j, g) )
                    {
                        // save the color
                        short color = g->game_board[i][j].color;
                        // delete old block
                        set_block(i, j, EMPTY_CELL, false, false, NO_COLOR, g);

                        // set new block at the updated position
                        set_block(i, new_j, CELL, true, false, color, g);

                    }
                }
            }
        }
        // update position of middle coordinate to the right
        g->middle_coordinate.col++;
    }

        // if dir == left, traverse the list the way around
    else
    {
        for (int i = 0; i < g->rows; i++)
        {
            for (int j = 0; j < g->cols; j++)
            {
                bool condition = g->game_board[i][j].value == CELL &&
                                 g->game_board[i][j].falling_piece;

                if (condition)
                {
                    // calculate the new column index
                    int new_j = j - 1;

                    if (is_valid_block(i, new_j, g))
                    {

                        // save the color
                        short color = g->game_board[i][j].color;

                        // delete old block
                        set_block(i, j, EMPTY_CELL, false, false, NO_COLOR, g);

                        // set new block at the updated position
                        set_block(i, new_j, CELL, true, false, color ,g);
                    }
                }
            }
        }
        // update position of middle coordinate to the left
        g->middle_coordinate.col--;
    }

}


bool is_valid_block(int rows, int cols, Game* g)
{
    bool condition1 = rows >= 0 && rows < g->rows - 2;
    bool condition2 = cols >= 0 && cols <= BOARD_EDGE_RIGHT;

    if (condition1 && condition2)
    { return true; }

    return false;
}


void set_block(int row, int col, int value, bool is_falling, bool moved_in_prev_iteration, short color, Game* g)
{
    g->game_board[row][col].value                        = value;
    g->game_board[row][col].color                        = color;
    if (value == EMPTY_CELL)
    {
        g->game_board[row][col].falling_piece            = false;
        g->game_board[row][col].fixed_piece              = false;
        g->game_board[row][col].moved_in_prev_iteration  = false;
    }

    else
    {
        g->game_board[row][col].falling_piece = is_falling;
        g->game_board[row][col].fixed_piece = !is_falling;
        g->game_board[row][col].moved_in_prev_iteration = moved_in_prev_iteration;
    }
}


Position block_position_after_rotation(int row, int col, direction dir, Game* g)
{
    Position pos;

    Position middle_position = g->middle_coordinate;

    // Calculate the delta from the middle coordinate
    int delta_row = row - middle_position.row;
    int delta_col = col - middle_position.col;

    // Apply rotation based on direction
    if (dir == right)
    {
        // Rotate 90 degrees clockwise
        pos.row = middle_position.row - delta_col;
        pos.col = middle_position.col + delta_row;
    }
    else if (dir == left)
    {
        // Rotate 90 degrees counterclockwise
        pos.row = middle_position.row + delta_col;
        pos.col = middle_position.col - delta_row;
    }

    return pos;
}


// rotate the piece
void rotate_piece(direction dir, Game* g)
{
    if (g->piece_type == O)
    { return; }
    // temporary copy of g board
    Block temp_board[BOARD_HEIGHT][BOARD_WIDTH];
    for(int i=0; i < g->rows; i++)
    {
        for(int j=0; j < g->cols; j++)
        {
            temp_board[i][j] = g->game_board[i][j];
            temp_board[i][j].rotated_in_prev_iteration = false;
        }
    }

    // perform rotation on copy
    for(int i=0; i < g->rows; i++)
    {
        for(int j=0; j < g->cols; j++)
        {
            bool condition = g->game_board[i][j].value == CELL
                             && g->game_board[i][j].falling_piece;

            if (condition)
            {
                // get position of each block after iteration
                Position rotated_position = block_position_after_rotation(i, j, dir, g);

                // check if the position of rotated block is valid
                bool is_valid = is_valid_block(rotated_position.row, rotated_position.col, g)
                                && is_empty_block(rotated_position.row, rotated_position.col, g);

                if (!is_valid)
                {
                    return;     // exit if it is not valid; temp board will be deleted.
                }

                // copy block from prev position
                temp_board[rotated_position.row][rotated_position.col] = g->game_board[i][j];
                temp_board[rotated_position.row][rotated_position.col].is_new = true;
                temp_board[rotated_position.row][rotated_position.col].rotated_in_prev_iteration = true;

            }
        }
    }

    // copy from temp board to real g board
    for(int i=0; i < g->rows; i++)
    {
        for(int j=0; j < g->cols; j++)
        {
            // delete all old blocks, only rotated blocks (marked with .is_new) will be copied.
            bool condition = temp_board[i][j].falling_piece && !temp_board[i][j].is_new;
            if (condition)
            {
                temp_board[i][j].value                      = EMPTY_CELL;
                temp_board[i][j].falling_piece              = false;
                temp_board[i][j].color                      = NO_COLOR;
                temp_board[i][j].moved_in_prev_iteration    = false;
                temp_board[i][j].rotated_in_prev_iteration  = false;

            }
            g->game_board[i][j] = temp_board[i][j];
            g->game_board[i][j].is_new = false;
        }
    }
}


void manage_full_lines(Game* g)
{
    for (int i= g->rows - 1; i >= 0; i--)
    {
        bool compare_value = true;
        for (int j= g->cols - 17; j >= 0; j--)   // needed -17 because of wrong calculation of blocks.
        {
            bool condition = g->game_board[i][j].value == EMPTY_CELL || g->game_board[i][j].falling_piece;

            if (condition)
            {
                compare_value = false;
                break;
            }
        }

        if (compare_value)
        {
            // increase score
            g->score++;

            // adjust difficulty after 5 points each
            if (g->score % 5 == 0)
            {
                int difficulty = g->difficulty;
                g->difficulty = (difficulty > 50) ? difficulty-=50: difficulty;
            }

            clear_line(i, g);

            // adjust all blocks from above
            adjust_blocks(i, g);
        }
    }
}


void clear_line(int row, Game* g)
{
    for (int col=0; col < g->cols; col++)
    {
        set_block(row, col, EMPTY_CELL, false, false, NO_COLOR, g);
    }
}


void adjust_blocks(int row, Game* g)
{
    for (int i=row; i>=0; i--)
    {
        for (int j=g->cols; j>=0; j--)
        {
            bool condition = g->game_board[i+1][j].value == EMPTY_CELL &&
                             g->game_board[i][j].fixed_piece;

            if (condition)
            {
                // copy block to one block down (i+1)
                g->game_board[i+1][j] = g->game_board[i][j];

                // remove old block
                set_block(i, j, EMPTY_CELL, false, false, NO_COLOR, g);
            }
        }
    }
}


void skip_tick_gravity(Game* g)
{
    int status_gravity = gravity(g);
    while (status_gravity != 1)
    {
        status_gravity = gravity(g);
    }
}


// check for highest current block
int check_game_state(Game* g)
{
    // check if highest current block reaches top -> game-over
    if (g->highest_fixed_block >= BOARD_HEIGHT-1)
    {
        return 1;
    }
    return 0;
}


/// converts all falling blocks to static/fixed blocks.
void falling_to_fixed(Game* g)
{
    for (int i=0; i < g->rows; i++)
    {
        for (int j=0; j < g->cols; j++)
        {
            bool condition = g->game_board[i][j].value == CELL &&
                             g->game_board[i][j].falling_piece;

            if (condition)
            {
                // save the color
                short color = g->game_board[i][j].color;

                // set to falling piece
                set_block(i, j, CELL, false, false, color, g);
                int height = BOARD_HEIGHT - i;

                bool condition_height = height > g->highest_fixed_block;

                if (condition_height)
                {
                    // update the highest block in the g
                    g->highest_fixed_block = height;
                }
            }
        }
    }
    // we need a new piece. update state variable
    g->need_new_piece = true;
}


bool can_piece_move(direction dir, Game* g)
{
    for (int i = 0; i < g->rows; i++)
    {
        for (int j = 0; j < g->cols; j++)
        {
            bool condition = g->game_board[i][j].value == CELL &&
                             g->game_board[i][j].falling_piece;

            if (condition)
            {
                int new_i = i;
                int new_j = j;

                if (dir == left)
                {
                    new_j = j - 1;
                }

                else if (dir == right)
                {
                    new_j = j + 1;
                }

                else if (dir == down)
                {
                    new_i = i + 1;
                }
                bool valid_block = is_valid_block(new_i, new_j, g);
                bool empty_block = is_empty_block(new_i, new_j ,g);

                

                // check if the new position is valid and not colliding with other pieces
                if (!is_valid_block(new_i, new_j, g) || !is_empty_block(new_i, new_j, g))
                {
                    //Logger("Checking move not valid: current=(" + std::to_string(i) + "," + std::to_string(j) + 
                    // ") new=(" + std::to_string(new_i) + "," + std::to_string(new_j) + ")");
                    //Logger("is_valid_block: " + std::to_string(valid_block));
                    //Logger("is_empty_block: " + std::to_string(empty_block));
                    return false; // collision detected
                }
            }
        }
    }
    return true; // no collision detected
}


bool is_empty_block(int row, int col, Game* g)
{
    bool condition = !g->game_board[row][col].fixed_piece;
    if (condition)
    {
        return true;
    }
    return false;
}

