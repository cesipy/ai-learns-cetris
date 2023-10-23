#include <ncurses.h>
#include "tetris.h"
#include <unistd.h>
#include <iostream>
#include <random>

WINDOW* board, * falling, *hold, *score;
Game* game        = new Game ;
State* state      = new State;
int piece_counter = 0;
Control* control  = new Control ;

//named pipe
int fd_states, fd_controls;                 // named pipe
const char* fifo_states  = "fifo_states";
const char* fifo_control = "fifo_controls";

// seed for random function
std::random_device rd;
std::mt19937 gen(rd());


/*  ------------------------------  */

int main (int argc, char* argv[])
{
    //game = new Game;    // alloc memory
    game_init(game, BOARD_HEIGHT, BOARD_WIDTH); // initialize the board struct and all its members

    initscr();
    init_colors();
    noecho();
    resize_term(BOARD_HEIGHT,  BOARD_WIDTH);
    timeout(0);
    curs_set(0);
    keypad(stdscr, TRUE);       // allow for arrow keys

    board = newwin(game->rows, game->cols, 0, 0);
    game->win = board;


    // example_fill_board(game);
    main_loop();

    // send closing message to fifo_states
    write(fd_states, "end", strlen("end"));

    // free allocated objects
    delete game;
    delete state;
    delete control;
    endwin();

    // close named pipe
    close(fd_controls);
    close(fd_states);
    unlink(fifo_control);
    unlink(fifo_states);

}

/*  ------------------------------  */

void main_loop()
{
    int tick = 0;   // used for gravity rate

    while (game->running)
    {
        // when a new piece is needed a new piece of random type is generated
        if (game->need_new_piece)
        {
            int random_piece = generate_random_number(0, AMOUNT_OF_PIECES-1);
            insert_falling_piece(static_cast<type>(random_piece), game);
            piece_counter_increase();

            game->need_new_piece = false;
        }

        if (control->new_control_available)
        {
            control->new_control_available = false;

            int delta_position = control->new_position;

            if (delta_position < 0)
            {
                while (delta_position < 0)
                {
                    move_piece(left);
                    delta_position++;
                }
            }
            else if (delta_position > 0)
            {
                while (delta_position > 0)
                {
                    move_piece(right);
                    delta_position--;
                }
            }
        }

        // only temp
        if (tick % (GRAVITY_TICKS) == 0)
        {
            // update state for current processing
            update_state(game, state);

        }

        int input = getch();
        // check for 'q' to quit the game
        if (input=='q') {break;}

        // handle input
        switch (input)
        {
            case KEY_LEFT:
                // Move Tetromino left
                move_piece(left);
                break;
            case KEY_RIGHT:
                // Move Tetromino right
                move_piece(right);
                break;
            case KEY_UP:
                // rotate
                rotate_piece(DIRECTION);
                break;
            case KEY_DOWN:
                skip_tick_gravity();
                // Move Tetromino down faster (if desired)
                break;
            default:
                break;
        }
        manage_full_lines();
        display_board(game);

        display_score();

        // update position if a falling piece aka gravity
        if (tick % GRAVITY_TICKS == 0) {
            gravity(game);
        }

        doupdate();             // update all windows
        usleep(SLEEP_TIME);     // sleep for a bit
        tick++;
        check_game_state();


        if (tick % (GRAVITY_TICKS) == 0) {
            char *message = state_to_string(state);
            write(fd_states, message, strlen(message));

            receive_message(fd_controls,control );       // receive new control instructions via named pipe
        }
    }
}


int hit_bottom()
{
    return 0;
}


void init_colors()
{
    start_color();
    for (int i = 1; i < 8; i++) 
    {
        init_pair(i, i, 0);
    }
    // Define other color pairs
}


void display_board(Game* g)
{
    WINDOW* win = g->win;

    // temp for debugging
    Position middle = game->middle_coordinate;

    for(int i=0; i < g->rows; i++)
    {
        wmove(win, i +1, 1);
        for (int j = 0; j < g->cols; j++)
        {
            // debugging: show middle coordinate
            if (g->game_board[i][j].value != EMPTY_CELL)
            {
                // draw block with saved color
                ADD_BLOCK(win, game->game_board[i][j].color);
            }

            else
            {
                // draw empty block
                ADD_EMPTY(win);
            }
        }
    }

    box(win, 0, 0);
    wnoutrefresh(win);
}


int gravity(Game* g)
{
    // check if the blocks below are free
    bool can_move_down = can_piece_move(down);
    if (!can_move_down)
    {
        falling_to_fixed();

        return 1;       // ultimate gravity turn
    }

    // save middle point
    int middle_row = game->middle_coordinate.row;
    int middle_col = game->middle_coordinate.col;

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

            if(condition && i < game->bottom_height)
            {

                // save the color
                short color = game->game_board[i][j].color;

                // update position of the falling piece on the board
                set_block(i, j, EMPTY_CELL, false, false, 8 );

                // update new position and set as falling piece
                set_block(i+1, j, CELL, true, true, color);

                if (i == middle_row && j == middle_col)
                {
                    game->middle_coordinate.row++;
                }
            }

            else if (g->game_board[i][j].value == 1 && g->game_board[i][j].moved_in_prev_iteration)
            {
                // update if block was moved
                g->game_board[i][j].moved_in_prev_iteration = false;
            }

            else if (condition && i+1 >= game->bottom_height)
            {
                falling_to_fixed();
                game->need_new_piece = true;
                return 1;       // return needed for skip_tick_gravity if down arrow is pressed
            }
        }
    }


    return 0;                    // return needed for skip_tick_gravity if down arrow is pressed
}


void game_init(Game* g, int rows, int cols)
{
    // Position for middle coordinate
    Position position = {.row=0, .col=0};

    g->rows                = rows;
    g->cols                = cols;
    g->running             = true;
    g->bottom_height       = BOARD_HEIGHT - 2;
    g->need_new_piece      = true;               // start with a new falling piece
    g->highest_fixed_block = 0;
    g->middle_coordinate   = position;
    g->score               = 0;
    g->piece_type          = initial;

    control->new_control_available = false;

    // further implementation

    // set up named pipe
    fd_states    = setup_named_pipe(fifo_states, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_RDWR );
    fd_controls  = setup_named_pipe(fifo_control, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_RDWR);

    for (int i=0; i<g->rows;i++)
    {
        for (int j=0; j < g->cols;j++)
        {
            // fill game board with empty cells at start -> '0' is emtpy

            set_block(i, j, EMPTY_CELL, false, false, NO_COLOR);

            game->game_board[i][j].rotated_in_prev_iteration = false;
            game->game_board[i][j].is_new = false;                // temp, make more beautiful!
        }
    }

    // for debugging
    //example_fill_board(game);
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
                set_block(i, j, CELL, false, false, 4);
            }

            if ( i == 21 && j < BOARD_EDGE_RIGHT && j >= 0)
            {
                set_block(i, j, CELL, false, false, 5);
            }
        }
    }
}


void insert_falling_piece(type type, Game* g)
{
    int mid = 8;
    short color = generate_random_number(0, 7);
    Position pos;

    if (type == J)
    {
        set_block(1, mid - 1, CELL, true, false, color);
        set_block(1, mid    , CELL, true, false, color);
        set_block(1, mid + 1, CELL, true, false, color);
        set_block(2, mid + 1, CELL, true, false, color);

        // assign middle position
        pos.row = 1;
        pos.col = mid;

        // assign piece type
        game->piece_type = J;
    }

    else if (type == I)
    {
        set_block(1, mid - 1, CELL, true, false, color);
        set_block(1, mid    , CELL, true, false, color);
        set_block(1, mid + 1, CELL, true, false, color);
        set_block(1, mid + 2, CELL, true, false, color);

        // assign middle position
        pos.row = 1;
        pos.col = mid + 1;

        // assign piece type
        game->piece_type = I;
    }

    else if (type == L)
    {
        set_block(1, mid - 1, CELL, true, false, color);
        set_block(1, mid    , CELL, true, false, color);
        set_block(1, mid + 1, CELL, true, false, color);
        set_block(2, mid - 1, CELL, true, false, color);

        // assign middle position
        pos.row = 1;
        pos.col = mid ;


        // assign piece type
        game->piece_type = L;
    }

    else if (type == T)
    {
        set_block(1, mid - 1, CELL, true, false, color);
        set_block(1, mid    , CELL, true, false, color);
        set_block(1, mid + 1, CELL, true, false, color);
        set_block(2, mid    , CELL, true, false, color);

        // assign middle position
        pos.row = 1;
        pos.col = mid;


        // assign piece type
        game->piece_type = T;
    }

    else if (type == S )
    {
        set_block(1, mid    , CELL, true, false, color);
        set_block(1, mid + 1, CELL, true, false, color);
        set_block(2, mid    , CELL, true, false, color);
        set_block(2, mid - 1, CELL, true, false, color);

        // assign middle person
        pos.row = 1;
        pos.col = mid;


        // assign piece type
        game->piece_type = S;
    }

    else if (type == Z )
    {
        set_block(1, mid    , CELL, true, false, color);
        set_block(1, mid + 1, CELL, true, false, color);
        set_block(2, mid + 1, CELL, true, false, color);
        set_block(2, mid + 2, CELL, true, false, color);

        // assign middle person
        pos.row = 1;
        pos.col = mid +1;

        // assign piece type
        game->piece_type = Z;
    }

    else if (type == O)
    {
        set_block(1, mid    , CELL, true, false, color);
        set_block(1, mid + 1, CELL, true, false, color);
        set_block(2, mid    , CELL, true, false, color);
        set_block(2, mid + 1, CELL, true, false, color);

        // only temporary, should not have a middle;
        pos.row = 1;
        pos.col = mid;

        // assign piece type
        game->piece_type = O;
    }

    game->middle_coordinate = pos;
}


void move_piece(direction dir) {
    bool can_move = can_piece_move(dir);

    if (!can_move)
    {
        return;
    }

    // traverse the game board in the direction based on the dir parameter
    if (dir == right)
    {
        for (int i = game->rows -1; i >= 0; i--)
        {
            for (int j = game->cols - 1; j >= 0; j--)
            {
                bool condition = game->game_board[i][j].value == CELL &&
                                 game->game_board[i][j].falling_piece;

                if (condition)
                {
                    // calculate the new column index
                    int new_j = j + 1;

                    if (is_valid_block(i, new_j) )
                    {
                        // save the color
                        short color = game->game_board[i][j].color;
                        // delete old block
                        set_block(i, j, EMPTY_CELL, false, false, 8);

                        // set new block at the updated position
                        set_block(i, new_j, CELL, true, false, color);

                    }
                }
            }
        }
        // update position of middle coordinate to the right
        game->middle_coordinate.col++;
    }

    // if dir == left, traverse the list the way around
    else
    {
        for (int i = 0; i < game->rows; i++)
        {
            for (int j = 0; j < game->cols; j++)
            {
                bool condition = game->game_board[i][j].value == CELL &&
                                 game->game_board[i][j].falling_piece;

                if (condition)
                {
                    // calculate the new column index
                    int new_j = j - 1;

                    if (is_valid_block(i, new_j))
                    {

                        // save the color
                        short color = game->game_board[i][j].color;

                        // delete old block
                        set_block(i, j, EMPTY_CELL, false, false, 8);

                        // set new block at the updated position
                        set_block(i, new_j, CELL, true, false, color);
                    }
                }
            }
        }
        // update position of middle coordinate to the left
        game->middle_coordinate.col--;
    }

}


bool is_valid_block(int rows, int cols)
{
    bool condition1 = rows >= 0 && rows < game->rows - 2;
    bool condition2 = cols >= 0 && cols <= BOARD_EDGE_RIGHT;

    if (condition1 && condition2)
    { return true; }

    return false;
}


void set_block(int row, int col, int value, bool is_falling, bool moved_in_prev_iteration, short color)
{
    game->game_board[row][col].value                        = value;
    game->game_board[row][col].color                        = color;
    if (value == EMPTY_CELL)
    {
        game->game_board[row][col].falling_piece            = false;
        game->game_board[row][col].fixed_piece              = false;
        game->game_board[row][col].moved_in_prev_iteration  = false;
    }

    else
    {
        game->game_board[row][col].falling_piece = is_falling;
        game->game_board[row][col].fixed_piece = !is_falling;
        game->game_board[row][col].moved_in_prev_iteration = moved_in_prev_iteration;
    }
}


void piece_counter_increase()
{
    piece_counter++;
    piece_counter = piece_counter % (AMOUNT_OF_PIECES + 1);
}

/// converts all falling blocks to static/fixed blocks.
void falling_to_fixed()
{
    for (int i=0; i<game->rows;i++)
    {
        for (int j=0; j<game->cols;j++)
        {
            bool condition = game->game_board[i][j].value == CELL &&
                    game->game_board[i][j].falling_piece;

            if (condition)
            {
                // save the color
                short color = game->game_board[i][j].color;

                // set to falling piece
                set_block(i, j, CELL, false, false, color);
                int height = BOARD_HEIGHT - i;

                bool condition_height = height > game->highest_fixed_block;

                if (condition_height)
                {
                    // update the highest block in the game
                    game->highest_fixed_block = height;
                }
            }
        }
    }
    // we need a new piece. update state variable
    game->need_new_piece = true;
}


bool can_piece_move(direction dir)
{
    for (int i = 0; i < game->rows; i++)
    {
        for (int j = 0; j < game->cols; j++)
        {
            bool condition = game->game_board[i][j].value == CELL &&
                             game->game_board[i][j].falling_piece;

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
                bool valid_block = is_valid_block(new_i, new_j);

                bool empty_block = is_empty_block(new_i, new_j);

                // check if the new position is valid and not colliding with other pieces
                if (!is_valid_block(new_i, new_j) || !is_empty_block(new_i, new_j))
                {
                    return false; // collision detected
                }
            }
        }
    }
    return true; // no collision detected
}


bool is_empty_block(int row, int col)
{
    bool condition = !game->game_board[row][col].fixed_piece;
    if (condition)
    {
        return true;
    }
    return false;
}


void skip_tick_gravity()
{
    int status_gravity = gravity(game);
    while (status_gravity != 1)
    {
        status_gravity = gravity(game);
    }
}


// check for highest current block
void check_game_state(void)
{
    // check if highest current block reaches top -> game-over
    if (game->highest_fixed_block >= BOARD_HEIGHT-1)
    {
        game->running = false;
    }
}

int generate_random_number(int min, int max)
{
    std::uniform_int_distribution<int> distribution(min, max);
    int random_number = distribution(gen);

    return random_number;
}


Position block_position_after_rotation(int row, int col, direction dir)
{
    Position pos;

    Position middle_position = game->middle_coordinate;

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
void rotate_piece(direction dir)
{
    if (game->piece_type == O)
    { return; }
    // temporary copy of game board
    Block temp_board[BOARD_HEIGHT][BOARD_WIDTH];
    for(int i=0; i<game->rows; i++)
    {
        for(int j=0; j<game->cols; j++)
        {
            temp_board[i][j] = game->game_board[i][j];
            temp_board[i][j].rotated_in_prev_iteration = false;
        }
    }

    // perform rotation on copy
    for(int i=0; i<game->rows; i++)
    {
        for(int j=0; j<game->cols; j++)
        {
            bool condition = game->game_board[i][j].value == CELL
                    && game->game_board[i][j].falling_piece;

            if (condition)
            {
                // get position of each block after iteration
                Position rotated_position = block_position_after_rotation(i, j, dir);

                // check if the position of rotated block is valid
                bool is_valid = is_valid_block(rotated_position.row, rotated_position.col)
                        && is_empty_block(rotated_position.row, rotated_position.col);

                if (!is_valid)
                {
                    return;     // exit if it is not valid; temp board will be deleted.
                }

                // copy block from prev position
                temp_board[rotated_position.row][rotated_position.col] = game->game_board[i][j];
                temp_board[rotated_position.row][rotated_position.col].is_new = true;
                temp_board[rotated_position.row][rotated_position.col].rotated_in_prev_iteration = true;

            }
        }
    }

    // copy from temp board to real game board
    for(int i=0; i<game->rows; i++)
    {
        for(int j=0; j<game->rows; j++)
        {
            // delete all old blocks, only rotated blocks (marked with .is_new) will be copied.
            bool condition = temp_board[i][j].falling_piece && !temp_board[i][j].is_new;
            if (condition)
            {
                temp_board[i][j].value                      = EMPTY_CELL;
                temp_board[i][j].falling_piece              = false;
                temp_board[i][j].color                      = 8;
                temp_board[i][j].moved_in_prev_iteration    = false;
                temp_board[i][j].rotated_in_prev_iteration  = false;

            }
            game->game_board[i][j] = temp_board[i][j];
            game->game_board[i][j].is_new = false;
        }
    }
}


void manage_full_lines()
{
    for (int i=game->rows-1; i >= 0; i--)
    {
        bool compare_value = true;
        for (int j=game->cols-17; j>= 0; j--)   // needed -17 because of wrong calculation of blocks.
        {
            bool condition = game->game_board[i][j].value == EMPTY_CELL || game->game_board[i][j].falling_piece;

            if (condition)
            {
                compare_value = false;
                break;
            }
        }

        if (compare_value)
        {
            // increase score
            game->score++;

            clear_line(i);

            // adjust all blocks from above
            adjust_blocks(i);
        }
    }
}


void clear_line(int row)
{
    for (int col=0; col < game->cols; col++)
    {
        set_block(row, col, EMPTY_CELL, false, false, 8);
    }
}


void adjust_blocks(int row)
{
    for (int i=row; i>=0; i--)
    {
        for (int j=game->cols; j>=0; j--)
        {
            bool condition = game->game_board[i+1][j].value == EMPTY_CELL &&
                    game->game_board[i][j].fixed_piece;

            if (condition)
            {
                // copy block to one block down (i+1)
                game->game_board[i+1][j] = game->game_board[i][j];

                // remove old block
                set_block(i, j, EMPTY_CELL, false, false, 8);
            }
        }
    }
}


void display_score()
{
    mvwprintw(game->win, 0, 0, "Score: %d", game->score);
    wrefresh(game->win);
}

