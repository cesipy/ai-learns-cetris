#include "tetris.h"

void initialize_game(Game* g)
{
    init_colors();      // initialize color pairs

    WINDOW* board = newwin(g->rows, g->cols, 0, 0);
    g->win = board;

    game_init(g, BOARD_HEIGHT, BOARD_WIDTH); // initialize the board struct and all its members
}


void main_loop(Game* g)
{
    int tick = 0;   // used for gravity rate
    int status;

    while (g->running)
    {
        // when a new piece is needed a new piece of random type is generated
        if (g->need_new_piece)
        {
            int random_piece = generate_random_number(0, AMOUNT_OF_PIECES-1);
            insert_falling_piece(static_cast<type>(random_piece), g);

            g->need_new_piece = false;
        }

        // check for input (q, arrow up, down, right, left)
        status = check_input(g);

        if (status == 0) { break; }

        manage_full_lines(g);
        display_board(g);

        display_score(g);

        // update position if a falling piece aka gravity
        // difficulty gets updated in manage_full_lines()
        if (tick % g->difficulty == 0) {
            gravity(g);
        }

        doupdate();             // update all windows
        usleep(SLEEP_TIME);     // sleep for a bit
        tick++;
        check_game_state(g);

    }
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

    for(int i=0; i < g->rows; i++)
    {
        wmove(win, i +1, 1);
        for (int j = 0; j < g->cols; j++)
        {

            if (g->game_board[i][j].value != EMPTY_CELL)
            {
                // draw block with saved color
                ADD_BLOCK(win, g->game_board[i][j].color);
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


int check_input(Game* g)
{
    int input = getch();
    // check for 'q' to quit the g
    if (input=='q') {return 0;}

    // handle input
    switch (input)
    {
        case KEY_LEFT:
            // move left
            move_piece(left, g);
            break;
        case KEY_RIGHT:
            // move right
            move_piece(right, g);
            break;
        case KEY_UP:
            // rotate
            rotate_piece(DIRECTION, g);
            break;
        case KEY_DOWN:
            skip_tick_gravity(g);
            break;
        default:
            // no key/ other key received
            break;
    }
    return 1;
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
    g->difficulty          = GRAVITY_TICKS;

    // further implementation

    // set up named pipes
    Communication* communication = new Communication;
    communication->fifo_control_name = "fifo_controls";
    communication->fifo_states_name  = "fifo_states";

    communication->fd_controls = setup_named_pipe(communication->fifo_control_name, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_RDWR);
    communication->fd_states   = setup_named_pipe(communication->fifo_states_name, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_RDWR );
  
    g->communication = communication;
    for (int i=0; i<g->rows;i++)
    {
        for (int j=0; j < g->cols;j++)
        {
            // fill game board with empty cells at start -> '0' is emtpy

            set_block(i, j, EMPTY_CELL, false, false, NO_COLOR, g);

            g->game_board[i][j].rotated_in_prev_iteration = false;
            g->game_board[i][j].is_new = false;                // temp, make more beautiful!
        }
    }
}




int generate_random_number(int min, int max)
{

    std::random_device rd;
    std::mt19937 gen(rd());


    std::uniform_int_distribution<int> distribution(min, max);
    int random_number = distribution(gen);

    return random_number;
}


void display_score(Game* g)
{
    mvwprintw(g->win, 0, 0, "Score: %d", g->score);
    wrefresh(g->win);
}