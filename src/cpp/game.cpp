#include "tetris.hpp"

void initialize_game(Game* g)
{
    init_colors();      // initialize color pairs

    WINDOW* board = newwin(g->rows, g->cols, 0, 0);
    g->win = board;

    game_init(g, BOARD_HEIGHT, BOARD_WIDTH); // initialize the board struct and all its members
}


int main_loop(Game* g)
{
    Logger("start another episonde in main loop");
    int tick = 0;                       // used for gravity rate
    int status;

    // is 'fake' control message?
    bool proper_state = true;

    while (g->running)
    {
        // when a new piece is needed a new piece of random type is generated
        if (g->need_new_piece)
        {
            int piece_type;
            if (DETERMINISTIC)
            {
                piece_type = static_cast<type>(g->piece_counter % AMOUNT_OF_PIECES);
            }
            else 
            {
                piece_type = generate_random_number(0, AMOUNT_OF_PIECES-1);
            }
            insert_falling_piece(static_cast<type>(piece_type), g);
            g->need_new_piece = false;
        }

        process_control(g);
       
        // check for input (q, arrow up, down, right, left)
        status = check_input(g);

        if (status == 0) 
        { 
            g->running = false;
            return EARLY_QUIT;      // to quit program before playing multiple iterations
        }

        manage_full_lines(g);
        display_board(g);

        display_score(g);

        // update position if a falling piece aka gravity
        // difficulty gets updated in manage_full_lines(    )
        if (tick % GRAVITY_TICKS == 0) {                        // no difficulty increasement
            gravity(g);
        }

        doupdate();             // update all windows
        usleep(SLEEP_TIME);     // sleep for a bit
        tick++;

        int game_state = check_game_state(g);
        if (game_state) 
        {
            g->running = false;
        }

        if (tick % GRAVITY_TICKS == 0) 
        {
            communicate(g);   // maybe put this on beginning. when tetris
                              // terminates, fifo_controls is called one more time
            if (!proper_state)
            {
                // new 'fake' control is ignored
                g->control->new_control_available = false;
                proper_state = true;
                //Logger("ignoring control");
            }
            else 
            {
                proper_state = false;
            }
        }
    }

    //manage_full_lines(g); // TODO: just testing, maybe this destroys something here
    communicate(g);
    return NORMAL_QUIT;
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
    g->piece_counter       = 0;

    // further implementation
    State* state     = new State;
    Control* control = new Control;

    state->bumpiness     = 0;
    state->height        = 0;
    state->holes         = 0;
    state->lines_cleared = 0;

    control->new_control_available = false;

    // assign control and state to game
    g->state         = state;
    g->control       = control;

    // TODO: decouple pipes from a single game, should be global ( in main.cpp)
    // set up named pipes
   
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