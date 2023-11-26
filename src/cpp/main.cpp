#include "tetris.hpp"


/*  ------------------------------  */


int main (int argc, char* argv[])
{
    initscr();
    noecho();
    resize_term(BOARD_HEIGHT,  BOARD_WIDTH);
    timeout(0);
    curs_set(0);
    keypad(stdscr, TRUE);       // allow  arrow keys

    // set up communication between c and python
    Communication* communication = new Communication;
    communication->fifo_control_name = "fifo_controls";
    communication->fifo_states_name  = "fifo_states";

    communication->fd_controls = setup_named_pipe(communication->fifo_control_name, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_RDWR);
    communication->fd_states   = setup_named_pipe(communication->fifo_states_name, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_RDWR );

    // send handshake message to python. establish connection
    // receive iterations. specifies the number of times the game should be repeated
    int iterations = handshake(communication);

    while (iterations > 0) {
        Game* game = new Game;    // alloc memory
        game->communication = communication;

        initialize_game(game);
        // example_fill_board(game);
        main_loop(game);
        delete game;

        iterations--;
    }

    // send closing message to fifo_states
    write(communication->fd_states, "end", strlen("end"));

    clean_up_named_pipes(communication);
    delete communication;
    endwin();
   
    return EXIT_SUCCESS;
}

/*  ------------------------------  */

