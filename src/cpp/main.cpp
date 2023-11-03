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

    Game* game = new Game;    // alloc memory
    initialize_game(game);
    // example_fill_board(game);
    main_loop(game);

      // send closing message to fifo_states
    write(game->communication->fd_states, "end", strlen("end"));

    endwin();
    // free allocated objects
    delete game;

     // close named pipe, make this in own cleanup function!
    close(game->communication->fd_controls);
    close(game->communication->fd_states);
    unlink(game->communication->fifo_control_name);
    unlink(game->communication->fifo_states_name);

}

/*  ------------------------------  */

