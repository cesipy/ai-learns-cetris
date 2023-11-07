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
  
    clean_up_named_pipes(game);
    // free allocated objects
    delete game;

}

/*  ------------------------------  */

