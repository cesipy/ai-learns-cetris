#include "tetris.h"

/*  ------------------------------  */

int main (int argc, char* argv[])
{
    initscr();
    noecho();
    resize_term(BOARD_HEIGHT,  BOARD_WIDTH);
    timeout(0);
    curs_set(0);
    keypad(stdscr, TRUE);       // allow for arrow keys

    Game* game = new Game;    // alloc memory
    initialize_game(game);
    // example_fill_board(game);
    main_loop(game);

    endwin();
    // free allocated objects
    delete game;

}

/*  ------------------------------  */

