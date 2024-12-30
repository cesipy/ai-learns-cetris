#include "tetris.hpp"
#include <thread>

/*  ------------------------------  */


int main (int argc, char* argv[])
{
    Logger("starting tetris code");
    initscr();
    noecho();
    resize_term(BOARD_HEIGHT,  BOARD_WIDTH);
    timeout(0);
    curs_set(0);
    keypad(stdscr, TRUE);       // allow  arrow keys

    //start_color();

    // set up communication between c and python
    Communication* communication = new Communication;
    communication->fifo_control_name = "fifo_controls";
    communication->fifo_states_name  = "./fifo_states";

    Logger("before setting up pipes");
    communication->fd_controls = setup_named_pipe(communication->fifo_control_name, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_RDONLY);
    communication->fd_states = setup_named_pipe(communication->fifo_states_name, (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH), O_WRONLY);
    
    Logger("named pipes set up");

    

    // small delay after setting up, because otherwise there is pa problem with python communication and it is stuck in a deadlock
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // send handshake message to python. establish connection
    // receive iterations. specifies the number of times the game should be repeated
    int iterations = handshake(communication);

    Logger("handshake finished, iterations: " + std::to_string(iterations));

    
    while (iterations > 0) {

        Game* game = new Game;    // alloc memory
        game->communication = communication;

        initialize_game(game);
        Logger("game initialized");

        int status = main_loop(game);

        Logger("main loop finished");

        // TODO: end of episode, notify python via pipe
        end_of_game_notify(communication);
        Logger("end of episode noticifaction sent");

        delete game;
        Logger("game deleted");

        
        // sleep for a bit after each game to let communication sync
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        Logger("sleeped for 500 milliseconds");


        if (status == EARLY_QUIT)
        {
            Logger("early quitting");
            iterations = 1;
        }
        iterations--;

        // reset the terminal
        endwin();
        refresh();
        initscr();
        noecho();
        resize_term(BOARD_HEIGHT, BOARD_WIDTH);
        timeout(0);
        curs_set(0);
        keypad(stdscr, TRUE);
    }

    // send closing message to fifo_states
    write(communication->fd_states, "end", strlen("end"));

    clean_up_named_pipes(communication);
    delete communication;
    endwin();
   
    return EXIT_SUCCESS;
}

/*  ------------------------------  */

