#include "tetris.hpp"

void handshake(Communication* communication) 
{
    std::string handshake_message = "handshake";
    write(communication->fd_states, handshake_message.c_str(), strlen(handshake_message.c_str()));
    Logger("sent handshake message");

    int fd = communication->fd_controls;
    
    char buffer[100];
    ssize_t bytesRead;

    // read data from the named pipe
    bytesRead = read(fd, buffer, sizeof(buffer) - 1);

    if (bytesRead > 0) 
    {
        // null-terminate the received data to make it a string
        buffer[bytesRead] = '\0';
       
        int iterations = std::stoi(buffer);
        Logger("received iterations from handshake: " + std::to_string(iterations));

    } 
    else if (bytesRead == 0) { return; } 
    else 
    {
        perror("read");
        exit(EXIT_FAILURE);
    }
}

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

    handshake(communication);


    Game* game = new Game;    // alloc memory
    game->communication = communication;

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

