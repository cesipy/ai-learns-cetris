#include "tetris.hpp"


void calculate_lines_cleared(Game* g)
{
    g->state->lines_cleared = g->score;
}


void calculate_height(Game* g)
{
    int height = 0;
    for(int i=0; i< g->rows; i++)
    {
        for (int j=0; j<g->cols; j++)
        {
            if (g->game_board[i][j].fixed_piece)
            {
                height++;
            }
        }
    }
    g->state->height = height;
}



void calculate_holes(Game* g)
{
    int number_holes = 0;

    for (int col=0; col < 15; col++)       
    {
        bool encountered_fixed_piece = false;

        // loop through column

        for (int row=0; row < 28; row++)
        {
            Block block = g->game_board[row][col];

            if (block.fixed_piece)
            {
                encountered_fixed_piece = true;
            }
            bool condition = encountered_fixed_piece
                    && block.value == EMPTY_CELL;

            if (condition)
            {
                number_holes++;
            }
        }
    }
    g->state->holes = number_holes;
}


void calculate_bumpiness(Game* g)
{
    int bumpiness = 0;
    int highest_point_in_a, highest_point_in_b;

    // get the highest position in first column
    for(int i=0; i<g->rows;i++)
    {
        if (g->game_board[i][0].fixed_piece)
        {
            highest_point_in_a = i;
            break;
        }
    }

    for (int j=1; j<15; j++)        // width = 14, so <15
    {
        for (int i=0; i<g->rows; i++)
        {
            if (g->game_board[i][j].fixed_piece)
            {
                highest_point_in_b = i;

                // calculate difference
                int delta = highest_point_in_a - highest_point_in_b;
                delta = delta < 0 ? -delta : delta;

                //add to overall bumpiness:
                bumpiness += delta;

                highest_point_in_a = highest_point_in_b;
                break;
            }
        }
    }
    // save to state
    g->state->bumpiness = bumpiness;
}



void update_state(Game* g)
{
    char buffer[1024];
    char* ptr = buffer;
    int remaining = sizeof(buffer);

    // include lines cleared
    *ptr++ = g->state->lines_cleared + '0';     // convert to char
    remaining--;
    *ptr++ = ',';
    remaining--;


    // serialize game board
    for (int i=0; i<g->rows-2; i++)             // TODO: find out magic numbers and replace with dynamic way
    {
        for (int j=0; j < g->cols -16; j++)     // TODO: find out magic numbers and replace with dynamic way
        {
            if (remaining <= 1) 
            {
                break; // ensure space for null termonation
            }
            if (g->game_board[i][j].fixed_piece)
            {
                *ptr++ = '1';
            }
            else if (g->game_board[i][j].falling_piece)
            {
                *ptr++ = '2';
            }
            else 
            {
            // no block
                *ptr++ = '0';
            }
            remaining--;
        }
        if (remaining <= 1) 
        {
            break;
        }
        *ptr++ = ',';
        remaining--;
    }
    *ptr = '\0';

    Logger(buffer);

    calculate_lines_cleared(g);
    calculate_height(g);
    calculate_bumpiness(g);
    calculate_holes(g);

    // save piece type to state
    strncpy(g->state->game_state, buffer, sizeof(g->state->game_state) - 1);
    g->state->game_state[sizeof(g->state->game_state) - 1] = '\0';

    Logger(g->state->game_state);
}


char* state_to_string(const State* s) {
    std::ostringstream output;

    // Format the struct values into a string
    output << "Lines Cleared: " << s->lines_cleared
           << ", Height: " << s->height
           << ", Holes: " << s->holes
           << ", Bumpiness: " << s->bumpiness
           << ", Piece Type: " << s->piece_type << "\n";

    // Create a copy of the string and return its c_str()
    std::string str = output.str();
    char* result = new char[str.length() + 1]; // +1 for null-terminator
    strcpy(result, str.c_str());

    return result;
}


void communicate(Game* g)
{
    update_state(g);

    //char* state_string = state_to_string(g->state);
    char* state_string = g->state->game_state;
    write(g->communication->fd_states, state_string, strlen(state_string));

    // temporarily write state to logger
    Logger(state_string);
    // save control struct
    receive_message(g);
}


const int setup_named_pipe(const char* name, mode_t permission, int mode)
{
    if (mkfifo(name, permission) != 0)
    {
        // add error handling
        fprintf(stderr, "error @ mkfifo");
        exit(EXIT_FAILURE);
    }


    const int fd = open(name, mode);
    if (fd < 0)
    {
        fprintf(stderr, "error @ opening fifo");
        exit(EXIT_FAILURE);
    }

    return fd;
}


void receive_message(Game* g)
{
    int fd = g->communication->fd_controls;

    char buffer[100];
    ssize_t bytesRead;

    // read data from the named pipe
    bytesRead = read(fd, buffer, sizeof(buffer) - 1);

    if (bytesRead > 0)
    {
        // null-terminate the received data to make it a string
        buffer[bytesRead] = '\0';

        parse_message(buffer, g->control);
    }
    else if (bytesRead == 0) { return; }
    else
    {
        perror("read");
    }
}


void process_control(Game* g)
{
    if (g->control->new_control_available)
        {
            // update state, no new control is available
            g->control->new_control_available = false;

            int new_relative_position = g->control->new_position;

            while (new_relative_position > 0)
            {
                move_piece(left, g);
                new_relative_position--;
            }

            while ( new_relative_position < 0)
            {
                move_piece(right, g);
                new_relative_position++;
            }

            if (g->control->should_rotate)
            {
                rotate_piece(DIRECTION, g);
            }

            // after each received control 'press down key'
            skip_tick_gravity(g);

        }
}


void parse_message(char* message, Control* control_message)
{
    // parse control message. string is split after ","
    char* new_relative_position = strtok(message, ",");

    char* should_rotate         = strtok(NULL, ", ");

    Logger("rel_pos: ");
    Logger(new_relative_position);

    Logger("should_rotate");
    Logger(should_rotate);

    if (!should_rotate)
    {
        fprintf(stderr, "incorrect strucuture of control struct. ");
        exit(EXIT_FAILURE);
    }
    control_message->should_rotate = (strcmp(should_rotate, "0")) ? true : false;

    control_message->new_control_available = true;
    control_message->new_position          = std::stoi(message);

    return;
}


void end_of_game_notify(Communication* communication)
{
    std::string end_message = "game_end";
    write(communication->fd_states, end_message.c_str(), strlen(end_message.c_str()));
    Logger("end of episode");
}


int handshake(Communication* communication)
{
    std::string handshake_message = "handshake";
    write(communication->fd_states, handshake_message.c_str(), strlen(handshake_message.c_str()));
    Logger("sent handshake message");

    // read iterations from pipe
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

        return iterations;
    }
    else if (bytesRead == 0) { return 0; }
    else
    {
        perror("read");
        exit(EXIT_FAILURE);
    }
}


void clean_up_named_pipes(Communication* communication)
{
    close(communication->fd_controls);
    close(communication->fd_states);
    unlink(communication->fifo_states_name);

    std::string message = "closed & cleaned up fifos!";
    Logger(message);
}