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
    int offset = 0;
    int remaining = sizeof(buffer);

    // add lines_cleared to buffer, at first position
    //offset += snprintf(buffer + offset, remaining, "%d,", g->state->lines_cleared);
    offset += snprintf(buffer + offset, remaining, "%d,", g->score);        // dont know if this works, lets test!!
    remaining = sizeof(buffer) - offset;

    offset += snprintf(buffer+offset, remaining, "%d,", g->piece_type);
    remaining = sizeof(buffer) - offset;

    // send middle point of current piece.
    // both are not pointers
    offset += snprintf(buffer+offset, remaining, "%d,", g->middle_coordinate.row);
    remaining = sizeof(buffer) - offset;

    offset += snprintf(buffer+offset, remaining, "%d,", g->middle_coordinate.col);
    remaining = sizeof(buffer) - offset;

    

    if (remaining <= 0) {
        Logger("Buffer overflow in update_state");
        return;
    }

    // serialize game
    for (int i = 0; i < g->rows - 2 && remaining > 1; i++)
    {
        for (int j = 0; j < g->cols - 16 && remaining > 1; j++)
        {
            char cell = '0';
            if (g->game_board[i][j].fixed_piece)
                cell = '1';
            else if (g->game_board[i][j].falling_piece)
                cell = '2';
            
            buffer[offset++] = cell;
            remaining--;
        }
        if (remaining > 1) {
            buffer[offset++] = ',';
            remaining--;
        }
    }

    if (remaining > 0) {
        buffer[offset] = '\0';
    } else {
        buffer[sizeof(buffer) - 1] = '\0';
    }

    //Logger(buffer);

    calculate_lines_cleared(g);
    calculate_height(g);
    calculate_bumpiness(g);
    calculate_holes(g);

    // save to state
    strncpy(g->state->game_state, buffer, sizeof(g->state->game_state) - 1);
    g->state->game_state[sizeof(g->state->game_state) - 1] = '\0';

    //Logger(g->state->game_state);
    //Logger("updated game state");
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
    //Logger(state_string);
    // save control struct
    receive_message(g);
    
    //TODO: problem is here, when another episode starts, the message gets not received and is stuck waiting.

    

}


const int setup_named_pipe(const char* name, mode_t permission, int mode)
{
    if (mkfifo(name, permission) != 0)
    {
        // add error handling
        fprintf(stderr, "error @ mkfifo: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }


    const int fd = open(name, mode);
    if (fd < 0)
    {
        fprintf(stderr, "error @ opening fifo");
        Logger("error @ opening fifo");
       
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

    std::stringstream ss;
    ss << "received message " << buffer;
    ss << ", byted read: "<< std::to_string(bytesRead);
    //Logger(ss.str());

    if (bytesRead > 0)
    {
        // null-terminate the received data to make it a string
        buffer[bytesRead] = '\0';
    }
    else if (bytesRead == 0) { 
        Logger("nothing read, just doing default 1,1");
        
        const char* default_message = "0,0";
        strcpy(buffer, default_message);

        buffer[3] = '\0';

        for (int i = 0; i < 10 && i < bytesRead; i++) {
            // Log ASCII values of first few bytes
            Logger("Byte " + std::to_string(i) + " value: " + std::to_string((unsigned char)buffer[i]));
        }

        // dont return, as this crashed the flow of the communication pipe
        //return; 
        }
    else
    {
        perror("read");
        Logger("error @ read");
        return;
    }
    parse_message(buffer, g->control);
}


void process_control(Game* g)
{
    if (g->control->new_control_available)
        {
            // update state, no new control is available
            g->control->new_control_available = false;

            int new_relative_position = g->control->new_position;

            //Logger("new relative position " + std::to_string(new_relative_position));

            //rotation
            if (g->control->rotation_amount)
            {
                //handle multiple rotations
                for(int i=0;i<g->control->rotation_amount; i++)
                {
                    rotate_piece(DIRECTION, g);
                }
            }
            // positive value -> move right
            while (new_relative_position > 0)
            {
                move_piece(right, g);
                new_relative_position--;
            }

            // negative valuen -> move left
            while ( new_relative_position < 0)
            {
                move_piece(left, g);
                new_relative_position++;
            }



            // after each received control 'press down key'
            skip_tick_gravity(g);

        }
}


void parse_message(char* message, Control* control_message)
{
    //Logger("Received raw message: " + std::string(message));

    // parse control message. string is split after ","
    char* new_relative_position = strtok(message, ",");
    char* rotation_amount = strtok(NULL, ", ");

    // Check if both parts of the message exist
    if (!new_relative_position || !rotation_amount)
    {
        Logger("Error parsing message - missing values. new_relative_position: " + 
              std::string(new_relative_position ? new_relative_position : "NULL") +
              ", rotation_amount: " + 
              std::string(rotation_amount ? rotation_amount : "NULL"));
        fprintf(stderr, "Error: incorrect structure of control struct - missing values\n");
        exit(EXIT_FAILURE);
    }

    //Logger("Parsed message parts - new_relative_position: " + std::string(new_relative_position) + 
     //      ", rotation_amount: " + std::string(rotation_amount));

    try {
        // Convert strings to integers with error checking
        control_message->new_position = std::stoi(new_relative_position);
        control_message->rotation_amount = std::stoi(rotation_amount);
        
        // Logger("Successfully converted to integers - new_position: " + 
        //       std::to_string(control_message->new_position) + 
        //       ", rotation_amount: " + std::to_string(control_message->rotation_amount));
    }
    catch (const std::invalid_argument& e) {
        Logger("Invalid number format in message - " + std::string(e.what()));
        fprintf(stderr, "Error: Invalid number format in control message\n");
        exit(EXIT_FAILURE);
    }
    catch (const std::out_of_range& e) {
        Logger("Number out of range in message - " + std::string(e.what()));
        fprintf(stderr, "Error: Number out of range in control message\n");
        exit(EXIT_FAILURE);
    }

    control_message->new_control_available = true;
    return;
}


void end_of_game_notify(Communication* communication)
{
    std::string end_message = "game_end";
    write(communication->fd_states, end_message.c_str(), strlen(end_message.c_str()));
    Logger("end of episode");

    // Wait for acknowledgment before starting new episode
    char buffer[100];
    ssize_t bytesRead = read(communication->fd_controls, buffer, sizeof(buffer));  // Wait for "ready" signal
    buffer[bytesRead] = '\0';

    std::stringstream ss;
    ss << "received ready signal, after 'game_end'. buffer: " << buffer;

    Logger(ss.str());
}


int handshake(Communication* communication)
{
    usleep(100000);  // 100ms delay
    std::string handshake_message = "handshake";
    // Before write:
    Logger("Writing to fifo: " + handshake_message);
    ssize_t bytes_written = write(communication->fd_states, handshake_message.c_str(), strlen(handshake_message.c_str()));
    Logger("Bytes written: " + std::to_string(bytes_written));
    if (bytes_written < 0) {
        perror("write");
        Logger("error @ write");
        exit(EXIT_FAILURE);
    }
    Logger("sent handshake message");

    fsync(communication->fd_states);  // Force flush
    Logger("After write, bytes_written = " + std::to_string(bytes_written));

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
        Logger("error @ read");
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