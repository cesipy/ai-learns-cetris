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
    calculate_lines_cleared(g);
    calculate_height(g);
    calculate_bumpiness(g);
    calculate_holes(g);

    // save piece type to state
    g->state->piece_type = g->piece_type;
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

    char* state_string = state_to_string(g->state);
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


void clean_up_named_pipes(Game* g) 
{
    close(g->communication->fd_controls);
    close(g->communication->fd_states);
    unlink(g->communication->fifo_states_name);

    std::string message = "closed & cleaned up fifos!";
    Logger(message);
}