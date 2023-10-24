#include "tetris.h"

//named pipe
int fd_states, fd_controls;                 // named pipe
const char* fifo_states  = "fifo_states";
const char* fifo_control = "fifo_controls";


void calculate_lines_cleared(Game* g, State* s)
{
    s->lines_cleared = g->score;
}


void calculate_height(Game* g, State* s)
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
    s->height = height;
}



void calculate_holes(Game* g, State* s)
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
    s->holes = number_holes;
}


void calculate_bumpiness(Game* g, State* s)
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

    s->bumpiness = bumpiness;
}


void update_state(Game* g, State * s)
{
    calculate_lines_cleared(g, s);
    calculate_height(g, s);
    calculate_bumpiness(g, s);
    calculate_holes(g, s);
    s->piece_type = g->piece_type;
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





void receive_message(int fd, Control* control_message) {
    char buffer[100];
    ssize_t bytesRead;

    // Read data from the named pipe
    bytesRead = read(fd, buffer, sizeof(buffer) - 1);

    if (bytesRead > 0) {
        // Null-terminate the received data to make it a valid C string
        buffer[bytesRead] = '\0';

        parse_message(buffer, control_message);

        // Print the received message to stdout
        //write(STDOUT_FILENO, buffer, bytesRead);

        // Assuming 'control_message' is a struct that you want to populate from the received data
        // You can parse the 'buffer' and populate 'control_message' accordingly here.
    } else if (bytesRead == 0) {
        return;
    } else {
        perror("read");
    }
}


void parse_message(char* message, Control* control_message)
{
    control_message->new_control_available = true;
    control_message->new_position          = std::stoi(message);
    return;
}


void close_named_pipe(Control* c) {
    
}