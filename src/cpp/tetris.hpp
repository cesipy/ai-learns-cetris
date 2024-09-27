#ifndef MAIN_H
#define MAIN_H

#include <ncurses.h>
#include <random>
#include <unistd.h>
#include <sstream>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <chrono>
#include <thread>


#define GRAVITY_TICKS 10
#define SLEEP_TIME 10
#define BOARD_WIDTH  30
#define BOARD_HEIGHT 30
#define EMPTY_CELL 0
#define CELL 1
#define AMOUNT_OF_PIECES 3
#define BOARD_EDGE_RIGHT (BOARD_WIDTH-17)
#define DIRECTION left
#define NO_COLOR 8
#define DETERMINISTIC 1

#define ADD_BLOCK(w,x) waddch((w),' '|A_REVERSE|COLOR_PAIR(x));     \
                       waddch((w),' '|A_REVERSE|COLOR_PAIR(x))
#define ADD_EMPTY(w) waddch((w), ' '); waddch((w), ' ')

#define EARLY_QUIT 1
#define NORMAL_QUIT 0

/* ----------------------------------------------------------- */

enum type {
    O, I, J, L, T, S, Z,  initial // initial only for the first initialization
};


enum direction {
    left, right, down
};

/**
 * struct for sharing the current state of the game.
 */
typedef struct {
    int lines_cleared;      // to be maximized
    int height;             // to be minimized
    int holes;              // to be minimized
    int bumpiness;          // to be minimized
    type piece_type;        // type of falling piece
    char game_state[1024];
}State;


typedef struct {
    bool new_control_available;       // indicate new control received from pipe
    int new_position;                 // relative to current position (can be negative or positive
    bool should_rotate;               // should piece rotate
    // needs further implementation

}Control;


/**
 * stores position with (x, y)
 *
 * @param x x coordinate
 * @param y y coordinate
 */
typedef struct {
    int row;
    int col;
}Position;


typedef struct {
    int fd_states;      // communicates game states
    int fd_controls;    // communicates game control structs. (= new position, etc)
    const char* fifo_states_name;
    const char* fifo_control_name;
} Communication; 


typedef struct {
    int value;
    bool falling_piece;
    bool fixed_piece;
    bool moved_in_prev_iteration;
    bool rotated_in_prev_iteration;
    bool is_new;                // used for copy in rotation
    short color;                // color from 0 to 7
}Block;

/**
 * game struct to store attributes of the game.
 * @param rows rows of game
 * @param cols cols of the game
 * @param game_board[][] stores the board consisting of an 2d array of blocks
 */
typedef struct {
    // for board:
    int rows;
    int cols;

    Block game_board[BOARD_HEIGHT][BOARD_WIDTH];
    WINDOW* win;
    bool running;
    char bottom_height;         // number of bottom row
    bool need_new_piece;        // is a piece still falling, or is a new one needed
    int highest_fixed_block;    // height of the highest current block. (to check for game-over)
    Position middle_coordinate; // store the middle point of current ffalling piece (to rotate)
    int score;                  // store of current game iteration
    type piece_type;            // type of falling piece
    //further add
    int difficulty;
    Control* control;
    State* state;
    Communication* communication;
    int piece_counter;
}Game;


typedef struct {
    int rows;
    int cols;

    WINDOW* win;
    int type;
}Piece;


typedef struct {
    int x;
    int y;
}Coordinate;

/* ----------------------------------------------------------- */
void initialize_game(Game* g);
int main_loop(Game* g);
void insert_falling_piece(type type, Game* g);
int hit_bottom();
void init_colors();
void game_init(Game*, int rows, int cols);
void display_board(Game* g);

/**
 * Apply gravity to the falling piece on the game board.
 *
 * This function checks if the blocks below the falling piece are free. If they are,
 * it moves the falling piece down one row. If the falling piece reaches the bottom
 * or encounters obstacles below, it sets the piece as a fixed piece on the board
 * and triggers the need for a new piece.
 *
 * @param g A pointer to the Game struct containing game information.
 * @return 1 if the falling piece has reached the bottom and needs a new piece, 0 otherwise.
 * Is used for skip_gravity function
 */
int gravity(Game* g);
void example_fill_board(Game* g);

/**
 * move a piece in direction 'dir'.
 */
void move_piece(direction, Game* g);

/**
 * Check if a block at the specified row and column is within the valid game board boundaries.
 *
 * @param row The row of the block to check.
 * @param col The column of the block to check.
 * @return true if the block is within the valid game board boundaries, false otherwise.
 */
bool is_valid_block(int rows, int cols, Game* g);


/**
 * @brief Set a block in a given row and column to a specified value with several attributes.
 *
 * @param row The row in which to set the block.
 * @param col The column in which to set the block.
 * @param value The value to set for the block (e.g., EMPTY_CELL or CELL).
 * @param is_falling Whether the block is part of a falling piece.
 * @param moved_in_prev_iteration Whether the block was moved in the previous iteration.
 * @param color Value from 0 to 7, indicates the color. 8 means field is free.
 */
void set_block(int row, int col, int value, bool is_falling, bool moved_in_prev_iteration, short color, Game* g);

void piece_counter_increase(void);

/**
 * Converts all falling blocks to static/fixed blocks.
 * Updates `need_new_piece`to true.
 * Updates `highest_fix_block` position in the game.
 */
void falling_to_fixed( Game* g);
bool can_piece_move(direction, Game* g);

/**
 * Checks if block is empty.
 * @param g
 * @return
 */
bool is_empty_block(int, int, Game* g);

/**
 * Skips gravity until block hits bottom.
 * Is used when down arrow is pressed.
 * @param g
 */
void skip_tick_gravity( Game* g);

int check_game_state(Game* g);

/**
 * Generates number in range (min, max).
 * @param min
 * @param max
 * @return int - random number
 */
int generate_random_number(int min, int max);

Position block_position_after_rotation(int row, int col, direction dir, Game* g);

/**
 * Rotates piece in direction `dir`
 * @param dir direction the piece should rotate. please use makro`DIRECTION`
 */
void rotate_piece(direction dir, Game* g);

/**
 * Detects rows full with fixed blocks.
 *
 * When a full row is found, `clear_line(row)`and `adjust_blocks(row)` is called.
 * In this case, the score is increased.
 */
void manage_full_lines(Game* g);

/**
 * Clears all blocks in row i
 * @param row  line to eliminate blocks in
 */
void clear_line(int row, Game* g);

/**
 * Moves all fixed blocks above the specified row one row down.
 *
 * Is called by `manage_full_lines` when full line is detected.
 * @param row adjust all all blocks above `row`.
 */
void adjust_blocks(int row, Game* g);

/**
 * Prints current score to the game window.
 */
void display_score(Game* g);

int check_input(Game* g);


// from communication file
/* ----------------------------------------------------------- */

// files from communication.cpp
void calculate_lines_cleared(Game* g);
void calculate_height(Game* g);
void calculate_holes(Game* g);
void calculate_bumpiness(Game* g);
void update_state(Game* g);
const int setup_named_pipe(const char* name);
char* state_to_string(const State* s);
void receive_message(Game* g);
void parse_message(char* message, Control* control_message);
void communicate(Game* g);
void process_control(Game* g);
void clean_up_named_pipes(Communication* communication);
int handshake(Communication* communication);
void end_of_game_notify(Communication* communication);

/**
 *
 * @param name
 * @param permission
 * @param mode specifies opening mode, 1 - write only only, 0 - read only
 * @return
 */
const int setup_named_pipe(const char* name, mode_t permission, int mode );


/* ----------------------------------------------------------- */

std::string getCurrentDateTime( std::string s );
void Logger( std::string logMsg );

#endif