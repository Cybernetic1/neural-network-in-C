#include <iostream>
#include <fstream>
#include <sstream>		// for converting double to string
#include <list>
#include <map>

using namespace std;

struct State {
   int x[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct smaller
{
  bool operator()(const State s1, const State s2) const
  {
	for (int i = 0; i < 9; ++i)
		if (s1.x[i] > s2.x[i])
			return true;
		else if (s1.x[i] < s2.x[i])
			return false;
	return false;
  }
};

State board;
State board2;

std::map<State, double, smaller> V1;
std::map<State, double, smaller> V2;

std::list<State> states1;
std::list<State> states2;

int totalStates1 = 0;
int totalStates2 = 0;

int tiles[3] = {0, 1, 2};

void initBoard()
	{
	for (int i = 0; i < 9; ++i)
		board.x[i] = 0;
	}

bool hasWinner(State _board)
	{
	for (int player = 1; player < 3; ++player)
		{
		int tile = tiles[player];

		// check horizontal
		for (int i = 0; i < 3; ++i)
			{
			i = i * 3;
			if ((_board.x[i] == tile) && (_board.x[i + 1] == tile) && (_board.x[i + 2] == tile))
				return 1;
			}

		// check vertical
		for (int i = 0; i < 3; ++i)
			{
			if ((_board.x[i] == tile) && (_board.x[i + 3] == tile) && (_board.x[i + 6] == tile))
				return 1;
			}

		// check backward diagonal
		if ((_board.x[0] == tile) && (_board.x[4] == tile) && (_board.x[8] == tile))
			return 1;

		// check forward diagonal
		if ((_board.x[6] == tile) && (_board.x[4] == tile) && (_board.x[2] == tile))
			return 1;
		}

	//check for draw
	for (int i = 0; i < 9; ++i)
		// 0 estimated probability of winning
		if (_board.x[i] == 0)
			return 0;

	// -1 is for draw match
	return -1;
	}

double determineValue(State _board, int player)
	{
	bool won = hasWinner(_board);

	// win
	if (won)
		if (1 == player)
			return 1.0;
		else
			return 0.0;
		// draw
	else if (!won)
		return 0.0;
	else
		return 0.5;
	}

int switchPlayer(int player)
	{
	if (player == 1)
		return 2;
	else
		return 1;
	}

using namespace std;

void printBoard(State _board)
	{
	for (int index = 0; index < sizeof (State); ++index)
		{
		if (_board.x[index] == 1)
			cout << "X";
		else if (_board.x[index] == 2)
			cout << "O";
		else
			cout << " ";
		if (0 == ((index + 1) % 3))
			cout << "\n";
		}
	}

bool updateBoard(State _board, int player, int index)
	{
	if (_board.x[index] == 0)
		{
		_board.x[index] = player;
		return true;
		}
	return false;
	}

void getListOfBlankTiles(std::list<int> blanks)
	{
	for (int i = 0; i < 9; ++i)
		{
		if (board.x[i] == 0)
			blanks.push_front(i);
		}
	}

using namespace std;
int greedyMove(std::list<State> &states, std::map<State,double,smaller> &V, int player, int *maxIndex)
	{
	// extern State board;
	double maxVal = 0.0;
	int boardIndex = 0;
	*maxIndex = 0;

	// get list of possible next moves
	std::list<int> nextMoves;
	getListOfBlankTiles(nextMoves);

	// make the 1st available move
	boardIndex = nextMoves.front();
	board.x[boardIndex] = player;

	// find max value of current state
	State s;
	maxVal = V.at(s);
	board.x[boardIndex] = 0;		// undo the move

	for (std::list<int>::iterator itr = nextMoves.begin(); itr != nextMoves.end(); ++itr)
		{
		int i = *itr;

		board.x[i] = player;
		double v = V.at(board);
		
		if (v > maxVal)
			{
			boardIndex = i;
			*maxIndex = 0; // = idx;
			maxVal = v;
			}

		board.x[i] = 0;
		}
	return boardIndex;
	}

using namespace std;
void updateEstimateValueOfS(State &sPrime, State &s, double alpha, std::map<State,double,smaller> &V)
	{
	V.at(s) += alpha * (V.at(sPrime) - V.at(s));
	}

void saveStatesToFile(string filename, std::list<State> &states, std::map<State,double,smaller> &V)
	{
	ofstream file2(filename);
	
	// file2 << "Size is " << to_string(states.size()); 
	// file2 << "Size is " << to_string(V.size()); 
	// file2 << "\nDon't know why... \n";
	
    for (std::list<State>::iterator it = states.begin(); it != states.end(); ++it)
		{
		char state_string[9 * 2 + 1];
		for (int i = 0; i < 9; ++i)
			{
			state_string[i*2] = (*it).x[i] + '0';
			state_string[i*2 + 1] = ':';
			}
		state_string[17] = ' ';
		state_string[18] = '\0';

		std::ostringstream strs;
		strs << V[*it];
		string value_string = strs.str();

        file2 << state_string << value_string << endl;
		// file2 << value_string << "\n";
		// file2 << state_string << "\n";
		// file2 << "testing" << endl;
		}
	file2.close();
	}

int loadStatesFromFile(string filename, std::list<State> &states, std::map<State,double,smaller> &V)
	{
	ifstream file1(filename);

	string line;
	State state;
    int total = 0;
    while (getline(file1, line))
		{
		// cout << line << endl;
		for (int i = 0; i < 9; ++i)
			state.x[i] = line[i * 2] - '0';
        states.push_front(state);

        double value = std::stod(line.substr(18));
		// cout << value << "\n";
        V[state] = value;
        ++total;
		}
	file1.close();
    return total;
	}

extern "C" int tic_tac_toe_test2()
	{
	double alpha = 0.01;

	double exploreRate = 0.1;

	// Build states for RL player 1
	cout << "Reading...\n";
	int totalStates1 = loadStatesFromFile("tictactoe.dat", states1, V1);
	cout << "Total read: " << to_string(totalStates1) << endl;
	
	cout << "Writing...\n";
	saveStatesToFile("test.dat", states1, V1);

	return 0;
	}
