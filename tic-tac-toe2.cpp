#include <iostream>
#include <fstream>
#include <sstream>		// for converting double to string
#include <list>
#include <map>
#include <math.h>		// floor

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

int hasWinner()
	{
	for (int player = 1; player < 3; ++player)
		{
		int tile = tiles[player];

		// check horizontal
		for (int i = 0; i < 3; ++i)
			{
			int j = i * 3;
			if ((board.x[j] == tile) && (board.x[j + 1] == tile) && (board.x[j + 2] == tile))
				return 1;
			}

		// check vertical
		for (int i = 0; i < 3; ++i)
			{
			if ((board.x[i] == tile) && (board.x[i + 3] == tile) && (board.x[i + 6] == tile))
				return 1;
			}

		// check backward diagonal
		if ((board.x[0] == tile) && (board.x[4] == tile) && (board.x[8] == tile))
			return 1;

		// check forward diagonal
		if ((board.x[6] == tile) && (board.x[4] == tile) && (board.x[2] == tile))
			return 1;
		}

	//check for draw
	for (int i = 0; i < 9; ++i)
		if (board.x[i] == 0)
			return 0;

	// -1 is for draw match
	return -1;
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
	for (int i = 0; i < 9; ++i)
		{
		if (_board.x[i] == 1)
			cout << "X";
		else if (_board.x[i] == 2)
			cout << "O";
		else
			cout << ".";
		if (0 == ((i + 1) % 3))
			cout << "\n";
		}
	}

bool updateBoard(int player, int index)
	{
	if (board.x[index] == 0)
		{
		board.x[index] = player;
		return true;
		}
	return false;
	}

void getListOfBlankTiles(std::list<int> &blanks)
	{
	for (int i = 0; i < 9; ++i)
		{
		if (board.x[i] == 0)
			blanks.push_front(i);
		}
	}

using namespace std;

int greedyMove(std::map<State, double, smaller> &V, int player)
	{
	// extern State board;
	double maxVal = 0.0;
	int bestMove = -1;

	// get list of possible next moves
	std::list<int> nextMoves;
	getListOfBlankTiles(nextMoves);
	// cout << "# next moves = " << to_string(nextMoves.size()) << "\n";

	maxVal = -100.0; // set to -∞ initially

	// For all possible next moves...
	for (std::list<int>::iterator itr = nextMoves.begin(); itr != nextMoves.end(); ++itr)
		{
		int i = *itr;

		board.x[i] = player;
		double v = V.at(board);
		// cout << "Value = " << to_string(v) << "\n";

		if (v > maxVal)
			{
			bestMove = i;
			maxVal = v;
			}

		board.x[i] = 0;
		}

	// cout << "Made greedy move...\n";
	return bestMove;
	}

using namespace std;

void BellmanUpdate(State &s2, State &s, std::map<State, double, smaller> &V)
	{
#define alpha	0.01

	// cout << "Making Bellman update...\n";

	// printBoard(s);
	// printBoard(s2);

	V.at(s) += alpha * (V.at(s2) - V.at(s));
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
		// cout << line;
		for (int i = 0; i < 9; ++i)
			state.x[i] = line[i * 2] - '0';
        states.push_front(state);

        double value = std::stod(line.substr(18));
		// cout << "\t" << value << "\n";
        V[state] = value;
        ++total;
		}
	file1.close();
    return total;
	}

extern "C" int tic_tac_toe_test2()
	{
	// Build states for RL player 1
	cout << "Loading player 1...\n";
	int totalStates1 = loadStatesFromFile("ttt1.dat", states1, V1);
	cout << "Total read: " << to_string(totalStates1) << endl;

	// Build states for RL player 1
	cout << "Loading player 2...\n";
	int totalStates2 = loadStatesFromFile("ttt2.dat", states2, V2);
	cout << "Total read: " << to_string(totalStates2) << endl;

	int playTimes = 1000;
	int numPlayer1Won = 0;
	int numPlayer2Won = 0;
	int numDraws = 0;
	int player = 1;

	while (true)			// Loop over 1000 trials
		{
		initBoard();

		player = ((rand() / (double) RAND_MAX) > 0.5) ? 1 : 0;

		cout << "New Game.... \n";
		printBoard(board);

		State prev_s1 = State(); // initialized as state "0"
		State max_s1 = State();

		State prev_s2 = State();
		State max_s2 = State();

		while (true)		// Loop over 1 single game
			{
			std::list<int> nextMoves;
			getListOfBlankTiles(nextMoves);
			int countNextMoves = nextMoves.size();

			cout << "Move of player: " << to_string(player) << "\n";

			double exploreRate = 0.1;
			double ex = (rand() / (double) RAND_MAX); // explore or not?

			// ************ Make 1 move
			int userPlay;
			if (player == 2)
				{
				if (ex <= exploreRate)
					{
					// generate random # within range of possible moves
					int move = (int) floor((rand() / (double) RAND_MAX) * countNextMoves);
					cout << "Exploring move = " << to_string(move) << "\n";
					std::list<int>::iterator it = nextMoves.begin();
					std::advance(it, move);
					userPlay = *it;
					updateBoard(player, userPlay);
					prev_s1 = board;
					}
				else
					{
					userPlay = greedyMove(V2, player);
					cout << "Greedy move = " << to_string(userPlay) << "\n";
					// max_s2 should be the new state
					updateBoard(player, userPlay);
					max_s2 = board;

					// cout << "V2(s) changed from " << to_string(V2[prev_s2]);
					BellmanUpdate(max_s2, prev_s2, V2);
					// cout << "to " << to_string(V2[prev_s2]);
					prev_s2 = max_s2;
					}
				}
			else // Player 1 (computer)
				{
				if (ex <= exploreRate)
					{
					int move = (int) floor((rand() / (double) RAND_MAX) * countNextMoves);
					cout << "Exploring move = " << to_string(move) << "\n";
					std::list<int>::iterator it = nextMoves.begin();
					std::advance(it, move);
					userPlay = *it;
					updateBoard(player, userPlay);
					prev_s2 = board;
					}
				else
					{
					userPlay = greedyMove(V1, player);
					cout << "Greedy move = " << to_string(userPlay) << "\n";
					updateBoard(player, userPlay);
					max_s1 = board;
					BellmanUpdate(max_s1, prev_s1, V1);
					prev_s1 = max_s1;
					}
				}

			printBoard(board);

			int won = hasWinner();
			if (won > 0)
				{
				if (1 == player)	// player 1 wins
					{
					++numPlayer1Won;
					max_s2 = board;
					// cout << "V2(s) changed from " << to_string(V2[prev_s2]);
					BellmanUpdate(max_s2, prev_s2, V2);
					// cout << "to " << to_string(V2[prev_s2]);
					}
				else				// player 2 wins
					{
					++numPlayer2Won;
					max_s1 = board;
					BellmanUpdate(max_s1, prev_s1, V1);
					}

				cout << "Winner is: player " << to_string(player) << "\n\n";
				break;
				}

			if (won == -1) // draw
				{
				numDraws++;
				cout << "It's a draw !\n\n";
				break;
				}

			// continue with game....
			player = switchPlayer(player);
			}

		// Next game...
		--playTimes;
		if (playTimes == 0)
			break;
		//if (getchar() == 'q')
		//	break;
		}

	cout << "Saving...\n";

	saveStatesToFile("ttt1.dat", states1, V1);
	saveStatesToFile("ttt2.dat", states2, V2);

	cout << "\n\nGame stats:\n";
	cout << "Player 1 # of Wins = " << to_string(numPlayer1Won) << "\n";
	cout << "Player 2 # of Wins = " << to_string(numPlayer2Won) << "\n";
	cout << "        # of Draws = " << to_string(numDraws) << "\n";

	return 0;
	}
