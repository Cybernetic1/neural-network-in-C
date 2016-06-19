#include <iostream>
#include <fstream>
#include <sstream>		// for converting double to string
#include <list>
#include <map>
#include <math.h>		// floor

using namespace std;

struct State
	{
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

int tiles[3] = {0, -1, 1};

void initBoard()
	{
	for (int i = 0; i < 9; ++i)
		board.x[i] = 0;
	}

int hasWinner()
	{
	for (int player = -1; player < 2; player += 2) // player = -1 and 1
		{
		int tile = player;

		// check horizontal
		for (int i = 0; i < 3; ++i)
			{
			int j = i * 3;
			if ((board.x[j] == tile) && (board.x[j + 1] == tile) && (board.x[j + 2] == tile))
				return player;
			}

		// check vertical
		for (int i = 0; i < 3; ++i)
			{
			if ((board.x[i] == tile) && (board.x[i + 3] == tile) && (board.x[i + 6] == tile))
				return player;
			}

		// check backward diagonal
		if ((board.x[0] == tile) && (board.x[4] == tile) && (board.x[8] == tile))
			return player;

		// check forward diagonal
		if ((board.x[6] == tile) && (board.x[4] == tile) && (board.x[2] == tile))
			return player;
		}

	// 0 is for game still open
	for (int i = 0; i < 9; ++i)
		if (board.x[i] == 0)
			return 0;

	// -2 is for draw match
	return -2;
	}

int switchPlayer(int player)
	{
	if (player == 1)
		return -1;
	else
		return 1;
	}

using namespace std;

void printState(State _board)
	{
	for (int i = 0; i < 9; ++i)
		{
		if (_board.x[i] == 1)
			cout << "X";
		else if (_board.x[i] == -1)
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
		// printState(board);
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

extern "C"
	{
	double get_V(int [9]);
	}

int computerMove(int player)
	{
	extern double get_V(int x[9]);
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
		double v = get_V(board.x);
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

	// printState(s);
	// printState(s2);

	V.at(s) += alpha * (V.at(s2) - V.at(s));
	}

void saveStatesToFile(string filename, std::list<State> &states, std::map<State, double, smaller> &V)
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
			state_string[i * 2] = (*it).x[i] + '0';
			state_string[i * 2 + 1] = ':';
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

int loadStatesFromFile(string filename, std::list<State> &states, std::map<State, double, smaller> &V)
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
		// printState(state);

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
	extern void init_Vnet(void);
	extern void load_Vnet(void);
	extern void save_Vnet(char *);
	extern double get_V(int x[9]);
	extern void train_V(int x[9], double v);
	extern void learn_V(int x[9], int y[9]);
	extern void beep();

	// Build states for RL player 1
	cout << "Loading player 1's V values...\n";
	states1.clear();
	int totalStates1 = loadStatesFromFile("ttt1.dat", states1, V1);
	cout << "Total read: " << to_string(totalStates1) << "\n";

	//*** Train player1's V network
	cout << "[i] = init new net and train with end state values\n";
	cout << "[o] = train with old V\n";
	cout << "[t] = train with end state values\n";
	cout << "[r] = init new net with random weights\n";
	cout << "[-] = just load net\n";
	char key;
	do
		key = getchar();
	while (key == '\n');

	if (key == 'i' || key == 'r')
		init_Vnet();
	else
		load_Vnet();

	if (key == 'o')
		{
		for (int t = 0; t < 10000; ++t)
			{
			// For all states
			for (std::list<State>::iterator itr = states1.begin(); itr != states1.end(); ++itr)
				{
				State s = *itr;

				double v = V1.at(s);

				train_V(s.x, v);
				}

			double absError = 0.0; // sum of abs(error)
			// Calculate error
			for (std::list<State>::iterator itr = states1.begin(); itr != states1.end(); ++itr)
				{
				State s = *itr;

				double v = V1.at(s);
				//cout << "v = " << to_string(v) << "\t";

				double v2 = get_V(s.x);
				//cout << "v2 = " << to_string(v2) << "\t";

				double error = v - v2; // ideal - actual
				//cout << "err = " << to_string(error) << "\n";

				absError += fabs(error);
				}
			printf("(%05d) ", t);
			printf("∑ abs err = %.1f (avg = %.3f)\r", absError, absError / 8533.0);

			if (isnan(absError))
				{
				init_Vnet();
				t = 0;
				}
			}
		cout << "\n\n";
		save_Vnet("v.net");
		}
	else if (key == 't' || key == 'i')
		{
		for (int t = 0; t < 500; ++t)
			{
			// For all states
			for (std::list<State>::iterator itr = states1.begin(); itr != states1.end(); ++itr)
				{
				State s = *itr;

				board = s;
				int result = hasWinner();
				double v;

				if (result == -2)
					v = 0.5;
				else if (result == -1)
					v = 0.0;
				else if (result == 1)
					v = 1.0;

				if (result != 0)
					train_V(s.x, v);
				}

			double absError = 0.0; // sum of abs(error)
			// Calculate error
			for (std::list<State>::iterator itr = states1.begin(); itr != states1.end(); ++itr)
				{
				State s = *itr;

				board = s;
				int result = hasWinner();
				double v;

				double v2 = get_V(s.x);
				//cout << "v2 = " << to_string(v2) << "\t";

				if (result == -2)
					v = 0.5;
				else if (result == -1)
					v = 0.0;
				else if (result == 1)
					v = 1.0;

				double error = v - v2; // ideal - actual
				//cout << "err = " << to_string(error) << "\n";

				if (result == 0)
					error = 0.0;

				absError += fabs(error);
				}
			printf("(%05d) ", t);
			printf("∑ abs err = %.1f (avg = %.3f)\r", absError, absError / 8533.0);

			if (isnan(absError))
				{
				init_Vnet();
				t = 0;
				}
			}
		cout << "\n\n";
		save_Vnet("v.net");
		}

	// Build states for RL player 1
	cout << "\n\nLoading player -1...\n";
	states2.clear();
	int totalStates2 = loadStatesFromFile("ttt2.dat", states2, V2);
	cout << "Total read: " << to_string(totalStates2) << "\n";

#define totalGames 500000
	int playTimes = 0;
	int numPlayer1Won = 0;
	int numPlayer_1Won = 0;
	int numDraws = 0;
	int player = 1;

	while (true) // Loop over 1000 trials
		{
		initBoard();

		player = ((rand() / (double) RAND_MAX) > 0.5) ? 1 : -1;

		cout << "Game # " << to_string(playTimes) << "\r";
		// printState(board);

		State prev_s1 = State(); // initialized as state "0"
		State max_s1 = State();

		State prev_s2 = State();
		State max_s2 = State();

		while (true) // Loop over 1 single game
			{
			std::list<int> nextMoves;
			getListOfBlankTiles(nextMoves);
			int countNextMoves = nextMoves.size();

			// cout << "Move of player: " << to_string(player) << "\n";

			double exploreRate = 0.1;
			double ex = (rand() / (double) RAND_MAX); // explore or not?

			// ************ Make 1 move
			int userMove;
			if (player == -1) // Old RL learner
				{
				if (ex <= exploreRate)
					{
					// generate random # within range of possible moves
					int move = (int) floor((rand() / (double) RAND_MAX) * countNextMoves);
					std::list<int>::iterator it = nextMoves.begin();
					std::advance(it, move);
					userMove = *it;
					//cout << "Exploring move = " << to_string(userMove) << "\n";
					updateBoard(player, userMove);
					prev_s1 = board;
					}
				else
					{
					userMove = greedyMove(V2, player);
					//cout << "Greedy move = " << to_string(userMove) << "\n";
					// max_s2 should be the new state
					updateBoard(player, userMove);
					max_s2 = board;

					// cout << "V2(s) changed from " << to_string(V2[prev_s2]);
					BellmanUpdate(max_s2, prev_s2, V2);
					// cout << "to " << to_string(V2[prev_s2]);
					prev_s2 = max_s2;
					}
				}
			else // Player 1 (our NN learner)
				{
				if (ex <= exploreRate)
					{
					int move = (int) floor((rand() / (double) RAND_MAX) * countNextMoves);
					//cout << "Exploring move = " << to_string(move) << "\n";
					std::list<int>::iterator it = nextMoves.begin();
					std::advance(it, move);
					userMove = *it;
					updateBoard(player, userMove);
					prev_s1 = board;
					}
				else
					{
					userMove = computerMove(player);
					//cout << "Computer move = " << to_string(userMove) << "\n";
					updateBoard(player, userMove);
					max_s1 = board;
					learn_V(max_s1.x, prev_s1.x);
					prev_s1 = max_s1;
					}
				}

			//printState(board);

			int won = hasWinner();

			if (won == -2) // draw
				{
				numDraws++;
				train_V(board.x, 0.5);
				// cout << "It's a draw !\n\n";
				break;
				}

			if (won != 0)
				{
				if (1 == player) // our NN learner wins
					{
					++numPlayer1Won;
					max_s2 = board;
					BellmanUpdate(max_s2, prev_s2, V2);
					train_V(max_s2.x, 1.0);
					// cout << "V2(s) changed from " << to_string(V2[prev_s2]);
					// cout << "to " << to_string(V2[prev_s2]);
					}
				else // old RL player (-1) wins
					{
					++numPlayer_1Won;
					max_s1 = board;
					train_V(max_s1.x, 0.0);
					learn_V(max_s1.x, prev_s1.x);
					}

				// cout << "Winner is: player " << to_string(player) << "\n\n";
				break;
				}

			// continue with game....
			player = switchPlayer(player);
			}

		// Next game...
		++playTimes;
		if (playTimes > totalGames)
			break;
		//if (getchar() == 'q')
		//	break;
		}

	// cout << "\n\nSaving RL values...\n";
	// saveStatesToFile("ttt1.dat", states1, V1);
	// saveStatesToFile("ttt2.dat", states2, V2);

	cout << "Saving NN learner values...\n";
	save_Vnet("v.net");

	cout << "\n\nGame stats:\n";
	printf("Player  1 Wins %d (%2.1f%%)\n", numPlayer1Won, ((float) numPlayer1Won) / totalGames * 100.0);
	printf("Player -1 Wins %d (%2.1f%%)\n", numPlayer_1Won, ((float) numPlayer_1Won) / totalGames * 100.0);
	printf("         Draws %d (%2.1f%%)\n", numDraws, ((float) numDraws) / totalGames * 100.0);

	beep();
	return 0;
	}
