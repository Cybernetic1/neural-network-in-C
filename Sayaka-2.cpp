// **************** In Sayaka-2, Q(K1,K2) where K2 is an "action" instead of a state

#include <cstdlib>		// rand()
#include <iostream>
#include <fstream>
#include <sstream>		// for converting double to string
#include <list>
#include <map>
#include <math.h>		// floor, nearbyint
#include "tic-tac-toe.h"

using namespace std;

extern State board;

extern std::map<State, double, smaller> V1; // V1 is needed to train Q-net
extern std::map<State, double, smaller> V2;

extern std::list<State> states1;
extern std::list<State> states2;

extern int totalStates1;
extern int totalStates2;

#define dimK 9

extern "C" // Functions from Q-learning.c
	{
	double getQ(double [dimK], double [dimK]);
	void init_Qnet(void);
	void load_Qnet(char const *);
	void save_Qnet(char const *);
	void train_Q(int x[dimK], double v);
	void Q_learn(int x[dimK], int y[dimK], double R);
	double maxQ(int [dimK], double [dimK]);

	// functions from visualization.c
	extern int  delay_vis(int);
	extern void pause_graphics();
	}

using namespace std;

// ******** Functions from tic-tac-toe2.cpp
extern void initBoard(void);
extern bool updateBoard(int player, int index);
extern int switchPlayer(int player);
extern void getListOfBlankTiles(std::list<int> &blanks);
extern void printState(State board);
extern int greedyMove(std::map<State, double, smaller> &V, int player);
extern int computerMove(int player);
extern int hasWinner(void);
extern void BellmanUpdate(State &s2, State &s, std::map<State, double, smaller> &V);
extern int loadVFromFile(string filename, std::list<State> &states, std::map<State, double, smaller> &V);
extern void saveVToFile(string filename, std::list<State> &states, std::map<State, double, smaller> &V);

// Original algorithm is to find max V amongst board positions.
// Now we output the next move based on max_Q algorithm
// 1. get current board position → K1
// 2. find maxQ for K1, obtaining K2
// 3. make move according to K2
// 4. if move is invalid, train Qnet and re-try

int Q_moveSayaka2()
	{
	int bestMove = -1;
	int K_out[dimK];
	double K2[dimK];

	int tries = 0;
#   define MaxTries 50
	while (tries++ < MaxTries)				// Try gradient descent with restart
		{
		maxQ(board.x, K2);			// we don't need the max Q value itself

		// Find max element in K2, its index would be the move #
		double max = -1000000.0;
		for (int k = 0; k < dimK; ++k)
			if (K2[k] > max)
				max = K2[k], bestMove = k;

		// Is the move valid?
		if (board.x[bestMove] != 0)
			bestMove = -1;				// square is already occupied

		// cout << "Made greedy move...\n";

		if (bestMove < 0)
			{
			for (int k = 0; k < dimK; ++k)
				K_out[k] = ((k == bestMove) ? 1 : 0);
			Q_learn(board.x, K_out, -0.1);
			}
		else
			{
			// printf("best move = %d\n", bestMove);
			break;
			}
		}

	printf("(%d tries) ", tries);
	return bestMove;						// returns -1 if fail to find valid move
	}

extern "C" int tic_tac_toe_test4()
	{
	extern void beep();

	// Read data for RL player 1 (Our learner)
	cout << "Loading player 1's Q values...\n";
	states1.clear();
	int totalStates1 = loadVFromFile("ttt1.dat", states1, V1);
	cout << "Total read: " << to_string(totalStates1) << "\n";

	//*** Train player1's Q network
	cout << "[i] = init new net and train with end state values\n";
	cout << "[o] = train with old V values\n";
	cout << "[t] = train with end state values\n";
	cout << "[r] = init new net with random weights\n";
	cout << "[-] = just load net\n";
	char key;
	do
		key = getchar();
	while (key == '\n');

	if (key == 'i' || key == 'r')
		init_Qnet();
	else
		load_Qnet("Q.net");

	if (key == 'o')
		{
		for (int t = 0; t < 10000; ++t)
			{
			// **** We need to make Q(x,x') consistent with V(x)
			// for each V(x), we make sure that Q(x,*) cannot exceed V(x)

			// For all states
			for (std::list<State>::iterator itr = states1.begin(); itr != states1.end(); ++itr)
				{
				State s = *itr;

				double v = V1.at(s);

				// This needs to be changed:
				train_Q(s.x, v);
				}

			double absError = 0.0; // sum of abs(error)
			// Calculate error
			for (std::list<State>::iterator itr = states1.begin(); itr != states1.end(); ++itr)
				{
				State s = *itr;

				double v = V1.at(s);
				//cout << "v = " << to_string(v) << "\t";

				double v2 = 0.0; // getQ(s.x);
				//cout << "v2 = " << to_string(v2) << "\t";

				double error = v - v2; // ideal - actual
				//cout << "err = " << to_string(error) << "\n";

				absError += fabs(error);
				}
			printf("(%05d) ", t);
			printf("∑ abs err = %.1f (avg = %.3f)\r", absError, absError / 8533.0);

			if (isnan(absError))
				{
				init_Qnet();
				t = 0;
				}
			}
		cout << "\n\n";
		save_Qnet("Q.net");
		}
	else if (key == 't' || key == 'i') // Train with end-state values
		{
		for (int t = 0; t < 1000; ++t)
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
					train_Q(s.x, v);
				}

			double absError = 0.0; // sum of abs(error)
			// Calculate error
			for (std::list<State>::iterator itr = states1.begin(); itr != states1.end(); ++itr)
				{
				State s = *itr;

				board = s;
				int result = hasWinner();
				double v;

				double x1[dimK], x2[dimK];

				for (int k = 0; k < dimK; ++k)
					{
					x1[k] = (double) s.x[k];

					// 2nd argument is random
					x2[k] = (rand() / (float) RAND_MAX) * 2.0 - 1.0; // in [+1,-1]
					}

				double v2 = getQ(x1, x2);
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
			printf("∑ abs err = %.1f (avg = %.3f)\n", absError, absError / 8533.0);

			if (isnan(absError))
				{
				init_Qnet();
				t = 0;
				}
			}
		cout << "\n\n";
		save_Qnet("Q.net");
		}

	// Build states for RL player -1 ("Computer player")
	cout << "\n\nLoading player -1...\n";
	states2.clear();
	int totalStates2 = loadVFromFile("ttt2.dat", states2, V2);
	cout << "Total read: " << to_string(totalStates2) << "\n";

#   define totalGames 100
	int playTimes = 0;
	int numPlayer1Won = 0;
	int numPlayer_1Won = 0;
	int numDraws = 0;
	int player = 1;
	int ourWins1K = 0;				// Number of times per 1000 games
	int ourMoves1K = 0;
	int userKey;

	while (true) // Loop over #totalGames trials
		{
		initBoard();

		player = ((rand() / (double) RAND_MAX) > 0.5) ? 1 : -1;

		printf("Game #%d\n", playTimes);
		// printState(board);

		State prev_s1 = State();		// initialized as state "0"
		State prev_move1 = State();		// not really a "state", just for storing the move index

		State prev_s_1 = State();
		State max_s_1 = State();

		while (true) // Loop over 1 single game
			{
			std::list<int> nextMoves;
			getListOfBlankTiles(nextMoves);
			int countNextMoves = nextMoves.size();

			// cout << "Move of player: " << to_string(player) << "\n";

			double ex;

			// ************ Make 1 move
			int userMove;
			if (player == -1) // Old RL learner
				{
				ex = (rand() / (double) RAND_MAX); // explore or not?

				if (ex <= 0.1)
					{
					// generate random # within range of possible moves
					int move = (int) floor((rand() / (double) RAND_MAX) * countNextMoves);
					std::list<int>::iterator it = nextMoves.begin();
					std::advance(it, move);
					userMove = *it;
					//cout << "Exploring move = " << to_string(userMove) << "\n";
					updateBoard(player, userMove);
					prev_s_1 = board;
					}
				else
					{
					userMove = greedyMove(V2, player);
					//cout << "Greedy move = " << to_string(userMove) << "\n";
					// max_s2 should be the new state
					updateBoard(player, userMove);
					max_s_1 = board;

					// cout << "V2(s) changed from " << to_string(V2[prev_s2]);

					// Is this update really needed?
					// Or if we just want it to perform statically...
					// BellmanUpdate(max_s2, prev_s2, V2);
					// cout << "to " << to_string(V2[prev_s2]);
					prev_s_1 = max_s_1;
					}
				}
			else // Player 1 (Genifer)
				{
				while (true)
					{
					ex = (rand() / (double) RAND_MAX); // explore or not?
					// printf("random # = %f\r", ex);

#                   define exploreRate 0.08
					if (ex <= exploreRate)
						{
						int moveIndex = (int) floor((rand() / (double) RAND_MAX) * countNextMoves);
						//cout << "Exploring move = " << to_string(move) << "\n";
						std::list<int>::iterator it = nextMoves.begin();
						std::advance(it, moveIndex);
						userMove = *it;
						for (int k = 0; k < 9; ++k)
							prev_move1.x[k] = (userMove == k) ? 1 : 0;
						Q_learn(board.x, prev_move1.x, 0.4);
						updateBoard(player, userMove);
						// prev_s1 = board;
						break;
						}
					else
						{
						userMove = Q_moveSayaka2();
						//cout << "Computer move = " << to_string(userMove) << "\n";
						if (userMove >= 0)
							{
							++ourMoves1K;
							for (int k = 0; k < 9; ++k)
								prev_move1.x[k] = (userMove == k) ? 1 : 0;
							Q_learn(board.x, prev_move1.x, 0.6);
							updateBoard(player, userMove);
							// prev_s1 = board;
							// prev_move1 = userMove;
							// printf("move made\n");
							break;
							}
						}
					}
				}

			//printState(board);

			int won = hasWinner();

			if (won == -2) // draw
				{
				numDraws++;
				// train_Q(board.x, 0.0);
				Q_learn(board.x, prev_move1.x, 0.5);
				printf("-");
				break;
				}

			if (won != 0)
				{
				if (1 == player) // Genifer (1) wins
					{
					++ourWins1K;
					++numPlayer1Won;
					// BellmanUpdate(max_s2, prev_s2, V2);
					// Still unsolved - need to train over all prev states:
					// train_Q(max_s_1.x, 10.0);
					Q_learn(board.x, prev_move1.x, 0.95);
					// cout << "V2(s) changed from " << to_string(V2[prev_s2]);
					// cout << "to " << to_string(V2[prev_s2]);
					}
				else // old RL player (-1) wins
					{
					++numPlayer_1Won;
					// Still unsolved - need to train over all prev states:
					// train_Q(max_s1.x, -0.7);
					Q_learn(board.x, prev_move1.x, -0.2);
					}

				printf(player == 1 ? "█" : " ");
				break;
				}

			// continue with game....
			player = switchPlayer(player);

			userKey = delay_vis(0);
			if (userKey == 1)
				break;
			}

		// Next game...
		++playTimes;
		// fflush(stdout);
		if ((playTimes % 1000) == 0)
			{
			printf("per 1K wins = %d (%2.1f%%)", ourWins1K, ((float) ourWins1K) / 1000.0 * 100.0);
			printf("    Genifer moves = %d\n", ourMoves1K);
			ourWins1K = 0;
			ourMoves1K = 0;
			}
		if (playTimes >= totalGames)
			break;
		if (userKey == 1)
			break;
		}

	// cout << "\n\nSaving RL values...\n";
	// saveStatesToFile("ttt1.dat", states1, V1);
	// saveStatesToFile("ttt2.dat", states2, V2);

	cout << "\n\nSaving NN learner values...\n";
	save_Qnet("Q.net");

	cout << "\n\nGame stats:\n";
	printf("Genifer (1) wins %d (%2.1f%%)\n", numPlayer1Won, ((float) numPlayer1Won) / totalGames * 100.0);
	printf("Player (-1) Wins %d (%2.1f%%)\n", numPlayer_1Won, ((float) numPlayer_1Won) / totalGames * 100.0);
	printf("           Draws %d (%2.1f%%)\n", numDraws, ((float) numDraws) / totalGames * 100.0);

	beep();
	pause_graphics();
	return 0;
	}

