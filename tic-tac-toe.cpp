#include <cstdlib>
using namespace std;

// int state[3][3];

#define EMPTY		0
#define PLAYER_X	1
#define PLAYER_O	2
#define DRAW		3

class Agent {

        public:

        Agent();

        void add(int state[3][3]);

		int player;

        private:

		bool verbose;
		bool learning;
		double epsilon;

};

void Agent::add(int state[3][3])
{
	
}

int game_over(int state[3][3])
	{
    for (int i = 0; i < 3; ++i)
		{
        if (state[i][0] != EMPTY && state[i][0] == state[i][1] && state[i][0] == state[i][2])
            return state[i][0];
        if (state[0][i] != EMPTY && state[0][i] == state[1][i] && state[0][i] == state[2][i])
            return state[0][i];
		}

    if (state[0][0] != EMPTY && state[0][0] == state[1][1] && state[0][0] == state[2][2])
        return state[0][0];
    if (state[0][2] != EMPTY && state[0][2] == state[1][1] && state[0][2] == state[2][0])
        return state[0][2];

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (state[i][j] == EMPTY)
                return EMPTY;
    return DRAW;
	}

int last_to_act(int state[3][3])
	{
    int countx = 0;
    int counto = 0;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (state[i][j] == PLAYER_X)
                countx += 1;
			else if (state[i][j] == PLAYER_O)
                counto += 1;

	if (countx == counto)
        return PLAYER_O;
    if (countx == (counto + 1))
        return PLAYER_X;
    return -1;
	}

int enumstates(int state[3][3], int idx, Agent agent)
	{
    if (idx > 8)
		{
		int player = last_to_act(state);
        if (player == agent.player)
            agent.add(state);
		}
    else
		{
        int winner = game_over(state);
        if (winner != EMPTY)
            return 0;
        int i = idx / 3;
        int j = idx % 3;
        for (int val = 0; val < 3; ++val)
			{
            state[i][j] = val;
            enumstates(state, idx + 1, agent);
			}
		}
	}

extern "C" int tic_tac_toe_test()
{
	return 0;
}