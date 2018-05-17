/*
 * File:   tic-tac-toe.h
 * Author: yky
 *
 * Created on July 22, 2016, 2:06 PM
 */

#ifndef TIC_TAC_TOE_H
#define	TIC_TAC_TOE_H

#ifdef	__cplusplus
extern "C" {
#endif

	struct State {
		int x[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	};

	// establish ordering among board states
	struct smaller {

		bool operator()(const State s1, const State s2) const {
			for (int i = 0; i < 9; ++i)
				if (s1.x[i] > s2.x[i])
					return true;
				else if (s1.x[i] < s2.x[i])
					return false;
			return false;
		}
	};

#ifdef	__cplusplus
}
#endif

#endif	/* TIC_TAC_TOE_H */

