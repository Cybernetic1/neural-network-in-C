dist/arithmetic-test.o: arithmetic-test.c BPTT-RNN.h feedforward-NN.h
	gcc -c $< -o $@

dist/experiments.o: experiments.c RNN.h feedforward-NN.h
	gcc -c $< -o $@

dist/real-time-recurrent-learning.o: real-time-recurrent-learning.c RNN.h
	gcc -c $< -o $@

dist/back-prop.o: back-prop.c feedforward-NN.h
	gcc -c $< -o $@

dist/genetic-NN.o: genetic-NN.c
	gcc -c $< -o $@ -std=c99

dist/Sayaka1.o: Sayaka1.c tic-tac-toe.h
	gcc -c $< -o $@

dist/backprop-through-time.o: backprop-through-time.c BPTT-RNN.h
	gcc -c $< -o $@

dist/Jacobian-NN.o: Jacobian-NN.c Jacobian-NN.h
	gcc -c $< -o $@

dist/stochastic-forward-backward.o: stochastic-forward-backward.c BPTT-RNN.h
	gcc -c $< -o $@

dist/basic-tests.o: basic-tests.c RNN.h feedforward-NN.h
	gcc -c $< -o $@ -fpermissive

dist/visualization.o: visualization.c feedforward-NN.h BPTT-RNN.h
	gcc -c $< -o $@

dist/Chinese-test.o: Chinese-test.c
	gcc -c $< -o $@

dist/Q-learning.o: Q-learning.c feedforward-NN.h
	gcc -c $< -o $@

dist/V-learning.o: V-learning.c feedforward-NN.h
	gcc -c $< -o $@

dist/maze.o: maze.cpp
	g++ -c $< -o $@

dist/Sayaka-1.o: Sayaka-1.cpp
	g++ -c $< -o $@

dist/Sayaka-2.o: Sayaka-2.cpp
	g++ -c $< -o $@

dist/tic-tac-toe.o: tic-tac-toe.cpp
	g++ -c $< -o $@

dist/symmetric-test.o: symmetric-test.cpp feedforward-NN.h
	g++ -c $< -o $@ -fpermissive

dist/main.o: main.c feedforward-NN.h
	gcc -c $< -o $@

CFLAGS=-lSDL2 -L/usr/lib64 -lgsl -lgslcblas -lm -lsfml-window -lsfml-graphics -lsfml-system

genifer: dist/main.o dist/arithmetic-test.o dist/back-prop.o dist/visualization.o dist/Q-learning.o dist/basic-tests.o dist/symmetric-test.o dist/tic-tac-toe.o dist/backprop-through-time.o dist/maze.o dist/genetic-NN.o dist/Sayaka-1.o dist/Sayaka-2.o dist/real-time-recurrent-learning.o dist/V-learning.o dist/symmetric-test.o
	g++ -o genifer $^ $(CFLAGS)
