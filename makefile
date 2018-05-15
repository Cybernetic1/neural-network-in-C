arithmetic-test.o: arithmetic-test.c BPTT-RNN.h feedforward-NN.h
	gcc -c $<

experiments.o: experiments.c RNN.h feedforward-NN.h
	gcc -c $<

real-time-recurrent-learning.o: real-time-recurrent-learning.c RNN.h
	gcc -c $<

back-prop.o: back-prop.c feedforward-NN.h
	gcc -c $<

genetic-NN.o: genetic-NN.c
	gcc -c $< -std=c99

Sayaka1.o: Sayaka1.c tic-tac-toe.h
	gcc -c $<

backprop-through-time.o: backprop-through-time.c BPTT-RNN.h
	gcc -c $<

Jacobian-NN.o: Jacobian-NN.c Jacobian-NN.h
	gcc -c $<

stochastic-forward-backward.o: stochastic-forward-backward.c BPTT-RNN.h
	gcc -c $<

basic-tests.o: basic-tests.c RNN.h feedforward-NN.h
	gcc -c $< -fpermissive

visualization.o: visualization.c feedforward-NN.h BPTT-RNN.h
	gcc -c $<

Chinese-test.o: Chinese-test.c
	gcc -c $<

Q-learning.o: Q-learning.c feedforward-NN.h
	gcc -c $<

V-learning.o: V-learning.c feedforward-NN.h
	gcc -c $<

maze.o: maze.cpp
	g++ -c $<

Sayaka-1.o: Sayaka-1.cpp
	g++ -c $<

Sayaka-2.o: Sayaka-2.cpp
	g++ -c $<

tic-tac-toe2.o: tic-tac-toe2.cpp
	g++ -c $<

tic-tac-toe.o: tic-tac-toe.cpp
	g++ -c $<

main.o: main.c feedforward-NN.h
	gcc -c $<

CFLAGS=-lSDL2 -L/usr/lib64 -lgsl -lgslcblas -lm -lsfml-window -lsfml-graphics -lsfml-system

genifer: main.o arithmetic-test.o back-prop.o visualization.o Q-learning.o basic-tests.o tic-tac-toe2.o backprop-through-time.o maze.o genetic-NN.o Sayaka-1.o Sayaka-2.o real-time-recurrent-learning.o V-learning.o
	g++ -o genifer $^ $(CFLAGS)
