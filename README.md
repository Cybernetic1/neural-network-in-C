# genifer5C
Genifer 5 prototype written in C

Dependencies:

- SDL2 (Simple DirectMedia Layer v2, for graphics plot)

You may use the NetBeans IDE, with C/C++ extensions.  Or compile from command line:

    gcc main.c back-prop.c graphics.c Q-learning.c -std=c99 -lm -lSDL2 -o genifer

Below are some images of the cognitive state vector K moving about chaotically when acted on by a random-weight recurrent NN.

This one shows convergence towards a fixed-point:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-fixed-point.png" title="Fixed point"/>

This one shows typical quasi-periodic behavior:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-quasi-orbit.png" title="Quasi-periodic"/>

The color boxes reveals that all the components share roughly the same quasi-period:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-color-boxes.png" title="Color boxes showing regularity of orbit"/>

This is just a single component of the K vector:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-1-component.png" title="Single component of the K vector"/>


