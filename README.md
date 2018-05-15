# Neural Network in C/C++

Experimental code where I test out my neural network ideas.

Installation
------------

Dependencies:

- SDL2 (Simple DirectMedia Layer v2, for graphics plot)
- GSL (GNU Scientific Library)
- SMFL (Simple and Fast Multimedia Library, for drawing maze)

To install SDL2 library:

	sudo apt-get install libsdl2-2.0-0
	sudo apt-get install libsdl2-dev
	sudo apt-get install libsdl2-mixer-dev

To install GSL library:

	sudo apt-get install libgsl-dev

To install SFML library:

	sudo apt-get install libsfml-dev

To compile:

	make genifer

For development, you may use the NetBeans IDE, with C/C++ extensions, but I have moved away from NetBeans and the config files may be old.

Screen Shots
------------

Below are some images of the cognitive state vector K moving about chaotically when acted on by a random-weight recurrent NN.

This one shows convergence towards a fixed-point:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-fixed-point.png" title="Fixed point"/>

This one shows typical quasi-periodic behavior:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-quasi-orbit.png" title="Quasi-periodic"/>

The color boxes reveals that all the components share roughly the same quasi-period:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-color-boxes.png" title="Color boxes showing regularity of orbit"/>

This is just a single component of the K vector:

<img src="https://raw.githubusercontent.com/Cybernetic1/genifer5-c/master/K-wandering-1-component.png" title="Single component of the K vector"/>
