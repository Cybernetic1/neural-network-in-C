// Implements the RTRL algorithm:

// First, some notation:

// Neurons are indexed by layers l, with a total of L layers
// Each layer contains N_l neurons indexed by n
// Output of each neuron is denoted Y(t) where t is the time step

// The weights W_i,j is FROM unit j TO unit i

// Each neuron recieves a weighted sum of inputs:
//		net_k(t) = sum W_ij Y(t)
// or with full indices:
//		net_[l,n](t) = sum_m W_[l,n],[l-1,m] Y_[l-1,m](t) 

// The above weighted sum then passes through the sigmoid function
//		Y_k(t+1) = sigmoid (net_k(t))
// where k is a generic index

// The individual ERROR is calculated at the output layer L:
//		e_k(t) = target_k(t) - Y_k(t)
// The total error for a single time step is 1/2 the squared sum of this.
// Our TARGET ERROR (ET) function is the integral (sum) over all time steps.

// The gradient of ET is the gradient for the current time step plus the gradient of
// previous time steps:
//		∇ ET(t0, t+1) = ∇ ET(t0,t) + ∇ E(t+1)

// As a time series is presented to the network, we can accumulate the values of the
// gradient, or equivalently, of the weight changes. We thus keep track of the value:
//		∆ W_ij(t) = -η ∂E(t)/∂W_ij
// After the network has been presented with the whole series, we alter each weight W by:
//		sum (over t) ∆ W_ij(t)

// We therefore need an algorithm that computes:
//		....

// Computation of ∂Y(t)/∂W

// ∂Y_k(t+1)/∂W_ij = sigmoid' (net_k(t)) [ sum_h W_kh ∂Y_h(t)/∂W_ij + δ_ik Y_j(t)]
