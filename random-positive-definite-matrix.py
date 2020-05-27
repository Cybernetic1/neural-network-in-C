from scipy import random
import numpy

N = 3

f = open("random-matrices.dat", "wb")

for i in range(0, N):
	A = random.rand(N,N)
	B = numpy.dot(A,A.transpose())
	print(B, end='\n\n')
	f.write(B.tostring())		# note that this is binary

f.close()
print("File written to: random-matrices.dat")
