import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons

data = []
for i in range(10):
	data.append([])

# data.append([0.000000, 0.104528, 0.207912, 0.309017, 0.406736, 0.500000, 0.587785, 0.669130, 0.743144, 0.809017, 0.866025, 0.913545, 0.951056, 0.978147, 0.994522, 1.000000, 0.994522, 0.978148, 0.951057, 0.913546, 0.866026, 0.809018, 0.743146, 0.669132, 0.587787, 0.500002, 0.406739, 0.309019, 0.207914, 0.104531, 0.000003, -0.104526, -0.207909, -0.309014, -0.406734, -0.499997, -0.587783, -0.669128, -0.743143, -0.809015, -0.866024, -0.913544, -0.951055, -0.978147, -0.994521, -1.000000, -0.994522, -0.978148, -0.951058, -0.913547, -0.866028, -0.809020, -0.743148, -0.669134, -0.587789, -0.500004, -0.406741, -0.309022, -0.207917, -0.104534])
# data.append([0.866026, 0.809018, 0.743146, 0.669132, 0.587787, 0.500002, 0.406739, 0.309019, 0.207914, 0.104531, 0.000003, -0.104526, -0.207909, -0.309014, -0.406734, -0.499997, -0.587783, -0.669128, -0.743143, -0.809015, -0.866024, -0.913544, -0.951055, -0.978147, -0.994521, -1.000000, -0.994522, -0.978148, -0.951058, -0.913547, -0.866028, -0.809020, -0.743148, -0.669134, -0.587789, -0.500004, -0.406741, -0.309022, -0.207917, -0.104534, -0.000005, 0.104523, 0.207906, 0.309012, 0.406731, 0.499995, 0.587781, 0.669126, 0.743141, 0.809013, 0.866022, 0.913543, 0.951055, 0.978146, 0.994521, 1.000000, 0.994523, 0.978149, 0.951059, 0.913548])
# data.append([-0.866024, -0.913544, -0.951055, -0.978147, -0.994521, -1.000000, -0.994522, -0.978148, -0.951058, -0.913547, -0.866028, -0.809020, -0.743148, -0.669134, -0.587789, -0.500004, -0.406741, -0.309022, -0.207917, -0.104534, -0.000005, 0.104523, 0.207906, 0.309012, 0.406731, 0.499995, 0.587781, 0.669126, 0.743141, 0.809013, 0.866022, 0.913543, 0.951055, 0.978146, 0.994521, 1.000000, 0.994523, 0.978149, 0.951059, 0.913548, 0.866029, 0.809021, 0.743150, 0.669136, 0.587791, 0.500007, 0.406744, 0.309024, 0.207919, 0.104536, 0.000008, -0.104520, -0.207904, -0.309009, -0.406729, -0.499993, -0.587778, -0.669124, -0.743139, -0.809012])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig2 = plt.figure()
bgcolor = 'lightgoldenrodyellow'

f0 = 3
axfreq = fig2.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=bgcolor)
sfreq = Slider(axfreq, 'no effect', 0.1, 30.0, valinit=f0)

a0 = 5
axamp  = fig2.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=bgcolor)
samp = Slider(axamp, 'no effect', 0.1, 10.0, valinit=a0)

i0 = 0
i1 = 1
i2 = 2

def update(val):
    amp = samp.val
    freq = sfreq.val
    ax.set_xlabel(val)
    fig.canvas.draw()
sfreq.on_changed(update)
samp.on_changed(update)

rax0 = fig2.add_axes([0.1, 0.5, 0.10, 0.4], axisbg=bgcolor)
radio0 = RadioButtons(rax0, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), active=0)
def colorfunc(label):
	i0 = int(label)
	dat0 = np.array(data[i0], dtype=np.double)
	dat1 = np.array(data[i1], dtype=np.double)
	dat2 = np.array(data[i2], dtype=np.double)
	ax.scatter(dat0, dat1, dat2, c='r', marker='o')
	fig.canvas.draw()
radio0.on_clicked(colorfunc)

rax1 = fig2.add_axes([0.4, 0.5, 0.10, 0.4], axisbg=bgcolor)
radio1 = RadioButtons(rax1, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), active=1)
def colorfunc(label):
	fig.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	i1 = int(label)
	dat0 = np.array(data[i0], dtype=np.double)
	dat1 = np.array(data[i1], dtype=np.double)
	dat2 = np.array(data[i2], dtype=np.double)
	ax.scatter(dat0, dat1, dat2, c='r', marker='o')
	fig.canvas.draw()
radio1.on_clicked(colorfunc)

rax2 = fig2.add_axes([0.7, 0.5, 0.10, 0.4], axisbg=bgcolor)
radio2 = RadioButtons(rax2, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), active=2)
def colorfunc(label):
	fig.clf()
	i2 = int(label)
	dat0 = np.array(data[i0], dtype=np.double)
	dat1 = np.array(data[i1], dtype=np.double)
	dat2 = np.array(data[i2], dtype=np.double)
	ax.scatter(dat0, dat1, dat2, c='r', marker='o')
	fig.canvas.draw()
radio2.on_clicked(colorfunc)

resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=bgcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

def plot_scatter():
	dat0 = np.array(data[i0], dtype=np.double)
	dat1 = np.array(data[i1], dtype=np.double)
	dat2 = np.array(data[i2], dtype=np.double)

	ax.scatter(dat0, dat1, dat2, c='r', marker='o')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()

if __name__ == "__main__":
	counter = 0

	while True:
		try:
			tmp = raw_input().strip().split()
			for i in range(10):
				data[i].append(tmp[i])
		except EOFError:
			print "Input has terminated."
			plot_scatter()
			exit()
		except ValueError:
			print "Invalid input, skipping.  Input was: %s"%tmp
			continue
 
		print "Recieved data block: %d"%counter
		counter += 1

quit()
#########################################################
