
import numpy as np 
import matplotlib.pyplot as plt

images = np.loadtxt("question2.csv", delimiter=',')
# indx = np.loadtxt("images_indx_gibb_k_8192.csv", delimiter=',')



for i in range(len(images)):
	plt.figure(0)
	plt.imshow(images[i,:].reshape(28,28), plt.cm.gray)
	plt.savefig("question2/qn2_i_{}.png".format(i))

