import numpy as np 
import matplotlib.pyplot as plt

data1 = np.loadtxt("Best_Model/image1a_lr_0.001_k_1_n_256.csv", delimiter=',')
data2 = np.loadtxt("Best_Model/image2b_lr_0.001_k_1_n_256.csv", delimiter=',')
data3 = np.loadtxt("Best_Model/image3c_lr_0.001_k_1_n_256.csv", delimiter=',')
data4 = np.loadtxt("Best_Model/image4_lr_0.001_k_1_n_256.csv", delimiter=',')
data5 = np.loadtxt("Best_Model/image5_lr_0.001_k_1_n_256.csv", delimiter=',')
data6 = np.loadtxt("Best_Model/image6_lr_0.001_k_1_n_256.csv", delimiter=',')

i=0
for image in data1 :
	i+=1
	plt.figure(0)
	plt.imshow(image.reshape(28,28), plt.cm.gray)
	plt.savefig("Best_Model/image1a/{}.png".format(i))

i=0
for image in data2 :
	i+=1
	plt.figure(0)
	plt.imshow(image.reshape(28,28), plt.cm.gray)
	plt.savefig("Best_Model/image2b/{}.png".format(i))

i=0
for image in data3 :
	i+=1
	plt.figure(0)
	plt.imshow(image.reshape(28,28), plt.cm.gray)
	plt.savefig("Best_Model/image3c/{}.png".format(i))


i=0
for image in data4 :
	i+=1
	plt.figure(0)
	plt.imshow(image.reshape(28,28), plt.cm.gray)
	plt.savefig("Best_Model/image4/{}.png".format(i))

i=0
for image in data5 :
	i+=1
	plt.figure(0)
	plt.imshow(image.reshape(28,28), plt.cm.gray)
	plt.savefig("Best_Model/image5/{}.png".format(i))

i=0
for image in data6 :
	i+=1
	plt.figure(0)
	plt.imshow(image.reshape(28,28), plt.cm.gray)
	plt.savefig("Best_Model/image6/{}.png".format(i))



# import numpy as np 
# import matplotlib.pyplot as plt

# train = np.loadtxt("Best_Model/cost_train_lr_0.001_k_1_n_256.csv", delimiter=',')
# val = np.loadtxt("Best_Model/cost_val_lr_0.001_k_1_n_256.csv", delimiter=',')

# plt.figure(0)
# plt.plot(np.arange(len(train)), train)
# plt.xlabel("Iterations x 5000")
# plt.ylabel("Reconstruction Loss")
# plt.title("Reconstruction Train Error")
# plt.savefig("Best_Model/cost_train_lr_0.001_k_1_n_256.png")

# plt.figure(1)
# plt.plot(np.arange(len(val)), val)
# plt.xlabel("Iterations x 5000")
# plt.ylabel("Reconstruction Loss")
# plt.title("Reconstruction Validation Error")
# plt.savefig("Best_Model/cost_val_lr_0.001_k_1_n_256.png")

