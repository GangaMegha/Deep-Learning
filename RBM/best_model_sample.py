import pickle
import numpy as np 
import matplotlib.pyplot as plt

def sigmoid(x):
  #Sigmoid activation
  #Implemented interms  of tanh for increased stability
  return .5 * (1 + np.tanh(.5 * x))

# load the model from disk

# Models = ["1e-5_pickle_files/cd_lr_1e-05_k_1_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_2_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_4_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_8_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_16_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_32_n_64.pkl"]

test = np.loadtxt("data_bin/val.csv", delimiter=',')
train = test[:10000,:]
n = 256
m = 784
k = 1

lr = 0.001

error = []

def image_reconstruct() :
	# Random Image
	v1 = test[2,:]

	p_h_v = sigmoid(np.dot(W,v1).T+c)
	# print(p_h_v.shape)
	h = np.zeros([1,n])
	h_sample = np.random.uniform(0,1,[1,n])
	# print(h_sample.shape)
	h[np.where(h_sample < p_h_v)] = 1
	# print(h)

	p_v_h = sigmoid(np.dot(h,W)+b)
	v1 = np.zeros([m,1])
	# print(p_v_h.shape)
	v_sample = np.random.uniform(0,1,[m,1])
	# print(v_sample.shape)
	v1[np.where(v_sample < p_v_h.T)] = 1


	# 1st validation image
	v2 = test[3,:]


	p_h_v = sigmoid(np.dot(W,v2).T+c)
	# print(p_h_v.shape)
	h = np.zeros([1,n])
	h_sample = np.random.uniform(0,1,[1,n])
	# print(h_sample.shape)
	h[np.where(h_sample < p_h_v)] = 1
	# print(h)

	p_v_h = sigmoid(np.dot(h,W)+b)
	v2 = np.zeros([m,1])
	# print(p_v_h.shape)
	v_sample = np.random.uniform(0,1,[m,1])
	# print(v_sample.shape)
	v2[np.where(v_sample < p_v_h.T)] = 1


	# 2nd validation image
	v3 = test[4,:]
	
	p_h_v = sigmoid(np.dot(W,v3).T+c)
	# print(p_h_v.shape)
	h = np.zeros([1,n])
	h_sample = np.random.uniform(0,1,[1,n])
	# print(h_sample.shape)
	h[np.where(h_sample < p_h_v)] = 1
	# print(h)

	p_v_h = sigmoid(np.dot(h,W)+b)
	v3 = np.zeros([m,1])
	# print(p_v_h.shape)
	v_sample = np.random.uniform(0,1,[m,1])
	# print(v_sample.shape)
	v3[np.where(v_sample < p_v_h.T)] = 1


	# 2nd validation image
	v4 = test[5,:]
	
	p_h_v = sigmoid(np.dot(W,v3).T+c)
	# print(p_h_v.shape)
	h = np.zeros([1,n])
	h_sample = np.random.uniform(0,1,[1,n])
	# print(h_sample.shape)
	h[np.where(h_sample < p_h_v)] = 1
	# print(h)

	p_v_h = sigmoid(np.dot(h,W)+b)
	v4 = np.zeros([m,1])
	# print(p_v_h.shape)
	v_sample = np.random.uniform(0,1,[m,1])
	# print(v_sample.shape)
	v4[np.where(v_sample < p_v_h.T)] = 1



	# 2nd validation image
	v5 = test[6,:]
	
	p_h_v = sigmoid(np.dot(W,v3).T+c)
	# print(p_h_v.shape)
	h = np.zeros([1,n])
	h_sample = np.random.uniform(0,1,[1,n])
	# print(h_sample.shape)
	h[np.where(h_sample < p_h_v)] = 1
	# print(h)

	p_v_h = sigmoid(np.dot(h,W)+b)
	v5 = np.zeros([m,1])
	# print(p_v_h.shape)
	v_sample = np.random.uniform(0,1,[m,1])
	# print(v_sample.shape)
	v5[np.where(v_sample < p_v_h.T)] = 1



	# 2nd validation image
	v6 = test[7,:]
	
	p_h_v = sigmoid(np.dot(W,v3).T+c)
	# print(p_h_v.shape)
	h = np.zeros([1,n])
	h_sample = np.random.uniform(0,1,[1,n])
	# print(h_sample.shape)
	h[np.where(h_sample < p_h_v)] = 1
	# print(h)

	p_v_h = sigmoid(np.dot(h,W)+b)
	v6 = np.zeros([m,1])
	# print(p_v_h.shape)
	v_sample = np.random.uniform(0,1,[m,1])
	# print(v_sample.shape)
	v6[np.where(v_sample < p_v_h.T)] = 1


	return v1,v2,v3, v4, v5, v6

image_1 = []
image_2 = []
image_3 = []
image_4 = []
image_5 = []
image_6 = []

for epoch in range(40) :

	try :
		W, b, c = pickle.load(open('Best_Model/Models/cd_lr_{}_k_{}_n_{}_epoch_{}.pkl'.format(lr,k,n,epoch), 'rb'))

		# v_rep = []
		# for v in train :
		# 	p_h_v = sigmoid( np.dot(model[0], v).T + model[2] )
		# 	h = np.zeros( [1, n] )
		# 	h_sample = np.random.uniform(0, 1, [1, n])
		# 	# print(h_sample.shape)
		# 	h[np.where(h_sample < p_h_v)] = 1  

		# 	p_v_h = sigmoid(np.dot(h,model[0]) + model[1])
		# 	v2 = np.zeros([m,1])
		# 	# print(p_v_h.shape)
		# 	v_sample = np.random.uniform(0,1,[m,1])
		# 	# print(v_sample.shape)
		# 	v2[np.where(v_sample < p_v_h.T)] = 1
		# 	v_rep.append(v2)

		# v_rep = np.array(v_rep)
		# v_rep = v_rep.reshape(v_rep.shape[0], v_rep.shape[1])

		# error.append(np.sum((train-v_rep)**2)*1.0/train.shape[0])

		image1, image2, image3, image4, image5, image6 = image_reconstruct()
		image_1.append(image1)
		image_2.append(image2)
		image_3.append(image3)
		image_4.append(image4)
		image_5.append(image5)
		image_6.append(image6)
	except :
		pass


# plt.semilogx([1,2,4,8,16,32], error)
# plt.title("train_cd_lr_0.00001_n_64")
# plt.xlabel("k")
# plt.ylabel("Reconstruction error")
# plt.xticks([1,2,4,8,16,32], ('1', '2', '4', '8', '16', '32'))
# plt.savefig("cd/Plots/train_cd_lr_0.00001_n_64.png")

# # np.savetxt('hidden_rep.csv', hidden_rep, delimiter=',')


image_1 = np.array(image_1)
image_2 = np.array(image_2)
image_3 = np.array(image_3)
image_4 = np.array(image_4)
image_5 = np.array(image_5)
image_6 = np.array(image_6)

image_1 = image_1.reshape(image_1.shape[0], image_1.shape[1])
image_2 = image_2.reshape(image_2.shape[0], image_2.shape[1])
image_3 = image_3.reshape(image_3.shape[0], image_3.shape[1])
image_4 = image_4.reshape(image_4.shape[0], image_4.shape[1])
image_5 = image_5.reshape(image_5.shape[0], image_5.shape[1])
image_6 = image_6.reshape(image_6.shape[0], image_6.shape[1])


np.savetxt('Best_Model/image1a_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_1, delimiter=',')
np.savetxt('Best_Model/image2b_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_2, delimiter=',')
np.savetxt('Best_Model/image3c_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_3, delimiter=',')
np.savetxt('Best_Model/image4_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_4, delimiter=',')
np.savetxt('Best_Model/image5_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_5, delimiter=',')
np.savetxt('Best_Model/image6_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_6, delimiter=',')