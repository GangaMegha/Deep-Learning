# import pickle
# import numpy as np 
# import matplotlib.pyplot as plt

# def sigmoid(x):
#   #Sigmoid activation
#   #Implemented interms  of tanh for increased stability
#   return .5 * (1 + np.tanh(.5 * x))

# # load the model from disk

# Models = ["1e-5_pickle_files/cd_lr_1e-05_k_1_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_2_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_4_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_8_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_16_n_64.pkl",
# 			"1e-5_pickle_files/cd_lr_1e-05_k_32_n_64.pkl"]

# test = np.loadtxt("data_bin/train.csv", delimiter=',')
# train = test[:10000,:]
# n = 64
# m = 784

# error = []

# for model in Models :
# 	model = pickle.load(open(model, 'rb'))

# 	v_rep = []
# 	for v in train :
# 		p_h_v = sigmoid( np.dot(model[0], v).T + model[2] )
# 		h = np.zeros( [1, n] )
# 		h_sample = np.random.uniform(0, 1, [1, n])
# 		# print(h_sample.shape)
# 		h[np.where(h_sample < p_h_v)] = 1  

# 		p_v_h = sigmoid(np.dot(h,model[0]) + model[1])
# 		v2 = np.zeros([m,1])
# 		# print(p_v_h.shape)
# 		v_sample = np.random.uniform(0,1,[m,1])
# 		# print(v_sample.shape)
# 		v2[np.where(v_sample < p_v_h.T)] = 1
# 		v_rep.append(v2)

# 	v_rep = np.array(v_rep)
# 	v_rep = v_rep.reshape(v_rep.shape[0], v_rep.shape[1])

# 	error.append(np.sum((train-v_rep)**2)*1.0/train.shape[0])


# plt.semilogx([1,2,4,8,16,32], error)
# plt.title("train_cd_lr_0.00001_n_64")
# plt.xlabel("k")
# plt.ylabel("Reconstruction error")
# plt.xticks([1,2,4,8,16,32], ('1', '2', '4', '8', '16', '32'))
# plt.savefig("cd/Plots/train_cd_lr_0.00001_n_64.png")

# # np.savetxt('hidden_rep.csv', hidden_rep, delimiter=',')



import pickle
import numpy as np 
import matplotlib.pyplot as plt

def sigmoid(x):
  #Sigmoid activation
  #Implemented interms  of tanh for increased stability
  return .5 * (1 + np.tanh(.5 * x))

# load the model from disk

Models = ["Best_Model/Models/cd_lr_0.001_k_1_n_256_epoch_40.pkl"]

val = np.loadtxt("data_bin/val.csv", delimiter=',')
# train = test[:10000,:]
n = 256
m = 784

error = []

i = 0

for model in Models :
	model = pickle.load(open(model, 'rb'))

	# for v in val[:9,:] :
	# 	p_h_v = sigmoid( np.dot(model[0], v).T + model[2] )
	# 	h = np.zeros( [1, n] )
	# 	h_sample = np.random.uniform(0, 1, [1, n])
	# 	# print(h_sample.shape)
	# 	h[np.where(h_sample < p_h_v)] = 1  
		
	# 	plt.figure(0)
	# 	plt.imshow(h.reshape(16,16), plt.cm.gray)
	# 	plt.savefig("Best_Model/hidden_i_{}.png".format(i))
	# 	plt.figure(0)
	# 	plt.imshow(v.reshape(28,28), plt.cm.gray)
	# 	plt.savefig("Best_Model/visual_i_{}.png".format(i))
	# 	i+=1

	plt.figure(0)
	plt.imshow(model[0].reshape(784,256)*255, cmap='hot', interpolation='nearest')
	plt.savefig("Best_Model/weights.png")

