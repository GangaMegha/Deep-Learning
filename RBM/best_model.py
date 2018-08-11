

import numpy as np 
import pickle 
import random
#Hyper parameters
restore = False
n = 256 #size of hidden representation
k = 1 #k-step
m = 784 # size of visible varables(fixed)
lr = 0.001
iterations = 2200000
epoch = 0
#Initialise the required variables
# if restore == True:
#   with open('cd_rbm_1.pkl',mode = 'rb') as f:
#     W,b,c = pickle.load(f)

#Load data
def sigmoid(x):
  #Sigmoid activation
  #Implemented interms  of tanh for increased stability
  return .5 * (1 + np.tanh(.5 * x))

def reconstruction_val():
	v_rep = []
	for v in val :
		p_h_v = sigmoid( np.dot(W, v).T + c)
		h = np.zeros( [1, n] )
		h_sample = np.random.uniform(0, 1, [1, n])
		# print(h_sample.shape)
		h[np.where(h_sample < p_h_v)] = 1  

		p_v_h = sigmoid(np.dot(h,W) + b)
		v2 = np.zeros([m,1])
		# print(p_v_h.shape)
		v_sample = np.random.uniform(0,1,[m,1])
		# print(v_sample.shape)
		v2[np.where(v_sample < p_v_h.T)] = 1
		v_rep.append(v2)

	v_rep = np.array(v_rep)
	v_rep = v_rep.reshape(v_rep.shape[0], v_rep.shape[1])

	error = (np.sum((val-v_rep)**2)*1.0/val.shape[0])
	return error


def reconstruction_train():
	indx_train = random.sample(range(D.shape[0]), len(val))
	test = D[indx_train]
	v_rep = []

	for v in test :
		p_h_v = sigmoid( np.dot(W, v).T + c)
		h = np.zeros( [1, n] )
		h_sample = np.random.uniform(0, 1, [1, n])
		# print(h_sample.shape)
		h[np.where(h_sample < p_h_v)] = 1  

		p_v_h = sigmoid(np.dot(h,W) + b)
		v2 = np.zeros([m,1])
		# print(p_v_h.shape)
		v_sample = np.random.uniform(0,1,[m,1])
		# print(v_sample.shape)
		v2[np.where(v_sample < p_v_h.T)] = 1
		v_rep.append(v2)

	v_rep = np.array(v_rep)
	v_rep = v_rep.reshape(v_rep.shape[0], v_rep.shape[1])

	error = (np.sum((test-v_rep)**2)*1.0/test.shape[0])
	return error

def image_reconstruct() :
	# Random Image
	v1 = np.random.randint(2, size=val.shape[1])

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
	v2 = val[0,:]


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
	v3 = val[1,:]
	
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

	return v1,v2,v3



D = np.loadtxt('data_bin/train.csv',delimiter=',')
val = np.loadtxt('data_bin/val.csv',delimiter=',')

epoch = 0
min_loss = 1000000

W = np.random.randn(n,m)/np.sqrt(0.5*(n+m)) #weights

b = np.zeros([1,m]) #bias visible
c = np.zeros([1,n]) #bias latent

cost_train = []
cost_val = []
image_1 = []
image_2 = []
image_3 = []

idx = 0

for i in range(iterations):
	# print(i)
	v0 = v = D[idx,:].reshape([m,1])
	idx +=1

	if(idx>=D.shape[0]) :
	  idx = 0

	for t in range(k+1):

	  p_h_v = sigmoid(np.dot(W,v).T+c)
	  # print(p_h_v.shape)
	  h = np.zeros([1,n])
	  h_sample = np.random.uniform(0,1,[1,n])
	  # print(h_sample.shape)
	  h[np.where(h_sample < p_h_v)] = 1
	  # print(h)

	  p_v_h = sigmoid(np.dot(h,W)+b)
	  v = np.zeros([m,1])
	  # print(p_v_h.shape)
	  v_sample = np.random.uniform(0,1,[m,1])
	  # print(v_sample.shape)
	  v[np.where(v_sample < p_v_h.T)] = 1
	  # print(v.shape)

	p_h_v_s = sigmoid(np.dot(W,v).T+c).reshape(n,1)
	p_h_v_d = sigmoid(np.dot(W,v0).T+c).reshape(n,1)

	dW = np.dot(p_h_v_d,v0.reshape(1,m)) - np.dot(p_h_v_s,v.reshape(1,m))
	db = (v0 - v).T 
	dc = (p_h_v_d - p_h_v_s).T
	# print(W.shape,b.shape,c.shape)
	# print(dW.shape,db.shape,dc.shape)
	W = W + lr*dW
	b = b + lr*db 
	c = c + lr*dc
	# print(W.shape,b.shape,c.shape)

	  # Compute the free energy for samples of train and validation
	if i%5000==0 :

		# Free energy computed for train and validation samples
		cost_train.append(reconstruction_train())
		cost_val.append(reconstruction_val())


		# At the end of each epoch
		if i%D.shape[0]==0 :
			epoch +=1
			print("Epoch : {} \t Train cost : {} \t Validation cost : {}".format(epoch, cost_train[-1], cost_val[-1]))

			image1, image2, image3 = image_reconstruct()
			image_1.append(image1)
			image_2.append(image2)
			image_3.append(image3)

			if cost_val[-1] < min_loss :
				min_loss = cost_val[-1]

				# Pickle the model
				with open('cd_lr_{}_k_{}_n_{}_epoch_{}.pkl'.format(lr,k,n,epoch), mode='wb') as f:
					pickle.dump([W,b,c],f)

image_1 = np.array(image_1)
image_2 = np.array(image_2)
image_3 = np.array(image_3)

image_1 = image_1.reshape(image_1.shape[0], image_1.shape[1])
image_2 = image_2.reshape(image_2.shape[0], image_2.shape[1])
image_3 = image_3.reshape(image_3.shape[0], image_3.shape[1])

np.savetxt('cost_train_lr_{}_k_{}_n_{}.csv'.format(lr,k,n),cost_train,delimiter=',')
np.savetxt('cost_val_lr_{}_k_{}_n_{}.csv'.format(lr,k,n),cost_val,delimiter=',')
np.savetxt('image1_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_1, delimiter=',')
np.savetxt('image2_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_2, delimiter=',')
np.savetxt('image3_lr_{}_k_{}_n_{}.csv'.format(lr,k,n), image_3, delimiter=',')

print('DOne')
# print(W[:2,:2],W1[:2,:2])
                
