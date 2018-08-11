# import numpy as np 
# import pickle 
# import random
# #Hyper parameters
# restore = False
# n = 100 #size of hidden representation
# k = 15000 #k-step
# r = 10
# m = 784 # size of visible varables(fixed)
# lr = 0.0001
# samples = 32 
# iterations = 1000
# epoch = 0
# #Initialise the required variables
# # if restore == True:
# #   with open('cd_rbm_1.pkl',mode = 'rb') as f:
# #     W,b,c = pickle.load(f)

# #Load data
# def sigmoid(x):
#   #Sigmoid activation
#   #Implemented interms  of tanh for increased stability
#   return .5 * (1 + np.tanh(.5 * x))


# def free_energy(x):
#   wx__plus_b = np.dot(W,v).T+c
#   b_v = np.dot(b,x.T)
#   big_sum = np.sum(np.log(1 + np.exp(wx__plus_b)))
#   return np.sum(-b_v-big_sum)


# D = np.loadtxt('data_bin/train.csv',delimiter=',')
# val = np.loadtxt('data_bin/val.csv',delimiter=',')

# image = []

# print(D.shape)
# for k in [11, 50, 100, 500, 1000, 5000]:
#   #image = []
#   # print(k)
#   # if restore == True:
#   #   with open('cd_rbm_k'+str(k)'.pkl',mode = 'rb') as f:
#   #     W,b,c = pickle.load(f)
#   # else:
#   W = np.random.randn(n,m)/np.sqrt(0.5*(n+m)) #weights
#   #W = np.zeros([n,m]) #weights
#   b = np.zeros([1,m]) #bias visible
#   c = np.zeros([1,n]) #bias latent
#   cost_train = []
#   cost_val = []
#   idx = 0
  
#   for i in range(iterations):
#     # print(i)
#     v0 = D[idx,:].reshape([m,1])
#     if(idx>=D.shape[0]):
#       idx = 0
#     v = np.random.randint(2, size=v0.shape[0])
#     dw = db = dc = 0
    
    
#     for t in range(k+r+1):

#       if t%k == 0 and i>998:
# 		p_h_v = sigmoid(np.dot(W,v).T+c)
# 		# print(p_h_v.shape)
# 		h = np.zeros([1,n])
# 		h_sample = np.random.uniform(0,1,[1,n])
# 		# print(h_sample.shape)
# 		h[np.where(h_sample < p_h_v)] = 1
# 		# print(h)

# 		p_v_h = sigmoid(np.dot(h,W)+b)
# 		v = np.zeros([m,1])
# 		# print(p_v_h.shape)
# 		v_sample = np.random.uniform(0,1,[m,1])
# 		# print(v_sample.shape)
# 		v[np.where(v_sample < p_v_h.T)] = 1

# 		image.append(v)
#       # print(t)
#       # print(v.shape,W.shape,(np.dot(W,v).T+c).shape)
#       p_h_v = sigmoid(np.dot(W,v).T+c)
#       # print(p_h_v.shape)
#       h = np.zeros([1,n])
#       h_sample = np.random.uniform(0,1,[1,n])
#       # print(h_sample.shape)
#       h[np.where(h_sample < p_h_v)] = 1
#       # print(h)

#       p_v_h = sigmoid(np.dot(h,W)+b)
#       v = np.zeros([m,1])
#       # print(p_v_h.shape)
#       v_sample = np.random.uniform(0,1,[m,1])
#       # print(v_sample.shape)
#       v[np.where(v_sample < p_v_h.T)] = 1
      
# #       if i<3 :
# #         if t in [32, 64, 128, 256, 512]:#, 1024, 2048, 4096, 6000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000] :
#       # print(v.shape)
      
#       if t>k : 
#         p_h_v_s = sigmoid(np.dot(W,v).T+c).reshape(n,1)
#         dw = dw - 1.0/r * np.dot(p_h_v_s,v.reshape(1,m))
#         db = db - 1.0/r * v
#         dc = dc - 1.0/r * p_h_v_s
    
#     p_h_v_d = sigmoid(np.dot(W,v0).T+c).reshape(n,1)

#     dW = dw + np.dot(p_h_v_d,v0.reshape(1,m))
#     db = db + v0
#     db = db.T
#     dc = dc + p_h_v_d
#     dc = dc.T
#     # print(W.shape,b.shape,c.shape)
#     # print(dW.shape,db.shape,dc.shape)
#     W = W + lr*dW
#     b = b + lr*db 
#     c = c + lr*dc
#     # print(W.shape,b.shape,c.shape)
    
#       # Compute the free energy for samples of train and validation
#     if i%1==0 :
#         # Indices for evaluation samples
#       indx_train = random.sample(range(D.shape[0]), samples)
#       indx_val = random.sample(range(val.shape[0]), samples)

#       # Evaluation matrix of samples from train and validation set
#       eval_train = D[indx_train]
#       eval_val = D[indx_val]

#       # Free energy computed for train and validation samples
#       cost_train.append(free_energy(eval_train))
#       cost_val.append(free_energy(eval_val))

#       # At the end of each epoch
#       if i%D.shape[0]==0 :
#         epoch +=1
#       # print("Epoch : {} \t Train cost : {} \t Validation cost : {}".format(epoch, cost_train[-1], cost_val[-1]))

# image = np.array(image)
# image = image.reshape(image.shape[0],image.shape[1])
# #np.savetxt('cost_train_'+str(k)+'.csv',cost_train,delimiter=',')
# #np.savetxt('cost_val_'+str(k)+'.csv',cost_val,delimiter=',')
# np.savetxt('images_gibbs.csv',image,delimiter=',')

#   # Pickle the model
# #  with open('cd_rbm_k'+str(k)+'.pkl',mode='wb') as f:
# #    pickle.dump([W,b,c],f)


# print('Done')
# # print(W[:2,:2],W1[:2,:2])






import numpy as np 
import pickle 
import random
#Hyper parameters
restore = False
n = 100 #size of hidden representation
k = 15000 #k-step
r = 10
m = 784 # size of visible varables(fixed)
lr = 0.0001
samples = 32 
iterations = 10000
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


def free_energy(x):
  wx__plus_b = np.dot(W,v).T+c
  b_v = np.dot(b,x.T)
  big_sum = np.sum(np.log(1 + np.exp(wx__plus_b)))
  return np.sum(-b_v-big_sum)


D = np.loadtxt('data_bin/train.csv',delimiter=',')
val = np.loadtxt('data_bin/val.csv',delimiter=',')

image = []
image_indx = []

print(D.shape)
k = 1024*8

W = np.random.randn(n,m)/np.sqrt(0.5*(n+m)) #weights
#W = np.zeros([n,m]) #weights
b = np.zeros([1,m]) #bias visible
c = np.zeros([1,n]) #bias latent
cost_train = []
cost_val = []
idx = 0

for i in range(iterations):
	# print(i)
	v0 = D[idx,:].reshape([m,1])
	if(idx>=D.shape[0]):
	  idx = 0
	v = np.random.randint(2, size=v0.shape[0])
	dw = db = dc = 0

	if i%10==0 and i!=0 and i>19000:
		k = k/2

	if k<2 :
		break
	for t in range(k+r+1):

		if i%10==0 and i>9000 and k>1:

			if t in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] :
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

				image.append(v)
				image_indx.append(np.array([i,k,t]))

		# print(t)
		# print(v.shape,W.shape,(np.dot(W,v).T+c).shape)
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

		#       if i<3 :
		#         if t in [32, 64, 128, 256, 512]:#, 1024, 2048, 4096, 6000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000] :
		# print(v.shape)

		if t>k : 
			p_h_v_s = sigmoid(np.dot(W,v).T+c).reshape(n,1)
			dw = dw - 1.0/r * np.dot(p_h_v_s,v.reshape(1,m))
			db = db - 1.0/r * v
			dc = dc - 1.0/r * p_h_v_s

	if i%10==0 and i>9000:
		img = np.zeros_like(v0)
		image.append(img)

	p_h_v_d = sigmoid(np.dot(W,v0).T+c).reshape(n,1)

	dW = dw + np.dot(p_h_v_d,v0.reshape(1,m))
	db = db + v0
	db = db.T
	dc = dc + p_h_v_d
	dc = dc.T
	# print(W.shape,b.shape,c.shape)
	# print(dW.shape,db.shape,dc.shape)
	W = W + lr*dW
	b = b + lr*db 
	c = c + lr*dc
	# print(W.shape,b.shape,c.shape)

	  # Compute the free energy for samples of train and validation
	if i%100==0 :
        # Indices for evaluation samples
		indx_train = random.sample(range(D.shape[0]), samples)
		indx_val = random.sample(range(val.shape[0]), samples)

		# Evaluation matrix of samples from train and validation set
		eval_train = D[indx_train]
		eval_val = D[indx_val]

		# Free energy computed for train and validation samples
		cost_train.append(free_energy(eval_train))
		cost_val.append(free_energy(eval_val))

		# At the end of each epoch
		if i%D.shape[0]==0 :
			epoch +=1
			print("Epoch : {} \t Train cost : {} \t Validation cost : {}".format(epoch, cost_train[-1], cost_val[-1]))

image = np.array(image)
print(image.shape)
image = image.reshape(image.shape[0],image.shape[1])

image_indx = np.array(image_indx)
image_indx = image_indx.reshape(image_indx.shape[0], image_indx.shape[1])

#np.savetxt('cost_train_'+str(k)+'.csv',cost_train,delimiter=',')
#np.savetxt('cost_val_'+str(k)+'.csv',cost_val,delimiter=',')
np.savetxt('images_gibb_k_{}.csv'.format(k),image,delimiter=',')
np.savetxt('images_indx_gibb_k_{}.csv'.format(k),image_indx,delimiter=',')

  # Pickle the model
with open('cd_rbm_k'+str(k)+'.pkl',mode='wb') as f:
   pickle.dump([W,b,c],f)


print('Done')
# print(W[:2,:2],W1[:2,:2])
