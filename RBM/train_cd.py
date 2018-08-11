import numpy as np 
import pickle 
import random
#Hyper parameters
restore = False
n = 100 #size of hidden representation
k = 1 #k-step
m = 784 # size of visible varables(fixed)
lr = 0.00001
samples = 32 
iterations = 5000*100 
epoch = 0
#Initialise the required variables
# if restore == True:
#   with open('cd_rbm_1.pkl',mode = 'rb') as f:
#     W,b,c = pickle.load(f)

thresh = 7800
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

v_rep = []

print(D.shape)
for lr in [0.001]:
  for n in[256]:
    for k in [1]:
      epoch = 0
      # print(k)
      # if restore == True:
      #   with open('cd_rbm_k'+str(k)'.pkl',mode = 'rb') as f:
      #     W,b,c = pickle.load(f)
      # else:
      W = np.random.randn(n,m)/np.sqrt(0.5*(n+m)) #weights
      #W = np.zeros([n,m]) #weights
      b = np.zeros([1,m]) #bias visible
      c = np.zeros([1,n]) #bias latent
      cost_train = []
      cost_val = []
      idx = 0
      # for i in range(2):
      for i in range(iterations):
        # print(i)
        v0 = v =D[idx,:].reshape([m,1])
        idx +=1
        if(idx>=D.shape[0]) :
          idx = 0

        for t in range(k+1):
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
          # print(v.shape)

        if i%thresh==0:
          v_rep.append(v)

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

      # np.savetxt('cost_train_lr_{}_k_{}_n_{}.csv'.format(lr,k,n),cost_train,delimiter=',')
      # np.savetxt('cost_val_lr_{}_k_{}_n_{}.csv'.format(lr,k,n),cost_val,delimiter=',')

      # # Pickle the model
      # with open('cd_lr_{}_k_{}_n_{}.pkl'.format(lr,k,n), mode='wb') as f:
      #   pickle.dump([W,b,c],f)
v_rep = np.array(v_rep)
v_rep = v_rep.reshape(v_rep.shape[0],v_rep.shape[1])
np.savetxt('question2.csv',v_rep,delimiter=',')


print('DOne')
# print(W[:2,:2],W1[:2,:2])
                
