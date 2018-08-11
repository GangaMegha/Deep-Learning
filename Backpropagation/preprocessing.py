import numpy as np

x_train = np.genfromtxt("data/train.csv", delimiter=',',missing_values="NaN",skip_header=1).astype("float")
x_val = np.genfromtxt("data/val.csv", delimiter=',',missing_values=".NaN",skip_header=1).astype("float")
x_test = np.genfromtxt("data/test.csv", delimiter=',',missing_values=".NaN",skip_header=1).astype("float")
print(x_val.shape,x_train.shape)

np.random.shuffle(x_train)
np.random.shuffle(x_val)
x_train[:,1:-1] = x_train[:,1:-1] / np.max(x_train[:,1:-1], axis=0)
x_val[:,1:-1] = x_val[:,1:-1] / np.max(x_val[:,1:-1], axis=0)
x_test[:,1:] = x_test[:,1:] / np.max(x_test[:,1:], axis=0)

#np.savetxt('small_train.csv',x_train[:10],delimiter=',',newline='\n')
#np.savetxt('small_val.csv',x_val[:10],delimiter=',',newline='\n')
np.savetxt('scale_train.csv',x_train,delimiter=',',newline='\n')
np.savetxt('scale_val.csv',x_val,delimiter=',',newline='\n')
np.savetxt('scale_test.csv',x_test,delimiter=',',newline='\n')
