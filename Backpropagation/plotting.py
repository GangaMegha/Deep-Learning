import numpy as np
import matplotlib.pyplot as plt

#x1=np.loadtxt('Loss_plots/Loss_val_3_50.csv',delimiter=',')
'''
x1=np.loadtxt('plots/q2/Loss_val_2_50_ce.csv',delimiter=',')
x2=np.loadtxt('plots/q2/Loss_val_2_100_ce.csv',delimiter=',')
#x2=np.loadtxt('Loss_plots/Loss_val_3_100.csv',delimiter=',')
x3=np.loadtxt('plots/q2/Loss_val_2_200_ce_sigmoid.csv',delimiter=',')
x4=np.loadtxt('plots/q2/Loss_val_2_300_ce_sigmoid.csv',delimiter=',')

'''

x1=np.loadtxt('plots/q2/Loss_train_2_50_ce.csv',delimiter=',')
x2=np.loadtxt('plots/q2/Loss_train_2_100_ce.csv',delimiter=',')
#x2=np.loadtxt('Loss_plots/Loss_train_3_100.csv',delimiter=',')
x3=np.loadtxt('plots/q2/Loss_train_2_200_ce_sigmoid.csv',delimiter=',')
x4=np.loadtxt('plots/q2/Loss_train_2_300_ce_sigmoid.csv',delimiter=',')

print(x2.shape[0])

x1 = np.mean(x1.reshape(-1, 100), axis=1)/20
x2 = np.mean(x2.reshape(-1, 100), axis=1)/20
x3 = np.mean(x3.reshape(-1, 100), axis=1)/20
x4 = np.mean(x4.reshape(-1, 100), axis=1)/20

ax=plt.subplot(111)
labels=[str(a) for a in np.arange(21)]
plt.plot(x1, color='r', label='50 neurons')
plt.plot(x2, color='b', label='100 neurons')
plt.plot(x3, color='g', label='200 neurons')
plt.plot(x4, color='orange', label='300 neurons')
ax.set_xticks(np.linspace(0,len(x1),21))
ax.set_xticklabels(labels)
#ax.set_ylim(70,115)
ax.set_xlim(0,len(x1)*1.02)
plt.legend(loc=1)
plt.xlabel('Epoch')
#plt.ylabel('Validation Loss')
#plt.title("Validation Loss w.r.t epoch (3 hidden layers)")
plt.ylabel('Train Loss')
plt.title("Train Loss w.r.t epoch (2 hidden layers)")
plt.show()


#For 2 hidden layers 30 epochs taken	