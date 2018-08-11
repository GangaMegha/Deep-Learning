import numpy as np
# import cv2 as cv

infile=['data/train.csv','data/val.csv','data/test.csv']
outfile=['data_bin/train.csv','data_bin/val.csv','data_bin/test.csv']
out_label=['data_bin/train_labels.csv','data_bin/val_labels.csv']

for i in range(2):
	print('start')
	data=np.loadtxt(infile[i],delimiter=',',skiprows=1)
	# np.random.shuffle(data)
	print(data.shape)
	# data_labels=data[:,-1]
	data_bin=np.zeros(data[:,1:-1].shape)
	data_bin[np.where(data[:,1:-1]>127)]=1

	np.savetxt(outfile[i],data_bin,delimiter=',',fmt='%d')
	# np.savetxt(out_label[i],data_labels,delimiter=',',fmt='%d')
	print('done')

