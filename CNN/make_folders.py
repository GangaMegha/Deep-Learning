import numpy as np
import cv2

x_train = np.genfromtxt("../Dataset/train.csv", delimiter=',',missing_values="NaN",skip_header=1)


train_images = np.reshape(x_train[:, 1:-1], (len(x_train),28,28))

labels = x_train[:,-1]
train_ids = x_train[:,0]

for i in range(len(labels)):
	if i%100==0 : 
		print(i)
	# cv2.imwrite('../Dataset/class_{}/{}.jpg'.format(labels[i], train_ids[i]),np.reshape(train_images[i,:,:],(28,28)))
	if labels[i]==0:
		cv2.imwrite("../Dataset/class_0/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==1:
		cv2.imwrite("../Dataset/class_1/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==2:
		cv2.imwrite("../Dataset/class_2/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==3:
		cv2.imwrite("../Dataset/class_3/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==4:
		cv2.imwrite("../Dataset/class_4/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==5:
		cv2.imwrite("../Dataset/class_5/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==6:
		cv2.imwrite("../Dataset/class_6/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==7:
		cv2.imwrite("../Dataset/class_7/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==8:
		cv2.imwrite("../Dataset/class_8/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
	if labels[i]==9:
		cv2.imwrite("../Dataset/class_9/" + str(train_ids[i]) + ".jpg", train_images[i,:,:])
