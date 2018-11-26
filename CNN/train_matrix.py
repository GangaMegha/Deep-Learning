import glob
import os
import numpy as np
import cv2

path = '../Dataset/' 
file_names=glob.glob(os.path.join(path,'class_*','output','*'))
no_of_files=len(file_names)

# unique_classes = [int(ele.split('/')[1]) for ele in glob.glob(os.path.join(path,'class_*'))]

labels=[int(ele.split('_')[1].split('/')[0]) for ele in file_names]

images = []
for i in range(no_of_files):
    a=np.array(cv2.imread(file_names[i], 0))
    images.append(a.ravel())

images = np.array(images)

data = np.column_stack((np.arange(len(file_names)), images))

data = np.column_stack((data, labels))

np.savetxt("../Dataset/Augmented_train.csv", data)