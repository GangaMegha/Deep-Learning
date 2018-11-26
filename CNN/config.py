
MAX_ITER=10000

# Conv1.
filter_size1 = 3 
num_filters1 = 64

# Conv2.
filter_size2 = 3
num_filters2 = 128

# Conv3.
filter_size3 = 3
num_filters3 = 256

# Conv4.
filter_size4 = 3
num_filters4 = 256

# Conv5.
filter_size5 = 3
num_filters5 = 256

# FC.
fc_size_1 = 1024        # Number of neurons in fully-connected layer.
fc_size_2 = 1024           # Number of neurons in fully-connected layer.

# image dimensions (only squares for now)
img_size = 28

# Size of image when flattened to a single dimension
img_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
num_classes = len(classes)

# batch size
batch_size = 100

early_stopping = 5 # None to diable
