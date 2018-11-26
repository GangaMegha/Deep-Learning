
import numpy as np
import argparse
print('Importing tensorflow')
import tensorflow as tf
import pandas as pd
import math 
import matplotlib.pyplot as plt  
print('imported tensorflow')
from config import *

# Function for acquiring parsed arguments
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', action="store", dest="lr", default=0.01, type = float)
  parser.add_argument('--batch_size', action="store", dest="batch_size", default=20,type = int)
  parser.add_argument('--init', action="store", dest="init", default=1, type = int)
  parser.add_argument('--save_dir', action="store", dest="save_dir", default='', type = str)
  parser.add_argument('--no_of_neurons', action="store", dest="no_of_neurons", default='', type = int)
  parser.add_argument('--non_linearity', action="store", dest="non_linearity", default='', type = int)
  
  return vars(parser.parse_args())

hyper_params = parse_arguments()
fc_size_1 = hyper_params['no_of_neurons']
fc_size_2 = hyper_params['no_of_neurons']
print('parsed arguments') 

def activation(layer):
  if hyper_params['non_linearity'] == 1:
    return tf.nn.relu(layer)
  if hyper_params['non_linearity'] == 2:
    return tf.nn.elu(layer)
  
# Function defining which initialisation technique to use
def initialise(option=1):
  # He initialiser
  if option == 1:
    return tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
  # Xavier initialiser
  if option == 2:
    return tf.contrib.layers.xavier_initializer()
  
# Function to add a new convolution layer to the architecture
def add_conv(input_layer, input_depth, filter_size, num_filters):
  

  bias = tf.Variable(tf.constant(0.05, shape=[num_filters]))
  weights = tf.Variable(initializer([filter_size, filter_size, input_depth, num_filters]))

  layer = tf.nn.conv2d(input=input_layer,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
  
    # Activation function used depend on non-linearity option
  layer = activation(layer+bias)

  return layer, weights

  
# Function to add max-pooling layer to the architecture
def max_pooling(layer):
  layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
                           
  return layer

def flatten(layer):
    dim = layer.get_shape()
    
    features = dim[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, features])

    return layer_flat, features
   
# Function to add a fully connected layer to the architecture                       
def add_fc_layer(input_layer, input_length, output_length):

  bias = tf.Variable(tf.constant(0.05, shape=[output_length]))
  weights = tf.Variable(initializer([input_length, output_length]))
                         
  layer = tf.matmul(input_layer, weights) + bias
                         
  return layer

# Plot the weights of a CNN layer 
def plot_conv_weights(weights):
    values = session.run(weights)

    
    grid_size = 8
    
    # grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<64:
            filter_i = values[:, :, 0, i]
            
            # Plot image.
            ax.imshow(img, vmin=np.min(values), vmax=np.max(values),interpolation='nearest', cmap='seismic')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
    
    

#print('Setting up graph')
configure = tf.ConfigProto()
configure.gpu_options.allow_growth = True

with tf.device('/gpu:0'):
 
  x = tf.placeholder(tf.float32, shape=[None, img_flat], name='image')

  x_image = tf.reshape(x, [-1, img_size, img_size, 1])

  y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='true_labels')

  y_true_cls = tf.argmax(y_true, axis=1)

  initializer = initialise()

  conv1, weights_conv1 = add_conv(input_layer=x_image, input_depth=1, filter_size=filter_size1, num_filters=num_filters1)
  conv1=max_pooling(conv1)

  conv2, weights_conv2 = add_conv(input_layer=conv1,input_depth=num_filters1,filter_size=filter_size2,num_filters=num_filters2)
  conv2=max_pooling(conv2)

  conv3, weights_conv3 = add_conv(input_layer=conv2,input_depth=num_filters2,filter_size=filter_size3,num_filters=num_filters3)

  conv4, weights_conv4 = add_conv(input_layer=conv3, input_depth=num_filters3,filter_size=filter_size4,num_filters=num_filters4)
  conv4=max_pooling(conv4)

  # conv5, weights_conv5 = add_conv(input_layer=conv4, input_depth=num_filters4,filter_size=filter_size5,num_filters=num_filters5)



  flat, num_features = flatten(conv4)



  fc1 = add_fc_layer(input_layer=flat,input_length=num_features,output_length=fc_size_1)

  fc1 = tf.nn.dropout(fc1, keep_prob=0.4, noise_shape=None, seed=1234, name="dropout_layer")
  #we need to add relu here

  fc1=activation(fc1)
  normal_1 = tf.layers.batch_normalization(inputs=fc1, axis=-1 , center=True, scale=True, training = True,name='batch_norm_1')

  fc2 = add_fc_layer(input_layer=normal_1,input_length=fc_size_1,output_length=fc_size_2)

  #we need to add relu here

  normal_logits = tf.layers.batch_normalization(inputs=fc2, axis=-1 , center=True, scale=True, training = True,name='batch_norm_for_logits')

  y_pred = tf.nn.softmax(normal_logits)

  y_pred_cls = tf.argmax(y_pred, axis=1)


  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=normal_logits,
                                                          labels=y_true)

  cost = tf.reduce_mean(cross_entropy)
    
  optimizer = tf.train.AdamOptimizer(learning_rate=hyper_params["lr"]).minimize(cost)

  correct_prediction = tf.equal(y_pred_cls, y_true_cls)
                        
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.device('cpu:0'):
  print('reading values')
  x_train = np.genfromtxt("CNN_fashionMNIST/CNN_fashionMNIST/Train.csv", delimiter=' ',missing_values="NaN").astype("float")
  x_val = np.genfromtxt("CNN_fashionMNIST/CNN_fashionMNIST/val.csv", delimiter=',',missing_values=".NaN",skip_header=1).astype("float")
  x_test = np.genfromtxt("CNN_fashionMNIST/CNN_fashionMNIST/test.csv", delimiter=',',missing_values=".NaN",skip_header=1).astype("float")
  print('read values')
  np.random.shuffle(x_train)

  train_images = x_train[:, 1:-1]
  train_images = np.multiply(train_images, 1.0/255.0)
  train_labels = np.eye(10)[np.array(x_train[:, -1]).astype("int")]
  train_ids = x_train[:,0]
  train_cls = x_train[:, -1]
                            
                            
  validation_images = x_val[:, 1:-1]
  validation_images = np.multiply(validation_images, 1.0/255.0)
  validation_labels = np.eye(10)[np.array(x_val[:, -1]).astype("int")]
  validation_ids = x_val[:,0]
  validation_cls = x_val[:, -1]

                             
  test_images = x_test[:, 1:]
  test_ids = x_test[:,0]
  test_images = np.multiply(test_images, 1.0/255.0)
  with tf.Session(config=configure) as session:

    # configure = tf.ConfigProto()
    # configure.gpu_options.allow_growth = True
    session.run(tf.global_variables_initializer())
    train_batch_size = batch_size
    saver = tf.train.Saver()
                           
    #def train() :
    t_loss=[]
    v_loss=[]
    best_val_loss = float("inf")
    patience = 0
    batch_no = 0
    batch_no_2 = 0
    epoch = 0
    i=0
    batch_size=hyper_params['batch_size']
    while i <MAX_ITER:
      i+=1

      #for j in range(3300):
        # x_train = np.genfromtxt("split_Dataset/train_"+str(j)+".csv", delimiter=' ',missing_values="NaN").astype("float")
        # train_images = x_train[:, 1:-1]
        # train_images = np.multiply(train_images, 1.0/255.0)
        # train_labels = np.eye(10)[np.array(x_train[:, -1]).astype("int")]
        # train_ids = x_train[:,0]
        # train_cls = x_train[:, -1]

                         
      if (batch_no+1)*batch_size > len(train_ids):
        batch_no = 0
        epoch = epoch + 1
        
      if (batch_no_2 +1)*batch_size > len(validation_ids):
        batch_no_2 = 0
                           
      start_indx = batch_no*batch_size
      end_indx = (batch_no+1)*batch_size
                           
      # Create batches
      x_train_batch, y_train_batch = train_images[start_indx:end_indx, :], train_labels[start_indx:end_indx]
      x_valid_batch, y_valid_batch = validation_images[batch_no_2*batch_size:batch_no_2*batch_size + batch_size, :], validation_labels[batch_no_2*batch_size:batch_no_2*batch_size + batch_size]

      # Create dictionaries for placeholders
      feed_dict_train = {x: x_train_batch,
                         y_true: y_train_batch}
      feed_dict_validate = {x: x_valid_batch,
                          y_true: y_valid_batch}

      
      # Append loss
      #t_loss.append(session.run(cost, feed_dict=feed_dict_train))
                           
      # Run optimiser
      session.run(optimizer, feed_dict=feed_dict_train)
      batch_no = batch_no + 1  
      batch_no_2 = batch_no_2 + 1

      # At the end of an epoch
      if (batch_no*batch_size)%5000 == 0: 
        # Save the model
        save_path = saver.save(session, "model/model.ckpt")
      
        # Print train progress
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        t_loss.append(session.run(cost, feed_dict=feed_dict_train))
        val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
        v_loss.append(session.run(cost, feed_dict=feed_dict_validate))
        #v_loss=session.run(cost, feed_dict=feed_dict_validate)
        
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, v_loss[-1]))
        if early_stopping: 
          if batch_no == 0:
            if v_loss[-1] < best_val_loss:
              best_val_loss = v_loss[-1]
              patience = 0
            else:
              patience += 1

            if patience == early_stopping:
              break
      # i=i+1
                         
    # Save loss
    val_accuracy=session.run(accuracy,feed_dict={x:validation_images,y_true:validation_labels})

    msg = "FINALLY Validation Accuracy: {}"
    print(msg.format(val_accuracy))

    # train_accuracy=session.run(accuracy,feed_dict={x:train_images,y_true:train_labels})

    # msg = "FINALLY Train Accuracy: {}"
    # print(msg.format(train_accuracy))

    y_predicted=session.run(y_pred_cls,feed_dict={x:test_images})
    pd.DataFrame(y_predicted).to_csv('prediction1.csv',index=True,header=['label'],index_label='id')
    pd.DataFrame(t_loss).to_csv('loss/train_loss.csv', index=False)
    pd.DataFrame(v_loss).to_csv('loss/val_loss.csv', index=False)
    #plt.
    plot_conv_weights(weights_conv1, num_filters1) 

                          
                           
# test_images = x_test[:, 1:]
# test_ids = x_test[:,0]
# test_images = np.multiply(test_images, 1.0/255.0)

#train()
   
'''
with tf.variable_scope('layer1', reuse=True):
  tf.get_variable('weights')
'''

                          
