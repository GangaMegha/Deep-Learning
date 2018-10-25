import tensorflow as tf
import math
import numpy as np
import collections
import pickle
import re
import os
from google.colab import files
import colab_utils.tboard
import matplotlib.pyplot as plt



import os

summary_file = "train/summaries_modified.txt"
enc_input_file = "train/train.combined"
table_onehot = 'train_onehot.json'
summary_in_file = "train/summaries.txt"
summary_out_file= "train/summaries_modified.txt"
summary_one_hot = 'train/summary_onehot.json'

summary_file_dev = "dev/summaries_modified.txt"
enc_input_file_dev = "dev/dev.combined"
table_onehot_dev = 'dev_onehot.json'
summary_in_file_dev = "dev/summaries.txt"
summary_out_file_dev= "dev/summaries_modified.txt"
summary_one_hot_dev = 'dev/summary_onehot.json'

enc_input_file_test = "test/test.combined"
table_onehot_test = 'test_onehot.json'

  
!ls
os.chdir('weathergov')
!ls
os.chdir('WeatherGov')
!ls

ROOT = %pwd
LOG_DIR = os.path.join(ROOT, 'log')
colab_utils.tboard.launch_tensorboard( bin_dir=ROOT, log_dir=LOG_DIR )
colab_utils.tboard.launch_tensorboard( bin_dir=ROOT, log_dir=LOG_DIR )

def vocab_gen(data,outfile,return_both_dict=False):
 
  vocab=data.split(' ')
  word_counts = collections.Counter(vocab)

  vocab_words =[''] + [word[0] for word in word_counts.most_common()]

  train_2_one_hot_vocab=dict(zip(vocab_words,list(range(len(vocab_words)))))
  if return_both_dict == True:
    
    one_hot_2_train_vocab=dict(zip(list(range(len(vocab_words))) ,vocab_words))
    return train_2_one_hot_vocab , one_hot_2_train_vocab
  
  return train_2_one_hot_vocab


def open_and_split(in_file,out_file=None,pad=False):
  with open(in_file,mode='r') as f_dec:
    raw_icase_data = f_dec.read().lower()
  split_data = raw_icase_data.split('\n')
  
  if pad == True:
    with open(out_file,mode='w') as f_dec:
      for sentence in split_data:
        f_dec.write('<GO> '+sentence+'<EOS>\n')
      
    with open(out_file,mode='r') as f_dec:
      raw_icase_data = f_dec.read().lower()
    split_data = raw_icase_data.split('\n')
    
  return split_data
    
def decode_using_dict_len(sentences,vocab,ret_len=True):
  sentence_words = [sentence.split(' ') for sentence in sentences]
  inputs2vocab=[]
  
  for sentence in sentence_words:
    line=[]
    for word in sentence:
      if word !='':
        line.append(vocab[word])
        
    if len(line)>1:
      inputs2vocab.append(line)
    
  if ret_len == True:
    sentence_length = np.array([len(sentence) for sentence in inputs2vocab])
    
    return inputs2vocab,sentence_length
  else: 
    return inputs2vocab


enc_train = open_and_split(enc_input_file)
dec_train = open_and_split(summary_in_file,out_file = summary_out_file,pad=True)
 
enc_dev = open_and_split(enc_input_file_dev)
dec_dev = open_and_split(summary_in_file_dev,out_file = summary_out_file_dev,pad=True)

enc_test = open_and_split(enc_input_file_test)
 
  
encoder_vocab,encoder_vocab_rev=vocab_gen(' '.join(enc_train),'outfile',return_both_dict=True)
summary2int , int2summary = vocab_gen(' '.join(dec_train),'outfile',return_both_dict=True)

encoder_inputs_vocab,encoder_in_length = decode_using_dict_len(enc_train,encoder_vocab,ret_len=True)
decoder_inputs_vocab,decoder_in_length = decode_using_dict_len(dec_train,summary2int,ret_len=True)

encoder_inputs_vocab_dev,encoder_in_length_dev = decode_using_dict_len(enc_dev,encoder_vocab,ret_len=True)
decoder_inputs_vocab_dev,decoder_in_length_dev = decode_using_dict_len(dec_dev,summary2int,ret_len=True)

encoder_inputs_vocab_test,encoder_in_length_test = decode_using_dict_len(enc_test,encoder_vocab,ret_len=True)




########################################## CONSTANTS ####################################################

start_token = 1
unk_token = 0
end_token = 2
Vs = 100 # Vs is the source vocabulary : Have to change value
inembsize = 256
encsize = 512
decsize = 512
batch_size = 20
outembsize = 256
max_predicted_sentence_length=150
Vs_summary = 100 # have to change
decoder_inputs_length_train = 256
hidden_units = 1024
restore = False
num_iter = 4000

tf.summary.FileWriterCache.clear()

########################################## PLACEHOLDERS  ####################################################

mode = "infer"    
#with tf.device('/gpu:0'):


#Here we accept indices instead of one_hot representation


is_train=tf.placeholder( tf.bool,name='train_or_infer')
encoder_inputs = tf.placeholder(dtype=tf.int32,
shape=(None, None), name='encoder_inputs')


encoder_inputs_length = tf.placeholder(
    dtype=tf.int32, shape=(None,), name='encoder_inputs_length')




########################################## INPUT EMBEDDING ####################################################

# Embedding
sqrt3 = math.sqrt(3)
with tf.variable_scope("encoder"):
  


  initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)  
  encoder_embeddings = tf.get_variable(name='embedding', shape=[Vs, inembsize], initializer=initializer, dtype=tf.float32)


  encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embeddings, encoder_inputs)



########################################## ENCODER ####################################################

  forward_cell = tf.nn.rnn_cell.BasicLSTMCell(encsize)
  backward_cell = tf.nn.rnn_cell.BasicLSTMCell(encsize)


  encoder_outputs, (encoder_state_fw,encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn( forward_cell, 
                                                               backward_cell, 
                                                               encoder_inputs_embedded,
                                                               sequence_length=encoder_inputs_length, 
                                                               dtype=tf.float32,
                                                               time_major=False )

  encoder_outputs = tf.concat(encoder_outputs, -1)

  W_project = tf.get_variable('W_project', shape=[2, 2*encsize, decsize],initializer=tf.contrib.layers.xavier_initializer())


    #print('attention')
########################################## ATTENTION ####################################################

  encoder_final_state_c = tf.concat((encoder_state_fw.c, encoder_state_bw.c), 1)
  decoder_init_state_c = tf.matmul(encoder_final_state_c,W_project[0])

  encoder_final_state_h = tf.concat((encoder_state_fw.h, encoder_state_bw.h), 1)
  decoder_init_state_h = tf.matmul(encoder_final_state_h,W_project[1])

  decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(
      c=decoder_init_state_c,
      h=decoder_init_state_h)

  # Create an attention mechanism
attention_mechanism =  tf.contrib.seq2seq.LuongAttention(
    encsize, encoder_outputs,
    memory_sequence_length=encoder_inputs_length)

from tensorflow.python.layers.core import Dense

cell =  tf.nn.rnn_cell.BasicLSTMCell(decsize)
cell = tf.contrib.seq2seq.AttentionWrapper(
    cell, attention_mechanism,
    attention_layer_size=decsize,
    alignment_history=True)
########################################## DECODER EMBEDDING ####################################################

# Embedding
sqrt3 = math.sqrt(3)

with tf.variable_scope('decoder'):
  decoder_inputs_index = tf.placeholder(
    dtype=tf.int32, shape=(None, None), name='decoder_inputs_index')


  decoder_inputs_length = tf.placeholder(
      dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

  decoder_inputs_length_train = decoder_inputs_length + 1

  decoder_targets_train=tf.placeholder(
      dtype=tf.int32, shape=(None, None), name='decoder_inputs')
  initializer = tf.random_uniform_initializer(-1, 1, dtype=tf.float32)  
  
  decoder_embeddings = tf.get_variable(name='embedding_', shape=[Vs_summary, outembsize], initializer=initializer, dtype=tf.float32)

  decoder_inputs_embedded = tf.nn.embedding_lookup(
      params=decoder_embeddings, ids=decoder_inputs_index)
  decoder_targets_embedded = tf.nn.embedding_lookup(
          params=decoder_embeddings, ids=decoder_targets_train)


  batch = tf.shape(encoder_inputs_length)[0]

  with tf.variable_scope('train'):
    helper_train = tf.contrib.seq2seq.TrainingHelper(
    inputs=decoder_inputs_embedded,
    sequence_length=decoder_inputs_length)
    
    decoder_train = tf.contrib.seq2seq.BasicDecoder(
    cell=cell,
    helper=helper_train,
    initial_state=cell.zero_state(batch,tf.float32).clone(cell_state=decoder_initial_state))#,
    
    decoder_outputs_train, final_state_train, sequence_length_train = tf.contrib.seq2seq.dynamic_decode(decoder_train,scope='train_dynamic',
                                                                    output_time_major=False, 
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_predicted_sentence_length-1)
    logits_train = decoder_outputs_train.rnn_output
    decoder_logits_train = tf.nn.softmax(decoder_outputs_train.rnn_output,name='train_softmax')
    
    loss_vec_train = tf.nn.sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=decoder_targets_train,
    logits=logits_train,
    name='train_loss')

    loss_train = tf.reduce_sum(loss_vec_train)/tf.cast(batch,tf.float32)
    
########################################## DECODER ####################################################

  with tf.variable_scope('infer'):

    helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding=decoder_embeddings,
    start_tokens=tf.tile([8], [batch]),
    end_token=9)
#     helper_infer.initialize(name='greedy')
    decoder_infer = tf.contrib.seq2seq.BasicDecoder(
    cell=cell,
    helper=helper_infer,
    initial_state=cell.zero_state(batch,tf.float32).clone(cell_state=decoder_initial_state))#,



    decoder_outputs_infer, final_state_infer, sequence_length_infer = tf.contrib.seq2seq.dynamic_decode(decoder_infer,scope='infer_dynamic',
                                                                    output_time_major=False, 
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_predicted_sentence_length-1)
    
    logits_infer = decoder_outputs_infer.rnn_output
    zero_pad=tf.zeros([tf.shape(logits_infer)[0], tf.shape(decoder_targets_train)[1] - tf.shape(logits_infer)[1], tf.shape(logits_infer)[2]], tf.float32) 
    
    logits_infer_modified = tf.concat([logits_infer,zero_pad],axis=1)
    
    decoder_logits_infer = tf.nn.softmax(logits_infer, name='infer_softmax')
    decoder_pred_infer = tf.argmax(decoder_logits_infer, axis=-1,
                                            name='decoder_pred_infer')

    loss_vec_infer = tf.nn.sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=decoder_targets_train,
    logits=logits_infer_modified,
    name='infer_loss')
    
    loss_infer = tf.reduce_sum(loss_vec_infer)/tf.cast(batch,tf.float32)
    a=tf.shape(loss_infer)
    b=tf.Print(a,[a])
    c=tf.add(a,a)
    




########################################## LOSS ####################################################


loss1=tf.summary.scalar('loss',loss_train)
loss2=tf.summary.scalar('loss',loss_infer)

tf.add_to_collection("decoder_pred_infer", decoder_pred_infer)

train_op = tf.contrib.layers.optimize_loss(
  loss_train,
  tf.train.get_global_step(),
  optimizer=tf.train.AdamOptimizer(learning_rate = 1e-3),
  learning_rate=1e-4, name="train_op")

infer_op = tf.contrib.layers.optimize_loss(
  loss_infer, 
  tf.train.get_global_step(),
  optimizer=tf.train.AdamOptimizer(learning_rate = 1e-3),
  learning_rate=1e-4, name="infer_op")

tf.add_to_collection("train_op", train_op)
tf.add_to_collection("infer_op", infer_op)



############################################# SAVER ###################################################

saver = tf.train.Saver()


############################################# MAIN ###################################################

with tf.device('/device:CPU:0'):
  
  init2 = tf.global_variables_initializer()
  
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  config.gpu_options.per_process_gpu_memory_fraction=0.333
  
  
  with tf.Session(config=config) as sess :
    
    sess.run(init2)
    print('done init')
    
    epoch = 0
    i = 0
    p = 0
    loss_list = []
    loss_list2 = []
      
    # Running iterations
    
    writer = tf.summary.FileWriter('log')
    writer.add_graph(sess.graph)

    for k in range(num_iter):
      
      if (i+1)*batch_size > len(encoder_in_length):
        i = 0
        epoch +=1
      
      max_indx_enc = max(encoder_in_length[int(i*batch_size):int((i+1)*batch_size)])
      max_indx_dec = max(decoder_in_length[int(i*batch_size):int((i+1)*batch_size)])
    

      enc_in = np.zeros((batch_size,max_indx_enc))
      dec_in = np.zeros((batch_size,max_predicted_sentence_length))# for infer


      for j in range(batch_size):
        enc_in[j,:encoder_in_length[i*batch_size+j]] = np.array(encoder_inputs_vocab[i*batch_size+j])
        dec_in[j,:decoder_in_length[i*batch_size+j]] = np.array(decoder_inputs_vocab[i*batch_size+j])
        
     
      diction_train_1 = {encoder_inputs : enc_in, 
              encoder_inputs_length : encoder_in_length[i*batch_size:(i+1)*batch_size],
              decoder_inputs_index : dec_in[:,:max_indx_dec],
              decoder_inputs_length : decoder_in_length[i*batch_size:(i+1)*batch_size],
              decoder_targets_train : dec_in[:,1:(max_indx_dec+1)],
              is_train : True
             }
      diction_train_0 = {encoder_inputs : enc_in, 
            encoder_inputs_length : encoder_in_length[i*batch_size:(i+1)*batch_size],
            decoder_inputs_index : dec_in[:,:-1],
            decoder_inputs_length : decoder_in_length[i*batch_size:(i+1)*batch_size],
            decoder_targets_train : dec_in[:,1:],
            is_train : False
           }

      if epoch < 3 :
        loss_value, loss_summary = sess.run([train_op,loss1], feed_dict=diction_train_1)
        writer.add_summary(loss_summary,k)
        
        
      else :
        loss_value, loss_summary2 = sess.run([infer_op,loss2], feed_dict=diction_train_0)
        writer.add_summary(loss_summary2,k)
      
        
    
      
      ############# For validation ############
      loss_value3 = 0
      if (p+1)*batch_size > len(encoder_in_length_dev):
        p = 0
        
      max_indx_enc = max(encoder_in_length_dev[int(p*batch_size):int((p+1)*batch_size)])
      max_indx_dec = max(decoder_in_length_dev[int(p*batch_size):int((p+1)*batch_size)])


      enc_in = np.zeros((batch_size,max_indx_enc))
      dec_in = np.zeros((batch_size,max_predicted_sentence_length))# for infer

      for j in range(batch_size):
        enc_in[j,:encoder_in_length_dev[p*batch_size+j]] = np.array(encoder_inputs_vocab_dev[p*batch_size+j])
        dec_in[j,:decoder_in_length_dev[p*batch_size+j]] = np.array(decoder_inputs_vocab_dev[p*batch_size+j])

      diction_val_1 = {encoder_inputs : enc_in, 
            encoder_inputs_length : encoder_in_length_dev[p*batch_size:(p+1)*batch_size],
            decoder_inputs_index : dec_in[:,:-1],
            decoder_inputs_length : decoder_in_length_dev[p*batch_size:(p+1)*batch_size],
            decoder_targets_train : dec_in[:,1:],
            is_train : 0
           }
     
      loss_value3 = sess.run(loss_infer, feed_dict=diction_val_1)

      p += 1
        
      loss_list.append(loss_value)
      loss_list2.append(loss_value3)
      
      #Save model
#       if i==0 and k!=0:
    

      i += 1
    
    save_path = saver.save(sess, "Models/model.ckpt",global_step=k)
    print("Model saved in path: %s" % save_path)

    np.savetxt("loss_val.txt", np.array(loss_list2), fmt='%5.3f', delimiter=',')
    np.savetxt("loss_train.txt", np.array(loss_list), fmt='%5.3f', delimiter=',')

    try :
      files.download('loss_val_7epoch.txt')
      files.download('loss_train_7epoch.txt')
    except :
      pass

    try :
      !zip -r model_7epoch.zip Models
      files.download('model_7epoch.zip')
    except :
      pass
    
    
  ############# For visualising attention weights ############
  
    p = 0
    
    max_indx_enc = max(encoder_in_length_dev[int(p*batch_size):int((p+1)*batch_size)])
    max_indx_dec = max(decoder_in_length_dev[int(p*batch_size):int((p+1)*batch_size)])


    enc_in = np.zeros((batch_size,max_indx_enc))
    dec_in = np.zeros((batch_size,max_predicted_sentence_length))# for infer

    for j in range(batch_size):
      enc_in[j,:encoder_in_length_dev[p*batch_size+j]] = np.array(encoder_inputs_vocab_dev[p*batch_size+j])
      dec_in[j,:decoder_in_length_dev[p*batch_size+j]] = np.array(decoder_inputs_vocab_dev[p*batch_size+j])

    diction_valid = {encoder_inputs : enc_in, 
          encoder_inputs_length : encoder_in_length_dev[p*batch_size:(p+1)*batch_size],
          decoder_inputs_index : dec_in[:,:-1],
          decoder_inputs_length : decoder_in_length_dev[p*batch_size:(p+1)*batch_size],
          decoder_targets_train : dec_in[:,1:],
          is_train : 0
         }
    
    output_val, alignments = sess.run([decoder_pred_infer, final_state_infer.alignment_history.stack()], feed_dict=diction_valid)
   
    def plot_attention(attention_map, input_tags = None, output_tags = None):    
      attn_len = len(attention_map)

      # Plot the attention_map
      plt.clf()
      f = plt.figure(figsize=(15, 10))
      ax = f.add_subplot(1, 1, 1)

      # Add image
      i = ax.imshow(attention_map[:20,:20], interpolation='nearest', cmap='Blues')

      
#       # Add labels
      ax.set_yticks(range(20))
      ax.set_yticklabels(output_tags)

      ax.set_xticks(range(20))
      ax.set_xticklabels(input_tags, rotation=45)


      plt.show()

    output_tags = [int2summary[word] for word in output_val[0,:20]]
    input_tags = [encoder_vocab_rev[word] for word in enc_in[0,:20]]

    plot_attention(alignments[:, 0, :], input_tags, output_tags)
    
    
    def plot_attention2(attention_map, input_tags = None, output_tags = None):    
      attn_len = len(attention_map)

      # Plot the attention_map
      plt.clf()
      f = plt.figure(figsize=(15, 10))
      ax = f.add_subplot(1, 1, 1)

      # Add image
      i = ax.imshow(attention_map[-20:,-20:], interpolation='nearest', cmap='Blues')

      
#       # Add labels
      ax.set_yticks(range(20))
      ax.set_yticklabels(output_tags)

      ax.set_xticks(range(20))
      ax.set_xticklabels(input_tags, rotation=45)


      plt.show()

    output_tags = [int2summary[word] for word in output_val[0,-20:]]
    input_tags = [encoder_vocab_rev[word] for word in enc_in[0,encoder_in_length_dev[0]-20:encoder_in_length_dev[0]]]

    plot_attention2(alignments[:, 0, :], input_tags, output_tags)
  
  
  
    def plot_attention3(attention_map, input_tags = None, output_tags = None):    
      attn_len = len(attention_map)

      # Plot the attention_map
      plt.clf()
      f = plt.figure(figsize=(15, 10))
      ax = f.add_subplot(1, 1, 1)

      # Add image
      i = ax.imshow(attention_map[:,:encoder_in_length_dev[0]], interpolation='nearest', cmap='Blues')

      
#       # Add labels
      ax.set_yticks(range(attention_map.shape[1]))
      ax.set_yticklabels(output_tags)

      ax.set_xticks(range(encoder_in_length_dev[0]))
      ax.set_xticklabels(input_tags, rotation=45)


      plt.show()

    output_tags = output_val[0]
    input_tags = enc_in[0,:encoder_in_length_dev[0]]

    plot_attention3(alignments[:, 0, :], input_tags, output_tags)
  
  
  ############# For validation ############
    print("\n Validating......")


    max_indx_enc = max(encoder_in_length_dev)
    max_indx_dec = max(decoder_in_length_dev)


    enc_in = np.zeros((len(encoder_in_length_dev),max_indx_enc))
    dec_in = np.zeros((len(encoder_in_length_dev),max_predicted_sentence_length))

    for j in range(len(encoder_in_length_dev)):
      enc_in[j,:encoder_in_length_dev[j]] = np.array(encoder_inputs_vocab_dev[j]).reshape(1,encoder_in_length_dev[j])
      dec_in[j,:decoder_in_length_dev[j]] = np.array(decoder_inputs_vocab_dev[j]).reshape(1,decoder_in_length_dev[j])

    diction_val = {encoder_inputs : enc_in, 
          encoder_inputs_length : encoder_in_length_dev,
          decoder_inputs_index : dec_in[:,:-1],
          decoder_inputs_length : decoder_in_length_dev,
          decoder_targets_train : dec_in[:,1:],
          is_train : 0
         }

    loss_value3 = sess.run(loss_infer, feed_dict=diction_val)
    
    
    print("Validation loss : {}".format(loss_value3))


  ############# For testing ############
    print("\n Testing......")

    p = 0
    n = int(len(encoder_in_length_test)/20)
    while p*n < len(encoder_in_length_test):

      max_indx_enc = max(encoder_in_length_test[int(p*n):int((p+1)*n)])

      enc_in = np.zeros((n,max_indx_enc))

      for j in range(n):
        enc_in[j,:encoder_in_length_test[p*n+j]] = np.array(encoder_inputs_vocab_test[p*n+j])

      diction_test = {encoder_inputs : enc_in, 
                      encoder_inputs_length : encoder_in_length_test[p*n:(p+1)*n]           
                     }

      output = sess.run(decoder_pred_infer, feed_dict=diction_test)

      p += 1
      np.savetxt("test_indx_7epoch_{}.txt".format(p), np.array(output), fmt='%d', delimiter=',')

      try :
        files.download("test_indx_7epoch_{}.txt".format(p))
      except :
        pass
