#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,time,csv,sys,argparse,math
import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.misc import imresize

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train AlexNet Street View House Number Classifier')
    parser.add_argument('-s','--scale', dest='scale', help='Training image downscale factor. Choose 1,4,16,64 (1,2,4,8 scaling) (default = 1(ground truth))',
                        default=1, type=int)
    parser.add_argument('-b','--mb_size', dest='mb_size', help='Minibatch size (default = 256)',
                        default=256, type=int)
    parser.add_argument('-e','--epoch', dest='epoch', help='Number of Epochs (default = 400)',
                        default=400, type=int)
    parser.add_argument('-i','--save_interval', dest='save_interval', help='Save graph every x epochs (default = 50)',
                        default=50, type=int)
    parser.add_argument('-g','--gpu_id', dest='gpu_id', help='GPU device to use (default = 0)',
                        default='0', type=str)
    parser.add_argument('-dir','--directory', dest='directory', help='Specify saving Directory (default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-m','--model_name', dest='model_name', help='Specify save model name (default = train_(type)_(scale))',
                        default=None, type=str)
    parser.add_argument('-im_p','--imdb_path', dest='imdb_path', help='Specify SHVN training imdb path(default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-im_s','--imdb_set', dest='imdb_set', help='Specify SHVN dataset to use (default = train)',
                        default='train', type=str)
    parser.add_argument('-t','--test', dest='test', help='Evaluate model with test dataset. Place train and test dataset in same path. (default = True)',
                        default=True, type=bool)                        

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    assert args.scale in [1,4,16,64],'Select Scale from [1,4,16,64]'
    assert args.imdb_set in ['train','test','extra'],'Select dataset type from [train,test,extra]'

    #Environment Initalization
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    if args.scale == 1:
        MODE = 'ground_truth'
    else:
        MODE = 'bilinear'
    if args.model_name is None:
        MODEL_NAME = 'train_%s_x%d'%(MODE,args.scale)
    else:
        MODEL_NAME = args.model_name
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print("Directory Created.")
    os.chdir(args.directory)

    #Load IMDB
    DATABASE = sio.loadmat(os.path.join(args.imdb_path, args.imdb_set+'_32x32.mat'))
    IMAGES = DATABASE['X']
    LABELS = DATABASE['y'].reshape((-1))

    FACTOR = int(pow(args.scale,0.5))
    IMG_HT, IMG_WT, N_CHANNELS, IMG_COUNT = IMAGES.shape
    MB_SIZE = args.mb_size
    EPOCH = args.epoch
    PERIOD = IMG_COUNT // MB_SIZE #Iterations per epoch
    NUM_LABELS = 10
    Chidden = [32,64,512,1024]
    SAVE_INTERVAL = args.save_interval #Save graph after x epochs

    #Formatting training labels
    LABELS[LABELS == 10] = 0
    label_onehot = np.eye(NUM_LABELS)[LABELS]

    #Formatting training images
    start_time = time.time()
    data = []
    print ("Formatting %d training dataset..." %(IMG_COUNT))
    for ind in range(IMG_COUNT):
        curr_img = IMAGES[:,:,:,ind]
        if FACTOR != 1:
            curr_img = imresize(curr_img,(IMG_HT/FACTOR,IMG_WT/FACTOR))
        data.append(curr_img)
    train_images = np.asarray(data, dtype = np.float32)
    print ("Finish formatting training dataset. Time: %.3f" % (time.time()-start_time))

    def conv2d(inp, n_filter, name, filtersize = (3,3),strides=[1, 1, 1, 1], bias = True):
        with tf.variable_scope(name):
            W = tf.Variable(tf.truncated_normal([filtersize[0], filtersize[1], inp.shape.as_list()[-1], n_filter], stddev=0.01))
            if bias:
                b = tf.Variable(tf.constant(0.1, shape=[n_filter]))
                conv = tf.nn.bias_add(tf.nn.conv2d(inp, W, strides, padding='SAME'), b)
            else:
                conv = tf.nn.conv2d(inp, W, strides, padding='SAME')
            return conv

    def fc(inp, n_filter, name, bias = True):
        with tf.variable_scope(name):
            shape = inp.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(inp, [-1, dim])
            W = tf.Variable(tf.truncated_normal([dim, n_filter], stddev=0.01))
            if bias:
                b = tf.Variable(tf.constant(0.1, shape=[n_filter]))
                fc = tf.nn.bias_add(tf.matmul(x, W), b)
            else:
                fc =tf.matmul(x, W)
            return fc

    def relu(u):
        return tf.nn.relu(u)

    def max_pool(bottom, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
        return tf.nn.max_pool(bottom, ksize, strides, padding='SAME', name=name)

    def drop(x, prob = 0.5):
        return tf.nn.dropout(x, keep_prob=prob)

    #Inputs
    x = tf.placeholder(tf.float32, shape=[None, IMG_HT/FACTOR, IMG_WT/FACTOR, N_CHANNELS], name = "x")
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name = "y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    #Layers
    conv1 = relu(conv2d(x,Chidden[0],'conv1',filtersize = (5,5)))
    pool1 = max_pool(conv1,'max_pool1') #[128,16,16,32]
    conv2 = relu(conv2d(pool1,Chidden[1],'conv2',filtersize = (5,5)))
    pool2 = max_pool(conv2,'max_pool1')#[128,8,8,64]
    fc1 = relu(fc(pool2,Chidden[2],'fc1'))
    fc1_drop = drop(fc1, keep_prob)
    fc2 = fc(fc1_drop,NUM_LABELS,'fc2')

    #Trainers
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc2), name = 'cross_entropy')
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(fc2,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name= "accuracy")
    #Tensorflow Initializer
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config = config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    loss_graph = []
    t0 = time.time()

    #Train
    for j in range(EPOCH):
      index = np.random.permutation(IMG_COUNT)
      for i in range(PERIOD):
        batch_img = train_images[index[i*MB_SIZE:(i+1)*MB_SIZE], :]
        batch_label = label_onehot[index[i*MB_SIZE:(i+1)*MB_SIZE], :]
        feed_dict = {x:batch_img, y_: batch_label, keep_prob: 1.0}
        _, train_accuracy, train_loss = sess.run([train_step, accuracy, cross_entropy], feed_dict=feed_dict)
        loss_graph.append(train_loss)
      endtime = int(time.time() - t0)
      print("Epoch %d, train acc: %.4f, loss: %.4f, time: %dm, %ds"%(j, train_accuracy,train_loss, endtime/60, endtime%60))
      if j%SAVE_INTERVAL == 0:
        saver.save(sess,MODEL_NAME, global_step= j)
        print ("Checkpoint saved.")

    saver.save(sess,MODEL_NAME) #Final Save
    print ("Final Checkpoint saved.")
    with open('./loss_log.csv','w') as myfile: #Loss Logfile
        wr = csv.writer(myfile)
        wr.writerow(loss_graph)

    if args.test:
        #Load test dataset
        DATABASE = sio.loadmat(os.path.join(args.imdb_path, 'test_32x32.mat'))
        IMAGES = DATABASE['X']
        LABELS = DATABASE['y'].reshape((-1))
        _, _, _, TEST_COUNT = IMAGES.shape
        
        #Formatting training labels
        LABELS[LABELS == 10] = 0
        label_onehot = np.eye(NUM_LABELS)[LABELS]
    
        #Formatting training images
        start_time = time.time()
        data = []
        print ("Formatting %d training dataset..." %(IMG_COUNT))
        for ind in range(IMG_COUNT):
            curr_img = IMAGES[:,:,:,ind]
            if FACTOR != 1:
                curr_img = imresize(curr_img,(IMG_HT/FACTOR,IMG_WT/FACTOR))
            data.append(curr_img)
        test_images = np.asarray(data, dtype = np.float32)
        print ("Finish formatting training dataset. Time: %.3f" % (time.time()-start_time))
        
        total_accuracy = 0
        steps = int(math.ceil(TEST_COUNT/float(MB_SIZE)))
        for j in range(steps):
            start = j*MB_SIZE
            if (j+1)*MB_SIZE > TEST_COUNT:
                end = TEST_COUNT
            else:
                end = (j+1)*MB_SIZE
        
            img_batch = test_images[start:end]
            label_batch = label_onehot[start:end]
            batch_accuracy = accuracy.eval(feed_dict={ x: img_batch, y_: label_batch, keep_prob: 1.0})
        
            print("Test batch %d:%d accuracy %g"%(start,end,batch_accuracy))
            total_accuracy += batch_accuracy*(end-start)
    
    print ("Total Accuracy: %f" %(total_accuracy/TEST_COUNT))
    print("Total time taken: %f"%(time.time()-start_time))