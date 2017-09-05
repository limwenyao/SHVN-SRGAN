#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys,math,argparse
import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.misc import imresize
DEFAULT = os.getcwd()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate AlexNet Classifier and SRGAN model.')
    parser.add_argument('-s','--scale', dest='scale', help='Target scale of trained AlexNet Classifier model. Choose 1,4,16,64 (1,2,4,8 scaling) (default = 1(ground truth))',
                        default=1, type=int)
    parser.add_argument('-b','--mb_size', dest='mb_size', help='Minibatch size (default = 128)',
                        default=128, type=int)
    parser.add_argument('-gen','--generate', dest='generate', help='Use SRGAN Generator? (default = False)',
                        default=False, type=bool)
    parser.add_argument('-gs','--gen_scale', dest='gen_scale', help='Low Resolution input scale to SRGAN Generator. Has to be lower resolution than target scale. (default = 16)',
                        default=16, type=int)
    parser.add_argument('-gi','--gen_img', dest='gen_img', help='Save the entire image database of SR images fed from SRGAN? (default = False)',
                        default=False, type=bool)
    parser.add_argument('-gsd','--gen_savedir', dest='gen_savedir', help='Specify generated imdb save directory. Name saved as SRGAN_(testset)_(target_res).mat (default = current)',
                        default=os.getcwd(), type=str) 
    parser.add_argument('-g','--gpu_id', dest='gpu_id', help='GPU device to use (default = 0)',
                        default='0', type=str)
    parser.add_argument('-cd','--class_dir', dest='class_dir', help='Specify trained classifier directory (default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-cm','--class_meta', dest='class_meta', help='Specify trained classifier .meta graph filename(default = None)',
                        default=None, type=str)
    parser.add_argument('-gd','--gen_dir', dest='gen_dir', help='Specify trained SRGAN directory (default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-gm','--gen_meta', dest='gen_meta', help='Specify trained SRGAN .meta graph filename(default = None)',
                        default=None, type=str)
    parser.add_argument('-im_p','--imdb_path', dest='imdb_path', help='Specify SHVN training imdb path(default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-im_s','--imdb_set', dest='imdb_set', help='Specify SHVN dataset to use for evaluation (default = test)',
                        default='test', type=str)
                        
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    if args.generate:
        assert args.gen_scale > args.scale, 'The LR input scale to SRGAN has to be of lower resolution than Target scale of AlexNet Classifier model. --gen_scale > --scale'
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    
    IMG_SCALE = args.scale
    if args.generate: #If generator is on, switch to generator img scale
        IMG_SCALE = args.gen_scale
    MB_SIZE = args.mb_size
    TARGET_RES = 32 / int(pow(args.scale,0.5)) #SHVN = 32
    NUM_LABELS = 10
    DB_MEAN, DB_STD = (115.11177966923525,50.819267906232888) #SHVN train mean, std
    
    #Testing Database
    DATABASE = sio.loadmat(os.path.join(args.imdb_path, args.imdb_set+'_32x32.mat'))
    IMAGES = DATABASE['X'].astype(np.float32)
    LABELS = DATABASE['y'].reshape((-1))
    IMG_HT, IMG_WT, N_CHANNELS, IMG_COUNT = IMAGES.shape
    
    #Formatting labels
    LABELS[LABELS == 10] = 0
    LABELS_ONEHOT = np.eye(NUM_LABELS)[LABELS]
    
    #Formatting training images
    imdb = []
    print ("Formatting %d %s dataset to %d scale. " %(IMG_COUNT,args.imdb_set,IMG_SCALE))
    for ind in range(IMG_COUNT):
        curr_img = IMAGES[:,:,:,ind]
        if IMG_SCALE != 1:
            curr_img = imresize(curr_img,(IMG_HT/IMG_SCALE,IMG_WT/IMG_SCALE))
        imdb.append(curr_img)
    imdb = np.asarray(imdb, dtype = np.float32)
    imdb = (imdb - DB_MEAN) / DB_STD
    print ("Finish formatting dataset.")
    
    #Import classifier trained model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    os.chdir(args.class_dir)
    if args.class_meta is None:
        found = False
        for file in os.listdir(args.class_dir):
            if file.endswith('.meta'): #Import any meta file
                saver = tf.train.import_meta_graph(os.path.join(args.class_dir,file))
                found = True
                break
        if not found:
            assert 'No classifier .meta file found to import.'
    else:
        saver = tf.train.import_meta_graph(os.path.join(args.class_dir,args.class_meta))      
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    
    #Import generator model (If applicable)
    if args.generate:
        tf.reset_default_graph()
        sess_g = tf.Session(config = config)
        os.chdir(args.gen_dir)
        if args.gen_meta is None:
            found = False
            for file in os.listdir(args.gen_dir):
                if file.endswith('.meta'): #Import any meta file
                    saver = tf.train.import_meta_graph(os.path.join(args.gen_dir,file))
                    found = True
                    break
            if not found:
                assert 'No SRGAN .meta file found to import.'
        else:
            saver = tf.train.import_meta_graph(os.path.join(args.gen_dir,args.gen_meta))
        saver.restore(sess_g,tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        Z = graph.get_tensor_by_name("Z:0")
        G = graph.get_tensor_by_name("generator:0")
    
    #Eval
    total_accuracy = 0.0
    gen_imdb = np.empty([0,IMG_HT,IMG_WT,N_CHANNELS])
    steps = int(math.ceil(IMG_COUNT/float(MB_SIZE)))
    for j in range(steps):
        start = j*MB_SIZE
        if (j+1)*MB_SIZE > IMG_COUNT:
            end = IMG_COUNT
        else:
            end = (j+1)*MB_SIZE
            
        img_batch = IMAGES[start:end]
        label_batch = LABELS_ONEHOT[start:end]
        if args.generate:
            img_batch = G.eval(session = sess_g, feed_dict = {Z:img_batch})
            _,gen_shape,_,_ = img_batch.shape
            assert gen_shape == TARGET_RES, 'Generator O/P size(%d) mismatch with classifier I/P size(%d)'%(gen_shape,TARGET_RES)
            gen_imdb = np.vstack((gen_imdb,img_batch))
        
        batch_accuracy = accuracy.eval(session = sess, feed_dict= { x: img_batch, y_: label_batch, keep_prob: 1.0})
        #print("Test batch %d:%d accuracy %g"%(start,end,batch_accuracy))
        total_accuracy += batch_accuracy*(end-start)
    
    print ("Total Accuracy: %f" %(total_accuracy/IMG_COUNT))

    if args.gen_img:
        gen_imdb = (gen_imdb * DB_STD) + DB_MEAN
        gen_imdb.astype(np.int32)
        img_batch[img_batch < 0] = 0
        img_batch[img_batch > 255] = 255
        save_dict = dict(X = gen_imdb.astype(np.uint8), y = LABELS)
        save_name = 'SRGAN_(%s)_%dx%d.mat'%(args.imdb_set,TARGET_RES,TARGET_RES)
        if args.save_dir == DEFAULT: #Create save folder if default directory
            image_path = os.path.join(args.save_dir,'SRGAN_imdb')
        else:
            image_path = args.save_dir
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        os.chdir(image_path)
        sio.savemat(save_name,save_dict)
        print ('Generated dataset saved to %s'%(os.path.join(os.getcwd(),save_name)))