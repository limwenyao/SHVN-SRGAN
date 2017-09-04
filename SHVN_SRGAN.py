#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys,argparse,math,time
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
DEFAULT = os.getcwd()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train SRGAN Street View House Number Image Regenerator')
    parser.add_argument('-hr','--highres', dest='highres', help='Target High Resolution Image for Low Resolution image to model after. Choose 1,4,16 (1,2,4 scaling) (default = 1)',
                        default=1, type=int)
    parser.add_argument('-lr','--lowres', dest='lowres', help='Target Low Resolution image scale to Super Resolve. Choose 4,16,64 (2,4,8 scaling) (default = 16)',
                        default=16, type=int)
    parser.add_argument('-b','--mb_size', dest='mb_size', help='Minibatch size (default = 128)',
                        default=128, type=int)
    parser.add_argument('-e','--epoch', dest='epoch', help='Number of Epochs (default = 200)',
                        default=200, type=int)
    parser.add_argument('-i','--save_interval', dest='save_interval', help='Save graph every x epochs (default = 10)',
                        default=10, type=int)
    parser.add_argument('-g','--gpu_id', dest='gpu_id', help='GPU device to use (default = 0)',
                        default='0', type=str)
    parser.add_argument('-cd','--class_dir', dest='class_dir', help='Specify trained classifier directory (default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-cm','--class_meta', dest='class_meta', help='Specify trained classifier .meta graph filename(default = None)',
                        default=None, type=str)
    parser.add_argument('-sd','--save_dir', dest='save_dir', help='Specify saving Directory (default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-m','--model_name', dest='model_name', help='Specify save model name (default = srgan-svhn.model)',
                        default=None, type=str)
    parser.add_argument('-im_p','--imdb_path', dest='imdb_path', help='Specify SHVN training imdb path(default = current)',
                        default=os.getcwd(), type=str)
    parser.add_argument('-im_s','--imdb_set', dest='imdb_set', help='Specify SHVN training dataset to use (default = train)',
                        default='train', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    assert args.lowres in [4,16,64],'Select target Low Resolution from [4,16,64]'
    assert args.highres in [1,4,16],'Select target High Resolution from [1,4,16]'
    assert args.lowres > args.highres, 'Cannot Super Resolve image to same or lower quality! (-lr > -hr)'
    assert args.imdb_set in ['train','extra'],'Select dataset type from [train,extra]. Test reserved for validation.'

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    # model parameter
    Ghidden = [64,256]
    Dhidden = [64, 128, 256, 512, 1024]
    samples=(6,10) #number of visualization samples. row, col
    vis_samples = samples[0] * samples[1]
    EPOCH = args.epoch
    MB_SIZE = args.mb_size
    SAVE_INTERVAL = args.save_interval
    NUM_LABELS = 10
    HR_FACTOR = args.highres #1,4,16,64
    HR_SCALE = int(math.sqrt(HR_FACTOR))
    if HR_SCALE == 1:
        HR_MODE = 'ground_truth'
    else:
        HR_MODE = 'bilinear'
    LR_FACTOR = args.lowres #1,4,16,64
    LR_SCALE = int(math.sqrt(LR_FACTOR))
    LR_MODE = 'bilinear' #ground_truth, bilinear
    PIXSHUF_ITER = int(math.log(LR_SCALE,2))-int(math.log(HR_SCALE,2))

    #Training Database
    DATABASE = sio.loadmat(os.path.join(args.imdb_path, args.imdb_set+'_32x32.mat'))
    IMAGES = DATABASE['X']
    train_label = DATABASE['y'].reshape((-1))

    DB_MEAN = IMAGES.mean()
    DB_STD = IMAGES.std()
    print ('Training Image Mean:',DB_MEAN,'Std:',DB_STD)
    IMG_HT, IMG_WT, N_CHANNELS, IMG_COUNT = IMAGES.shape
    PERIOD = IMG_COUNT // MB_SIZE

    #Formatting training labels
    train_label[train_label == 10] = 0
    train_label_onehot = np.eye(NUM_LABELS)[train_label]

    #Formatting training images
    train_data = []
    train_data_lr = []
    print ("Formatting %d training dataset..." %(IMG_COUNT))
    for ind in range(IMG_COUNT):
        curr_img_hr = IMAGES[:,:,:,ind]
        curr_img_lr = IMAGES[:,:,:,ind]
        if HR_SCALE != 1:
            curr_img_hr = imresize(curr_img_hr,(IMG_HT/HR_SCALE,IMG_WT/HR_SCALE))
        if LR_SCALE != 1:
            curr_img_lr = imresize(curr_img_lr,(IMG_HT/LR_SCALE,IMG_WT/LR_SCALE))
        train_data.append(curr_img_hr)
        train_data_lr.append(curr_img_lr)
    train_data = np.asarray(train_data, dtype = np.float32)
    train_data_lr = np.asarray(train_data_lr, dtype = np.float32)
    train_data_norm = (train_data - DB_MEAN) / DB_STD
    train_data_norm_lr = (train_data_lr - DB_MEAN) / DB_STD
#    IMG_HT, IMG_WT = IMG_HT/factor,IMG_WT/factor
    print ("Finish formatting training dataset.")

    #Validation Database
    DATABASE = sio.loadmat(os.path.join(args.imdb_path, 'test_32x32.mat'))
    IMAGES = DATABASE['X']
    test_label = DATABASE['y'].reshape((-1))
    _,_,_,IMG_COUNT_TEST = IMAGES.shape

    #Formatting validation labels
    test_label[test_label == 10] = 0
    test_label_onehot = np.eye(NUM_LABELS)[test_label]

    #Formatting validation images
    test_data = []
    test_data_lr = []
    print ("Formatting %d validation dataset..." %(IMG_COUNT))
    for ind in range(IMG_COUNT_TEST):
        curr_img_hr = IMAGES[:,:,:,ind]
        curr_img_lr = IMAGES[:,:,:,ind]
        if HR_SCALE != 1:
            curr_img_hr = imresize(curr_img_hr,(IMG_HT/HR_SCALE,IMG_WT/HR_SCALE))
        if LR_SCALE != 1:
            curr_img_lr = imresize(curr_img_lr,(IMG_HT/LR_SCALE,IMG_WT/LR_SCALE))
        test_data.append(curr_img_hr)
        test_data_lr.append(curr_img_lr)
    test_data = np.asarray(test_data, dtype = np.float32)
    test_data_lr = np.asarray(test_data_lr, dtype = np.float32)
    test_data_norm = (test_data - DB_MEAN) / DB_STD
    test_data_norm_lr = (test_data_lr - DB_MEAN) / DB_STD
    print ("Finish formatting validation dataset.")

    #Import classifier pre-trained model
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    sess1 = tf.Session(config=run_config)
    os.chdir(args.class_dir) #Change to classifier directory
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

    saver.restore(sess1,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob1 = graph.get_tensor_by_name("keep_prob:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    cross_entropy = graph.get_tensor_by_name("cross_entropy:0")
    tf.reset_default_graph()

    #Filepath for saving model + images
    if args.save_dir == DEFAULT: #Create save folder if default directory
        image_path = os.path.join(args.save_dir,'SRGANmodels/train_%s_x%d/with_%s_x%d/'%(HR_MODE,HR_FACTOR,LR_MODE,LR_FACTOR))
    else:
        image_path = args.save_dir
    if args.model_name is None:
        model_prefix = 'srgan-svhn.model'
    else:
        model_prefix = args.model_name
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    os.chdir(image_path)
    print ('Save directory:',image_path)

    #Layers
    def conv2d(inp, n_filter, name, filtersize = (3,3),strides=[1, 1, 1, 1], bias = True):
        with tf.variable_scope(name):
            W = tf.Variable(tf.truncated_normal([filtersize[0], filtersize[1], inp.shape.as_list()[-1], n_filter], stddev=0.01))
            if bias:
                b = tf.Variable(tf.constant(0.1, shape=[n_filter]))
                conv = tf.nn.bias_add(tf.nn.conv2d(inp, W, strides, padding='SAME'), b)
            else:
                conv = tf.nn.conv2d(inp, W, [1,strides[0],strides[1],1], padding='SAME')
            return conv

    def residualblk(inp, activation, name):
        n_filter = inp.shape.as_list()[-1]
        conv1 = activation(bn(conv2d(inp, n_filter, name+'_conv1')))
        conv2 = conv2d(conv1, n_filter, name+'_conv2')
        return tf.add(conv2, inp)

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

    def PRelu(u): #Currently relu, consider parametric relu.
        return tf.nn.relu(u)

    def LRelu(u, a = 0.2):
        return tf.maximum(a * u, u)

    def bn(u):
        mean, variance = tf.nn.moments(u, axes=[0, 1, 2])
        return tf.nn.batch_normalization(u, mean, variance, None, None, 1e-5)

    def _phase_shift(I, r):
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))
        X = tf.split(X, a, 1)
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
        X = tf.split(X, b, 1)
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
        return tf.reshape(X, (bsize, a*r, b*r, 1))

    def SubpixelConv2d(X, r, n_out_channel):
        _err_log = ''
        if n_out_channel > 1:
            assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel, _err_log
            Xc = tf.split(X, n_out_channel, 3)
            X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
        elif n_out_channel == 1:
            assert int(X.get_shape()[-1]) == (r ** 2), _err_log
            X = _phase_shift(X, r)
        else:
            print"Error channel size"
        if _err_log != None:
            print _err_log
        return X

    #Model input placeholder
    X = tf.placeholder(tf.float32, shape=(None, IMG_WT//HR_SCALE, IMG_HT//HR_SCALE, N_CHANNELS), name = 'X')
    Z = tf.placeholder(tf.float32, shape=(None, IMG_WT//LR_SCALE, IMG_HT//LR_SCALE, N_CHANNELS), name = 'Z')
    
    #Generator
    B_ITER = 16 #Based on SRGAN paper
    G0 = PRelu(conv2d(Z, Ghidden[0], 'G0', bias=False))
    G_B = residualblk(G0, PRelu, 'G_B0') #First RB
    for ind in range(1,B_ITER): #Rest of RB
        G_B = residualblk(G_B, PRelu, 'G_B%d'%ind)

    Gsum = tf.add(bn(conv2d(G_B, Ghidden[0], 'G1')), G0)
    for ind in range(PIXSHUF_ITER): #iterate number of pix shuf based on SR scale
        G2 = conv2d(Gsum, Ghidden[1], 'G%d'%(ind+2), bias=False)
        Gsum = PRelu(SubpixelConv2d(G2, 2, Ghidden[1] // (2 ** 2)))

    G = tf.nn.tanh(conv2d(Gsum, N_CHANNELS, 'G%d'%(ind+3), filtersize=(9,9), bias=False), name = "generator")

    #Discriminator
    def discriminator(xx):
        for ind in range(4):
            if ind == 0: #Exceptional case for first single stride conv2d. No bn, no bias
                Ds1 = LRelu(conv2d(xx, Dhidden[ind], 'D%d_1'%ind, strides=[1, 1, 1, 1], bias=False))
            else:
                Ds1 = LRelu(bn(conv2d(xx, Dhidden[ind], 'D%d_1'%ind, strides=[1, 1, 1, 1])))
            xx = LRelu(bn(conv2d(Ds1, Dhidden[ind], 'D%d_2'%ind, strides=[1, 2, 2, 1])))
            
        D_fc0 = LRelu(fc(xx, Dhidden[4], 'D_fc0'))
        D_fc1 = fc(D_fc0, 1, 'D_fc1')
        return tf.nn.sigmoid(D_fc1), D_fc1

    D_fake, D_logit_fake = discriminator(G)
    D_real, D_logit_real = discriminator(X)
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    Dloss = D_loss_real + D_loss_fake

    Genloss = 1e-3 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(G, X), [1, 2, 3]))
    Gloss = Genloss + mse

    vars = tf.trainable_variables()
    Dvars = [v for v in vars if v.name.startswith("D")]
    Gvars = [v for v in vars if v.name.startswith("G")]

    Doptimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(Dloss, var_list=Dvars)
    Goptimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(Gloss, var_list=Gvars)

    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    graph1 = sess.graph
    saver_final = tf.train.Saver(max_to_keep=15)

    #Visualization (Saves a predetermined bunch of images from the training and testing set)
    #Includes HR, LR, and SR images (LR fed to generator)
    train_idx = np.random.permutation(IMG_COUNT)[:vis_samples] #Train img set
    test_idx = np.random.permutation(IMG_COUNT_TEST)[:vis_samples] #Test img set
    def save_figure(path, mode):
        if mode == 'train_hr':
            test_acc = evaluate(train_data/255.0, train_label_onehot, False)
            imgs = train_data[train_idx]/255.0
            label = train_label[train_idx]
        elif mode == 'train_lr':
            train_data_lr_32 = []
            for a in train_data_lr:
                train_data_lr_32.append(imresize(a,(IMG_HT,IMG_WT)))
            train_data_lr_32 = np.asarray(train_data_lr_32, dtype = np.float32)
            test_acc = evaluate(train_data_lr_32/255.0, train_label_onehot, False)
            imgs = train_data_lr_32[train_idx]/255.0
            label = train_label[train_idx]
        elif mode == 'train_sr':
            test_acc = evaluate(train_data_lr/255.0, train_label_onehot, True)
            imgs = sess.run(G, feed_dict={Z: train_data_norm_lr[train_idx]/255.0})
            imgs = np.asarray(((imgs * DB_STD) + DB_MEAN)/255, dtype = np.float32)
            imgs[imgs < 0] = 0
            imgs[imgs > 1] = 1
            label = train_label[train_idx]
        elif mode == 'eval_hr':
            test_acc = evaluate(test_data/255.0, test_label_onehot, False)
            imgs = test_data[test_idx]/255.0
            label = test_label[test_idx]
        elif mode == 'eval_lr':
            test_data_lr_32 = []
            for a in test_data_lr:
                train_data_lr_32.append(imresize(a,(IMG_HT,IMG_WT)))
            train_data_lr_32 = np.asarray(train_data_lr_32, dtype = np.float32)
            test_acc = evaluate(test_data_lr_32/255.0, test_label_onehot, False)
            imgs = test_data_lr_32[test_idx]/255.0
            label = test_label[test_idx]
        elif mode == 'eval_sr':
            test_acc = evaluate(test_data_lr/255.0, test_label_onehot, True)
            imgs = sess.run(G, feed_dict={Z: test_data_norm_lr[test_idx]/255.0})
            imgs = np.asarray(((imgs * DB_STD) + DB_MEAN)/255, dtype = np.float32)
            imgs[imgs < 0] = 0
            imgs[imgs > 1] = 1
            label = test_label[test_idx]
        else:
            print ("Save image mode error.")
        fig = plt.gcf()
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
        for i in range(vis_samples):
            ax = fig.add_subplot(samples[0], samples[1], i + 1)
            ax.axis("off")
            ax.set_title(label[i])
            ax.imshow(imgs[i,:,:,:])
        plt.savefig(path)
        plt.close()
        return test_acc

    #Evaluate accuracy of model prediction with entire dataset
    def evaluate(img, label, gen): 
        total_acc = 0.0
        batch_size = 128
        count,_,_,_ = img.shape
        steps = int(math.ceil(count/float(batch_size)))
        for ind in range(steps):
            start = ind * batch_size
            if (ind+1) * batch_size > count:
                end = count
            else:
                end = (ind+1) * batch_size

            img_batch = img[start:end]
            label_batch = label[start:end]
            if gen:
                img_batch = sess.run(G, feed_dict={Z: img_batch})
                img_batch = np.asarray(((img_batch * DB_STD) + DB_MEAN)/255, dtype = np.float32)
                img_batch[img_batch < 0] = 0
                img_batch[img_batch > 1] = 1
            batch_acc = sess1.run(accuracy, feed_dict={ x: img_batch, y_: label_batch, keep_prob1: 1.0})
            total_acc += batch_acc*(end-start)
        return total_acc/count

    #Train/testing
    hr_acc = save_figure(image_path + "train_HR.png", 'train_hr')
    print "HR train accuracy: %.6f"%hr_acc
    lr_acc = save_figure(image_path + "train_LR.png", 'train_lr')
    print "LR train accuracy: %.6f"%lr_acc
    hr_acc = save_figure(image_path + "eval_HR.png", 'eval_hr')
    print "HR eval accuracy: %.6f"%hr_acc
    lr_acc = save_figure(image_path + "eval_LR.png", 'eval_lr')
    print "LR eval accuracy: %.6f"%lr_acc
    t0 = time.time()
    c_logits_fake = c_logits_real = dloss = gloss = 0.0
    limit = 0 #Scheduled training. Set 0 for no schedule
    for e in range(EPOCH):
        index = np.random.permutation(IMG_COUNT)
        dtloss = gtloss = 0.0
        dskip = gskip = 0
        for i in range(PERIOD):
            gloss_flag = dloss_flag = True
            batch_train_img = train_data[index[i*MB_SIZE:(i+1)*MB_SIZE], :]
            batch_lr8_img = train_data_lr[index[i*MB_SIZE:(i+1)*MB_SIZE], :]

            if abs(dloss - gloss) > limit and limit != 0:
                if gloss > dloss:
                    dloss_flag = False
                    dskip += 1
                else:
                    gloss_flag = False
                    gskip += 1

            if dloss_flag:
                _, dloss = sess.run([Doptimizer, Dloss], feed_dict={X:batch_train_img, Z:batch_lr8_img})
            else:
                dloss= sess.run(Dloss, feed_dict={X:batch_train_img, Z:batch_lr8_img})
            if gloss_flag:
                _, gloss, generated_img = sess.run([Goptimizer, Gloss, G], feed_dict={X:batch_train_img, Z:batch_lr8_img})
            else:
                gloss, generated_img = sess.run([Gloss, G], feed_dict={X:batch_train_img, Z:batch_lr8_img})
            dtloss += dloss
            gtloss += gloss

            if math.isnan(dtloss) or math.isnan(gtloss):
                sess.run(tf.initialize_all_variables()) # initialize & retry if NaN
                print("...initialize parameters for nan...")
                dtloss = gtloss = 0.0
                
        sr_train = save_figure(image_path + "train_SR_%03d.png" % (e+1), 'train_sr')
        sr_eval = save_figure(image_path + "eval_SR_%03d.png" % (e+1), 'eval_sr')
        endtime = int(time.time() - t0)
        print("%d: dloss=%.3f, gloss=%.3f, train_pred=%.6f, eval_pred=%.6f, dskip=%d, gskip=%d, time=%dmin %ds" % (e+1, dtloss / PERIOD, gtloss / PERIOD, sr_train, sr_eval, dskip, gskip, endtime / 60,endtime % 60))
        if e%SAVE_INTERVAL == 0:
            saver_final.save(sess,model_prefix, global_step= e)
            print("Checkpoint saved.")

    saver_final.save(sess, model_prefix)