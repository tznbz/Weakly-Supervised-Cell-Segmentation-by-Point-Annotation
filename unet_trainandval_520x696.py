##-------------------------------------------------------------------
## Copyright (C) 201 Tianyi Zhao
## File : 5Conv_trainandval.py
## Author : Tianyi Zhao <tianyi@12sigmamedicalimaging.com>
## Description :
## --
## Created : <2017-05-25>
## Updated: Time-stamp: <2017-05-25 22:49:10>
##-------------------------------------------------------------------


import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
gpuid = '3'
if '-g' in sys.argv:
    gpuid = sys.argv[sys.argv.index('-g') + 1]
os.environ['CUDA_VISIBLE_DEVICES']=gpuid
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from skimage import io
import time
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import erosion,dilation,cube
from skimage.filters import gaussian
#from load_data import load_data
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True


batchsize = 4

def weight_variable(shape,stddev=0.05):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,st=1,pad=0):
    return tf.nn.conv2d(x, W, strides=[1, st, st, 1], padding='SAME')

def conv2d_BN(x,W,st=1,pad=0):
    epsilon = 1e-3
    h_conv = tf.nn.conv2d(x, W, strides=[1, st, st, 1], padding='SAME')
    batch_mean2, batch_var2 = tf.nn.moments(h_conv,[0,1,2])
    scale2 = tf.Variable(tf.ones([W.shape[-1]]))
    beta2 = tf.Variable(tf.zeros([W.shape[-1]]))
    h_fl3_BN2 = tf.nn.batch_normalization(h_conv,batch_mean2,batch_var2,beta2,scale2,epsilon)
    return h_fl3_BN2


def max_pool(x,st=2,pad=0,ks=2):
  return tf.nn.max_pool(x, ksize=[1, ks, ks, 1],
                        strides=[1, st, st, 1], padding='SAME')

  
def max_pool_all(x):
    return tf.reduce_max(x, axis=-1)
  
def avg_pool_all(x):
    return tf.reduce_mean(x, axis=-1)

def createweights():
    W_conv1 = weight_variable([3, 3, 1, 64])
    b_conv1 = bias_variable([64])
    W_conv2 = weight_variable([3, 3, 64,64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    W_conv4 = weight_variable([3, 3, 128, 128])
    b_conv4 = bias_variable([128])
    
    W_conv5 = weight_variable([3, 3, 128, 256])
    b_conv5 = bias_variable([256])
    W_conv6 = weight_variable([3, 3, 256, 256],stddev=0.01)
    b_conv6 = bias_variable([256])

    W_conv7 = weight_variable([3, 3, 256, 512],stddev=0.01)
    b_conv7 = bias_variable([512])
    W_conv8 = weight_variable([3, 3, 512, 512],stddev=0.01)
    b_conv8 = bias_variable([512])


    W_deconv9 =  weight_variable([2, 2, 256, 512],stddev=0.01)
    b_deconv9 =  bias_variable([512])
    W_conv10 = weight_variable([3, 3, 512, 256],stddev=0.001)
    b_conv10 = bias_variable([256])
    W_conv11 = weight_variable([3, 3, 256, 256],stddev=0.01)
    b_conv11 = bias_variable([256])


    W_deconv12 =  weight_variable([2, 2, 128, 256])
    b_deconv12 =  bias_variable([256])
    W_conv13 = weight_variable([3, 3, 256, 128])
    b_conv13 = bias_variable([128])
    W_conv14 = weight_variable([3, 3, 128, 128])
    b_conv14 = bias_variable([128])

    W_deconv15 =  weight_variable([2, 2, 64, 128])
    b_deconv15 =  bias_variable([128])
    W_conv16 = weight_variable([3, 3, 128, 64])
    b_conv16 = bias_variable([64])
    W_conv17 = weight_variable([3, 3, 64, 64])
    b_conv17 = bias_variable([64])

    W_conv18 =  weight_variable([1, 1, 64, 2])
    b_conv18 =  bias_variable([2])




    weights = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,W_conv6,W_conv7,W_conv8,W_deconv9,W_conv10,
		W_conv11,W_deconv12,W_conv13,W_conv14,W_deconv15,W_conv16,W_conv17,W_conv18]
    beights = [b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,b_conv6,b_conv7,b_conv8,b_deconv9,b_conv10,
		b_conv11,b_deconv12,b_conv13,b_conv14,b_deconv15,b_conv16,b_conv17,b_conv18]

    upop = tf.contrib.keras.layers.UpSampling2D((2,2),'channels_last')
    return weights,beights,upop


def ConvNetwork(x_image,weights,beights,upop):
    #x = tf.placeholder(tf.float64, shape=[None, 256,256,1])
    #xm = tf.placeholder(tf.float64, shape=[None, 32,32,1]) 

    h_conv1 = tf.nn.relu(conv2d_BN(x_image, weights[0]) + beights[0])
    h_conv2 = tf.nn.relu(conv2d_BN(h_conv1, weights[1]) + beights[1])
    h_pool2 = max_pool(h_conv2,ks=2)
    print(h_conv1,h_conv2,h_pool2)
    h_conv3 = tf.nn.relu(conv2d_BN(h_pool2, weights[2]) + beights[2])
    h_conv4 = tf.nn.relu(conv2d_BN(h_conv3, weights[3]) + beights[3])
    h_pool4 = max_pool(h_conv4,ks=2)
    print(h_conv3,h_conv4,h_pool4)

    h_conv5 = tf.nn.relu(conv2d_BN(h_pool4, weights[4]) + beights[4])
    h_conv6 = tf.nn.relu(conv2d_BN(h_conv5, weights[5]) + beights[5])
    h_pool6 = max_pool(h_conv6,ks=2)
    print(h_conv5,h_conv6,h_pool6)


    h_conv7 = tf.nn.relu(conv2d_BN(h_pool6, weights[6]) + beights[6])
    h_conv8 = tf.nn.relu(conv2d_BN(h_conv7, weights[7]) + beights[7])
    #h_pool8 = max_pool(h_conv8,ks=2)
    print(h_conv7,h_conv8)




    h_deconv9 = tf.nn.conv2d_transpose(h_conv8, weights[8],[batchsize,130,174,256], strides=[1, 2, 2, 1], padding='SAME') #+ beights[10]
    h_concat9 = tf.concat([h_conv6,h_deconv9],axis=-1)
    h_conv10 = tf.nn.relu(conv2d(h_concat9, weights[9]) + beights[9])
    h_conv11 = tf.nn.relu(conv2d(h_conv10, weights[10]) + beights[10])


    h_deconv12 = tf.nn.conv2d_transpose(h_conv11, weights[11],[batchsize,260,348,128], strides=[1, 2, 2, 1], padding='SAME') #+ beights[16]
    h_concat12 = tf.concat([h_conv4,h_deconv12],axis=-1)
    h_conv13 = tf.nn.relu(conv2d(h_concat12, weights[12]) + beights[12])
    h_conv13 = tf.nn.relu(conv2d(h_conv13, weights[13]) + beights[13])


    h_deconv14 = tf.nn.conv2d_transpose(h_conv13, weights[14],[batchsize,520,696,64], strides=[1, 2, 2, 1], padding='SAME') #+ beights[19]
    h_concat14 = tf.concat([h_conv2,h_deconv14],axis=-1)
    h_conv15 = tf.nn.relu(conv2d(h_concat14, weights[15]) + beights[15])
    h_conv16 = tf.nn.relu(conv2d(h_conv15, weights[16]) + beights[16])


    h_conv17 = (conv2d(h_conv16, weights[17]) + beights[17])
    
  

    
    #h_conv23 = (conv2d(h_conv22, weights[22]) + beights[22])


    softmax = tf.nn.softmax(h_conv17)


    return h_conv17,softmax

def dice_coieffience_loss(result_mask,x_mask):
    eps = 1e-5
    intersection =  tf.reduce_sum(result_mask * x_mask,axis=[1,2,3])
    union =  eps + tf.reduce_sum(result_mask,axis=[1,2,3]) + tf.reduce_sum(x_mask,axis=[1,2,3])
    loss = tf.reduce_mean(-2 * intersection/ (union))

    return loss


def cross_entropy_loss(result_mask,x_mask,wmask):
    #softmax = tf.reshape(result_mask,[200*16*16,2])
    #loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #        logits=tf.reshape(result_mask,[batchsize*512*512,2]),  labels=tf.reshape(x_mask,[batchsize*512*512]))
    print(wmask)
    print(result_mask)
    print(x_mask)
    loss1 = tf.reshape(wmask,[batchsize*520*696])*tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.reshape(result_mask,[batchsize*520*696,2]),  labels=tf.reshape(x_mask,[batchsize*520*696]))
    print('lossssss',loss1)
    loss = tf.reduce_sum(loss1)
    
    return loss/batchsize/1000


def cross_entropy_prob_loss(result_mask, x_mask, wmask):
    # softmax = tf.reshape(result_mask,[200*16*16,2])
    # loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #        logits=tf.reshape(result_mask,[batchsize*512*512,2]),  labels=tf.reshape(x_mask,[batchsize*512*512]))
    print(wmask)
    print(result_mask)
    print(x_mask)
    loss1 = tf.reshape(wmask, [batchsize * 520 * 696]) * tf.nn.softmax_cross_entropy_with_logits(
        logits=tf.reshape(result_mask, [batchsize * 520 * 696, 2]), labels=tf.reshape(x_mask, [batchsize * 520 * 696,2]))
    print('lossssss', loss1)
    loss = tf.reduce_sum(loss1)

    return loss / batchsize / 1000



def my_cross_entropy_loss(result_mask,x_mask):
    softmax = tf.nn.softmax(result_mask)
    imask = tf.concat([1-x_mask,x_mask],axis=-1)
    loss = tf.reduce_mean(tf.reduce_sum(-imask*tf.log(softmax),axis=[1,2,3]))
    return loss,softmax


def augmentation(images,labels):
    if np.random.random()>0.5:
        images = images[:,::-1]
        labels = labels[:,::-1]
    if np.random.random()>0.5:
        images = images[:,:,::-1]
        labels = labels[:,:,::-1]
    '''if np.random.random()>0.5: #ppadding
        scale =  int(np.random.random() * 20)
        images[:,scale:(images.shape[1]-scale),scale:(images.shape[2]-scale)] = zoom(images,(1,(images.shape[1]-2*scale)/images.shape[1],(images.shape[2]-2*scale)/images.shape[2],1),order=1 )
        labels[:,scale:(labels.shape[1]-scale),scale:(labels.shape[2]-scale)] = zoom(labels,(1,(labels.shape[1]-2*scale)/labels.shape[1],(labels.shape[2]-2*scale)/labels.shape[2],1),order=1 )
        labels[labels!=0] = 1
    else:
        scale =  int(np.random.random() * 20)
        images = zoom(images[:,scale:(images.shape[1]-scale),scale:(images.shape[2]-scale)],(1,images.shape[1]/(images.shape[1]-2*scale),images.shape[2]/(images.shape[2]-2*scale),1),order=1 )
        labels = zoom(labels[:,scale:(labels.shape[1]-scale),scale:(labels.shape[2]-scale)],(1,labels.shape[1]/(labels.shape[1]-2*scale),labels.shape[2]/(labels.shape[2]-2*scale),1),order=1 )
        labels[labels!=0] = 1'''
    return images,labels

def augmentation2(images,labels):
    new_images = images*0
    scale =  int(np.random.random() * 40)
    if np.random.random()>0.5:
        new_images[:,scale:] = images[:,:(images.shape[1]-scale)]
        new_images[:,:scale] = images[:,(images.shape[1]-scale):]
    else:
        new_images[:,:(images.shape[1]-scale)] = images[:,scale:]
        new_images[:,(images.shape[1]-scale):] = images[:,:scale]
    scale =  int(np.random.random() * 40)
    if np.random.random()>0.5:
        new_images[:,:,scale:] = images[:,:(images.shape[1]-scale)]
        new_images[:,:,:scale] = images[:,:,(images.shape[1]-scale):]
    else:
        new_images[:,:,:(images.shape[1]-scale)] = images[:,:,scale:]
        new_images[:,:,(images.shape[1]-scale):] = images[:,:,:scale]
    

def run_training(dataset = 1,sigma = 1,lbsthresh=1,maxIter=2500,selfiter = True,ita = 0.01,selfth = 0.1,selfstart = 0, selfinterval = 1,lbsshiftid = 1):
    with tf.Graph().as_default():
    #with tf.variable_scope('input'):
        ###get data, mask
        #images = load_data()
        #masks = load_data(filename='../data/train_mask32.tfrecords',imsize=32)+0.25
        if dataset ==1:
            im = np.load('data/PhC-C2DH-U373_train_data.npy')
            lbs = np.load('data/PhC-C2DH-U373_train_label.npy')
            #im = zoom(im,(1,512./im.shape[1],512./im.shape[2],1),order=1 )
            #lbs = zoom(lbs,(1,512./lbs.shape[1],512./lbs.shape[2],1),order=1 )
            dname = 'PhC-C2DH-U373'
            test_im = np.load('data/PhC-C2DH-U373_test_data.npy')
            test_lbs = np.load('data/PhC-C2DH-U373_test_label.npy')
        if dataset ==2:
            im = np.load('data/PhC-C2DH-U373_train_data.npy')
            lbs = np.load('data/PhC-C2DH-U373_train_label.npy')
            for i in range(len(lbs)):
                lbs[i,:,:,0] = cv2.GaussianBlur(lbs[i,:,:,0],(sigma,sigma),0)
            lbs[lbs>lbsthresh] = 1
            dname = 'PhC-C2DH-U373'
            test_im = np.load('data/PhC-C2DH-U373_test_data.npy')
            test_lbs = np.load('data/PhC-C2DH-U373_test_label.npy')
        if dataset ==3:
            im = np.load('data/PhC-C2DH-U373_train_data.npy')
            lbs = np.load('data/PhC-C2DH-U373_train_label_random'+str(lbsshiftid)+'.npy')
            dname = 'PhC-C2DH-U373_random'+str(lbsshiftid)
            test_im = np.load('data/PhC-C2DH-U373_test_data.npy')
            test_lbs = np.load('data/PhC-C2DH-U373_test_label.npy')
            #im = im[:len(im//8)]
            #lbs = lbs[:len(im//8)]
        if dataset ==4:
            im = np.load('data/PhC-C2DH-U373_train_data.npy')
            lbs = np.load('data/PhC-C2DH-U373_train_label_random'+str(lbsshiftid)+'.npy')
            #lbs2 = nib.load('data/PhC-C2DH-U373_train_label_center.nii').get_data()
            lbs2 = nib.load('data/PhC-C2DH-U373_train_label_boundary.nii').get_data()
            lbs = lbs2.astype(lbs.dtype).reshape(lbs.shape)
            dname = 'PhC-C2DH-U373_random'+str(lbsshiftid)
            test_im = np.load('data/PhC-C2DH-U373_test_data.npy')
            test_lbs = np.load('data/PhC-C2DH-U373_test_label.npy')



        #lbs_gradiens = np.gradient(lbs,axis=[1,2])[0]
        #lbs_gradiens[lbs_gradiens!=0] = 1
        #lbs_gradiens = lbs_gradiens[:,:,:,0]
        #lbs_gradiens = dilation(lbs_gradiens, cube(3)[:1])
        #lbs_gradiens = np.ones(lbs_gradiens.shape)
        lbs_gradiens = np.array(lbs*2)
        lbs_gradiens = lbs_gradiens[:,:,:,0]
        #print(im.shape)
        #print(lbs.shape)

        print('lbs_gradiens',lbs_gradiens.shape)
        '''for i in range(len(lbs)):
            print('lbs[i,:,:,0]',lbs[i,:,:,0].shape)
            print('lbs[i,:,:,0]',np.gradient(lbs[i,:,:,0])[0].shape)
            print('lbs_gradiens[i]',lbs_gradiens[i].shape)
            lbs_gradiens[i] =  np.gradient(lbs[i,:,:,0])[0]
            lbs_gradiens[i][lbs_gradiens[i]!=0] = 1
            lbs_gradiens[i] = dilation(lbs_gradiens[i], cube(3)[:1])'''


        images = tf.placeholder(tf.float32, [batchsize, 520,696,1])
        masks = tf.placeholder(tf.float32, [batchsize, 520,696,1])
        wmasks = tf.placeholder(tf.float32, [batchsize, 520,696])
        imasks = tf.cast(masks, tf.int32) 
       

        #Graph
        weights,beights,upop = createweights()
        h5,h5_softmax = ConvNetwork(images,weights,beights,upop)
        print('h5',h5)

        masks2 = tf.concat([1-masks,masks],3)
        loss = cross_entropy_prob_loss(h5,masks2,wmasks)

        count_constrain = tf.nn.relu(40+tf.reduce_sum(h5_softmax[:,:,:,1])-tf.reduce_sum(masks))
        #count_constrain = tf.nn.relu(0.0003+2*(tf.reduce_sum(h5_softmax[:,:,:,1])-tf.reduce_sum(masks))/(tf.reduce_sum(h5_softmax[:,:,:,1])+tf.reduce_sum(masks)))

        #loss = cross_entropy_loss(h5,imasks,wmasks)
        #mloss = my_cross_entropy_loss(h5,masks)
        pre_mask = tf.expand_dims(h5_softmax[:,:,:,1],-1) 
        pre_mask = pre_mask-tf.reduce_min(pre_mask,axis=[1,2,3],keep_dims=True)
        pre_mask = pre_mask/tf.reduce_max(pre_mask,axis=[1,2,3],keep_dims=True)
        dcloss = dice_coieffience_loss(pre_mask,masks)

        regularizer = 0
        for w in weights:
            regularizer += tf.nn.l2_loss(w)

        #Train
        lr=0.0001
        theta = 0
        #train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss+theta*regularizer)
        train_step = tf.train.MomentumOptimizer(lr,0.9).minimize(loss+theta*regularizer + ita * count_constrain)
        #Run
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

        #saver = tf.train.Saver({"my_v2": v2})
        #[<tf.Variable 'Variable:0' shape=(8, 8, 1, 48) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(48,) dtype=float32_ref>, <tf.Variable 'Variable_2:0' shape=(2, 2, 48, 128) dtype=float32_ref>, <tf.Variable 'Variable_3:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'Variable_4:0' shape=(0,) dtype=float32_ref>, <tf.Variable 'Variable_2_1:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Variable4:0' shape=(8,) dtype=float32_ref>]
        #
        
        
        saver = tf.train.Saver({'Variable':weights[0],
            'Variable_1':beights[0],
            'Variable_2':weights[1],
            'Variable_3':beights[1],
            'Variable_4':weights[2],
            'Variable_5':beights[2],
            'Variable_6':weights[3],
            'Variable_7':beights[3],
            'Variable_8':weights[4],
            'Variable_9':beights[4],
            'Variable_10':weights[5],
            'Variable_11':beights[5],
            'Variable_12':weights[6],
            'Variable_13':beights[6],
            'Variable_14':weights[7],
            'Variable_15':beights[7],
            'Variable_16':weights[8],
            'Variable_17':beights[8],
            'Variable_18':weights[9],
            'Variable_19':beights[9],
            'Variable_20':weights[10],
            'Variable_21':beights[10],
            'Variable_22':weights[11],
            'Variable_23':beights[11],
            'Variable_24':weights[12],
            'Variable_25':beights[12],
            'Variable_26':weights[13],
            'Variable_27':beights[13],
            'Variable_28':weights[14],
            'Variable_29':beights[14],
            'Variable_30':weights[15],
            'Variable_31':beights[15],
            'Variable_32':weights[16],
            'Variable_33':beights[16],
            'Variable_34':weights[17],
            'Variable_35':beights[17]
                })


        saver3 = tf.train.Saver(max_to_keep=1000)
        sess =   tf.Session(config=config)  
        print('run initial')
        sess.run(init_op)
        print('coordinator')
        #coord = tf.train.Coordinator()
        #print('threads')
        #threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        #print('threads end')

        '''epoch = 0
        rstdir = '../experiment2/C2MT_ep'+str(epoch)+'/model.ckpt'
        saver.restore(sess,rstdir)
        print('Model retored from',rstdir)'''

        
        #epoch = 260
        #rstdir = '../experiment2/C5_130_loadC20_lr0.0001_theta0_DC110.5_ep'+str(epoch)+'/model.ckpt'
        
        '''epoch = 2400
        rstdir = '../experiment/C5_130_loadC58360_lr1e-05_theta1_DC0.605_ep'+str(epoch)+'/model.ckpt'
        saver.restore(sess,rstdir)
        print('Model retored from',rstdir)'''
        
        epoch = 340
        rstdir = './experiment_unet/unet23_lr0.0001_theta0_doneatep340/model.ckpt'
        epoch = 6600
        rstdir = './experiment_unet/unet23_lr0.0001_epoch340_theta0_DC0.892_ep6600/model.ckpt'



        #datawet1
        epoch = 200
        rstdir = './experiment_unet/unet23_lr1e-05_epoch6600_theta0_DC0.366_ep200/model.ckpt'
        epoch = 400
        rstdir = './experiment_unet/unet23_lr1e-05_epoch200_theta0_DC0.613_ep400/model.ckpt'
        epoch = 520
        rstdir = './experiment_unet/unet23_lr1e-05_epoch400_theta0_DC0.788_ep520/model.ckpt'


        #datawet others
        epoch = 2420
        rstdir = './experiment_unet/PhC-C2DH-U373_downsize128_unet23_lr1e-05_epoch520_theta0_DC0.669_ep2420/model.ckpt'

        rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_epoch2420_theta0_DC0.190_ep640/model.ckpt'
        rstdir = './experiment_unet/unet17_lr0.0001_theta0_sigma5_lbsthresh1.0_doneatep111/model.ckpt'

        rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_theta0_sigma11_lbsthresh1.0_DC0.028_v0.577_ep20/model.ckpt'

        rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_theta0_sigma11_lbsthresh1.0_DC0.044_v0.548_ep28/model.ckpt'

        #rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_theta0_sigma11_lbsthresh1.0_DC0.026_v0.487_ep20/model.ckpt'

        rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_theta0_sigma15_lbsthresh0.1_DC0.197_v0.478_ep340/model.ckpt'  # 330

        rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_theta0_sigma15_lbsthresh1.0_DC0.062_v0.559_ep69/model.ckpt'  # 330

        rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_epoch2420_theta0_DC0.213_v0.388_ep100/model.ckpt'  # 389
        kk=100
        '''#rstdir = './experiment_unet/PhC-C2DH-U373_selfiter_unet17_lr0.0001_theta0_sigma11_lbsthresh1.0_DC0.650_v0.531_ep540/model.ckpt'  #

        #rstdir = './experiment_unet/PhC-C2DH-U373_selfiter_unet17_lr0.0001_theta0_sigma1_lbsthresh1.0_DC0.691_v0.603_ep20/model.ckpt'  #

        #rstdir = './experiment_unet/PhC-C2DH-U373_selfiter_unet17_lr0.0001_theta0_sigma1_lbsthresh1.0_DC0.767_v0.651_ep40/model.ckpt'  #

        #rstdir = './experiment_unet/PhC-C2DH-U373_selfiter_unet17_lr0.0001_theta0_sigma1_lbsthresh1.0_DC0.860_v0.688_ep160/model.ckpt'  #

        #rstdir = './experiment_unet/PhC-C2DH-U373_selfiter_unet17_lr0.0001_theta0_sigma1_lbsthresh1.0_DC0.769_v0.657_ep40/model.ckpt'  #

        #rstdir = './experiment_unet/PhC-C2DH-U373_selfiter20_1_unet17_lr0.0001_theta0_ita0.1_sigma1_lbsthresh1.0_DC0.864_v0.516_ep840/model.ckpt'  #
        rstdir = './experiment_unet/PhC-C2DH-U373_selfjump80_20_unet17_lr0.0001_theta0_ita0.1_sigma11_lbsthresh1.0_DC0.865_v0.743_ep2480/model.ckpt'  #

        #rstdir = './experiment_unet/PhC-C2DH-U373_unet17_lr0.0001_epoch2420_theta0_DC0.213_v0.388_ep100/model.ckpt'  # 389
        kk = 0
        #rstdir = './experiment_shift/PhC-C2DH-U373stfc1_0_unet17_lr0.0001_theta0_ita0.001_sigma11_lbsthresh1.0_DC0.113_v0.353_ep360/model.ckpt'  # 389
        kk = 0
        rstdir = './experiment_shift/PhC-C2DH-U373_random1stfc1_0_unet17_lr0.0001_theta0_ita1e-05_sigma11_lbsthresh1.0_DC0.052_v0.409_ep400/model.ckpt'  # 389
        kk = 400
        rstdir = './experiment_shift/PhC-C2DH-U373_random1stfc1_0_unet17_lr0.0001_theta0_ita1e-05_sigma11_lbsthresh1.0_DC0.059_v0.500_ep700/model.ckpt'  # 389
        kk = 0
        #rstdir = './experiment_shift/PhC-C2DH-U373_random2stfc1_0_unet17_lr0.0001_theta0_ita0.0_sigma11_lbsthresh1.0_DC0.057_v0.432_ep700/model.ckpt'  # 389
        #kk = 700
        rstdir = './experiment_shift/PhC-C2DH-U373_random1stfc1_0_unet17_lr0.0001_theta0_ita0.0001_sigma11_lbsthresh1.0_DC0.861_v0.698_ep2480/model.ckpt'  # 389
        kk = 2480
        rstdir = './experiment_shift/PhC-C2DH-U373_random4stfc1_0_unet17_lr0.0001_theta0_ita0.0001_sigma11_lbsthresh1.0_DC0.837_v0.719_ep2260/model.ckpt'  # 389
        kk = 2480
        rstdir = './experiment_shift/PhC-C2DH-U373_random2stfc1_0_unet17_lr0.0001_theta0_ita0.0001_sigma11_lbsthresh1.0_DC0.860_v0.714_ep2480/model.ckpt'  # 389
        kk = 2480'''
        rstdir = './experiment_shift/PhC-C2DH-U373_random4stfc1_0_unet17_lr0.0001_theta0_ita0.0001_sigma11_lbsthresh1.0_DC0.835_v0.724_ep3740/model.ckpt'  # 389
        kk = 3740
        saver3.restore(sess,rstdir)
        print('Model retored from',rstdir)
        #kk = 360#2480
        #imgss = np.zeros([200,256,256,1])
        #mskss = np.zeros([200,32,32,1])
        kk=0
        loss_all = np.zeros(3)
        print_freqence = 20
        val_freqence = print_freqence
        save_freqence = 20#print_freqence
        #for kk in range(1200):
        pointer = 0
        shuffleidx = np.arange(len(im))
        np.random.shuffle(shuffleidx)
        bestv = -0.2
        print('lbssssssss',lbs.sum(),lbs.shape)
        jump1 = 1
        jump2 = 0
        if selfiter:
            #dname += '_selfiter'+str(selfstart)+'_'+str(selfinterval)
            dname += 'stfc'+str(jump1)+'_'+str(jump2)
        try:
            while kk<maxIter:

                point_end = pointer+batchsize
                if point_end>len(im):
                    batch_xs = np.concatenate( [im[shuffleidx][pointer:],im[shuffleidx][:batchsize-(len(im)-pointer)]], axis=0)
                    batch_ys = np.concatenate( [lbs[shuffleidx][pointer:],lbs[shuffleidx][:batchsize-(len(im)-pointer)]], axis=0)
                    pointer=batchsize-(len(im)-pointer)
                else:
                    batch_xs = im[shuffleidx][pointer:point_end]
                    batch_ys = lbs[shuffleidx][pointer:point_end]
                    pointer+=batchsize
                '''ii,mm = sess.run([images,masks],feed_dict={images: batch_xs, masks: batch_ys})
                print('iimm',ii.shape,mm.shape)
                plt.imshow(ii[0,:,:,0])
                plt.show()
                plt.imshow(mm[0,:,:,0])
                plt.show()
                
                break'''
                batch_xs,batch_ys = augmentation(batch_xs,batch_ys)
                #batch_ws = np.array((batch_ys+0.2)*5)
                '''batch_ws = np.array((1.5-batch_ys))
                batch_ws = np.ones(batch_ys.shape)
                batch_ws = batch_ws[:,:,:,0]

                batch_ws = np.array(batch_ys)
                batch_ws = gaussian(batch_ys[:,:,:,0].astype(np.float64),sigma=(0,2,2))
                batch_ws[batch_ws>=1] = 0
                batch_ws[batch_ws>0] = 1
                #batch_ws = batch_ws+0.1
                batch_ys = batch_ys - np.expand_dims(batch_ws,axis=-1)
                batch_ys[batch_ys<0] = 0
                #print('batch_ys',batch_ys.min(),batch_ys.max())
                #print('batch_ws',batch_ws.min(),batch_ws.max())
                
                
                #print('lbssssssss',lbs.sum(),lbs.shape)
                #print('lbssssssss',batch_ys.sum(),batch_ys.shape)
                #break'''


                batch_ws = np.ones(batch_ys.shape)
                batch_ws = batch_ws[:,:,:,0]
                batch_ws = 10*batch_ys+1
                batch_ws = batch_ws[:,:,:,0]
                if selfiter and kk > selfstart:
                    #if kk%selfinterval ==0:
                    #if kk%120  > 700:
                    batch_ws = np.ones(batch_ys.shape)
                    batch_ws = batch_ws[:,:,:,0]
                    [iiss] = sess.run([h5_softmax[:,:,:,1:]],feed_dict={images: batch_xs})
                    if kk%(jump1+jump2)  < jump1:
                        print('YYYy')
                        batch_ys = np.maximum(batch_ys,iiss)
                        [countc] = sess.run([count_constrain],feed_dict={images: batch_xs, masks: batch_ys})
                        print('selfiter',np.maximum(batch_ys,iiss).sum()-iiss.sum(),2*(np.maximum(batch_ys,iiss).sum()-iiss.sum())/(np.maximum(batch_ys,iiss).sum()+iiss.sum()),countc)

                    else:
                        batch_ys = iiss                    #batch_ys = iiss
                    #batch_ys = (iiss>selfth).astype(np.float32)

                    #batch_ys2 = np.copy(batch_ys)
                    #for i in range(len(batch_ys2)):
                    #    batch_ys2[i, :, :, 0] = cv2.GaussianBlur(batch_ys2[i, :, :, 0], (101, 101), 0)
                    #iiss[batch_ys2==0.] = 0
                    #batch_ys = np.maximum(batch_ys,(iiss>selfth).astype(np.float32))

                #print(im.shape,lbs.shape,batch_xs.shape,batch_ys.shape,batch_ws.shape)
                #[_,rr,dd,ll,iiss] = sess.run([train_step,regularizer,dcloss,loss,h5_softmax[:,:,:,1]],feed_dict={images: batch_xs, masks: batch_ys, wmasks:batch_ws})
                print('loss',loss)
                [ rr, dd, ll,cc] = sess.run([ regularizer, dcloss, loss,count_constrain],
                                                 feed_dict={images: batch_xs, masks: batch_ys, wmasks: batch_ws})

                loss_all += [rr,dd,ll]
                print('loss', ll)
                print('count_constrain', cc)



                if kk%val_freqence==0:
                    bbs = []
                    areas = []
                    vpointer = 0
                    flag = 0
                    while True:
                        vpoint_end = vpointer+batchsize
                        if vpoint_end>len(test_im):
                            vbatch_xs = np.concatenate( [test_im[vpointer:],test_im[:batchsize-(len(test_im)-vpointer)]], axis=0)
                            vbatch_ys = np.concatenate( [test_lbs[vpointer:],test_lbs[:batchsize-(len(test_im)-vpointer)]], axis=0)
                            vpointer=batchsize-(len(test_im)-vpointer)
                            flag=1
                        else:
                            vbatch_xs = test_im[vpointer:vpoint_end]
                            vbatch_ys = test_lbs[vpointer:vpoint_end]
                            vpointer+=batchsize

                        [dd,iiss] = sess.run([dcloss, h5_softmax[:, :, :, 1]],feed_dict={images: vbatch_xs, masks: vbatch_ys})

                        bbs += [dd]
                        areas += [iiss.sum()]
                        if flag==1:
                            break
                    print(kk,time.strftime("%H:%M:%S"), 'val dcloss',np.mean(bbs),np.mean(areas)/batchsize/30000)





                if kk % save_freqence == 0:# and np.mean(bbs) < bestv:
                    bestv = min(bestv,np.mean(bbs))
                    if kk == 0:
                        dirname = './experiment_shift2/' + dname + '_unet17_lr' + str(lr)  + '_theta' + str(theta)+'_ita'+str(ita)+ '_sigma' + str(sigma) + '_lbsthresh' + str(lbsthresh)+ '_DC' + str(loss_all[-2])[1:6] + '_v'+  str(np.mean(bbs))[1:6]+ '_ep' + str(kk)
                    else:
                        dirname = './experiment_shift2/' + dname + '_unet17_lr' + str(lr) + '_theta' + str(theta)+'_ita'+str(ita)+ '_sigma' + str(sigma)+ '_lbsthresh' + str(lbsthresh) + '_DC' + str(loss_all[-2] / print_freqence)[
                                                                     1:6]  + '_v'+  str(np.mean(bbs))[1:6] + '_ep' + str(kk)

                    while os.path.exists(dirname):
                        dirname = dirname + '_'
                    os.makedirs(dirname)
                    save_path = saver3.save(sess, dirname + "/model.ckpt")
                    print("Model saved in file: %s" % save_path)
                    new_imagek = nib.Nifti1Image(vbatch_xs.astype(np.float64), affine=np.eye(4))
                    nib.save(new_imagek, dirname + '/image_save.nii')
                    new_imagek = nib.Nifti1Image(vbatch_ys.astype(np.float64), affine=np.eye(4))
                    nib.save(new_imagek, dirname + '/lables_save.nii')
                    new_imagek = nib.Nifti1Image(iiss.astype(np.float64), affine=np.eye(4))
                    nib.save(new_imagek, dirname + '/softnax_save.nii')
                    #new_imagek = nib.Nifti1Image(batch_ws.astype(np.float64), affine=np.eye(4))
                    #nib.save(new_imagek, dirname + '/mask_save.nii')
                if kk%print_freqence==0:
                    if kk==0:
                        print(kk,time.strftime("%H:%M:%S"), 'regularizer,dcloss,loss',loss_all,'lr',lr,'theta',theta)
                    else:
                        print(kk,time.strftime("%H:%M:%S"), 'regularizer,dcloss,loss',loss_all/print_freqence,'lr',lr,'theta',theta)
                    loss_all[:] = 0
                kk += 1

        except (tf.errors.OutOfRangeError,KeyboardInterrupt) as e:
            print('Done training for %d steps.' % (kk))
            dirname ='./experiment_shift2/unet17_lr'+str(lr)+'_theta'+str(theta)+'_ita'+str(ita)+ '_sigma' + str(sigma) + '_lbsthresh' + str(lbsthresh)+'_doneatep'+str(kk)
            while os.path.exists(dirname):
                dirname = dirname+'_'
            os.makedirs(dirname)
            save_path = saver3.save(sess, dirname+"/model.ckpt")

        #finally:
        #    coord.request_stop()

        #coord.join(threads)
        sess.close()


if __name__ == '__main__':
    datasaetid = 4
    sigma = 11
    lbsthresh = 1.
    maxIter = 2500
    ita = 0.1
    selfiter = True
    selfstart = 100
    selfinterval = 1
    lbsshiftid = 1
    if '-d' in  sys.argv:
        datasaetid = int(sys.argv[sys.argv.index('-d')+1])
        if datasaetid == 1:
            sigma = 1
        if datasaetid == 2:
            if '-sigma' in sys.argv:
                sigma = int(sys.argv[sys.argv.index('-sigma')+1])
            if '-thlbs' in sys.argv:
                lbsthresh = float(sys.argv[sys.argv.index('-thlbs')+1])
    if '-maxIter' in  sys.argv:
        maxIter = int(sys.argv[sys.argv.index('-maxIter')+1])
    if '-ita' in  sys.argv:
        ita = float(sys.argv[sys.argv.index('-ita')+1])
    if '-selfiter' in sys.argv:
        selfiter = bool(int(sys.argv[sys.argv.index('-selfiter')+1]))
    if selfiter:
        if '-selfstart' in sys.argv:
            selfstart = int(sys.argv[sys.argv.index('-selfstart')+1])
        if '-selfinterval' in sys.argv:
            selfinterval = int(sys.argv[sys.argv.index('-selfinterval')+1])


    if '-lbsshiftid' in  sys.argv:
        lbsshiftid = int(sys.argv[sys.argv.index('-lbsshiftid')+1])
    print('ita',ita)
    run_training(dataset = datasaetid,sigma = sigma,lbsthresh=lbsthresh,maxIter=maxIter,selfiter=True,selfstart = selfstart,selfinterval = selfinterval,ita=ita,lbsshiftid=lbsshiftid)



