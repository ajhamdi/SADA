import numpy as np
import os
import math
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import scipy.misc
import time
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from distutils.dir_util import copy_tree
import shutil
from scipy.sparse import lil_matrix
import scipy 
import matplotlib.pyplot as plt
from scipy import signal
import hdf5storage
import cv2
from utils import *
from models import *
from scipy.linalg import circulant
from scipy.linalg import dft
import scipy.io as sio
import tensorflow as tf
from skimage.util.shape import view_as_blocks
slim = tf.contrib.slim


n= 129    # the image shape
m = 5  # the kernel shape
RANDOM_NO = np.abs(np.random.random_integers(1000))
LOAD_MAT = True
skip = int(np.abs(n-m)/2)
path = "D:\\mywork\\sublime\\GAN2\\data\\celebB"
images = read_images_to_np(path,n,n,extension="all",allowmax=True,maxnbr=1000,d_type=np.float32,mode="RGB")
gray_images  = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
plt.figure()
plt.imshow(gray_images[RANDOM_NO],cmap='gray')
plt.title("Filtering by matrix multiplicatin")
plt.show(block=False)
print("finished loading the images\n")
if LOAD_MAT:
  mat = hdf5storage.loadmat('filtering.mat')
  k_padded = mat["k_padded"]
  A= mat["A_unfolded"]
  K = mat["kernel"]
  print("the Kernel\n",K,"\n\n")
  # print("the Kernel padded \n",k_padded,"\n\n")
  # k_padded = sio.loadmat("filtering.mat")["kernel"]
  # A = sio.loadmat("filtering.mat")["A_unfolded"]
  X = gray_images[RANDOM_NO]
  x = X.reshape(-1,1)
 

train_log_dir = os.path.join(path,'logs')
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

with tf.Session() as sess:
  in_images = tf.placeholder(tf.float32, (None,) + n)
  with tf.Graph().as_default():
    output = DiscriminatorCNN_encoder(in_images,n, n**2, 5,  data_format=tf.float32)
    predictoins = A[0,:].reshape(-1,1).astype(np.float32)
    loss = tf.reduce_mean(tf.abs(AE_G - G))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    tf.summary.scalar('losses/total_loss', loss)
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)
    slim.learning.train(train_tensor, train_log_dir)

