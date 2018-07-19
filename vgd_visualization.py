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
# from tqdm import trange
from itertools import chain
# from collections import deque
from distutils.dir_util import copy_tree
import shutil
from scipy.sparse import lil_matrix
import scipy 
import matplotlib.pyplot as plt
from scipy import signal
# import cv2
# from utils import *
# from models import *
# from scipy.linalg import circulant
# from scipy.linalg import dft
import scipy.io as sio
import tensorflow as tf
# from skimage.util.shape import view_as_blocks
slim = tf.contrib.slim


def objective_function(x,output_size):
  hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
  output = slim.fully_connected(hidden, output_size, scope='objective/fc_2')
  output = slim.fully_connected(output, output_size, scope='objective/fc_3')
  output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_4')

  return output

def black_box(x,output_size):
  hidden = slim.fully_connected(x, 50, scope='blackbox/fc_1')
  output = slim.fully_connected(hidden, output_size,activation_fn=None, scope='blackbox/fc_2')
  return output

def virtual_gradient(x,output_size):
  hidden = slim.fully_connected(x, 50, scope='virtualgradient/fc_1')
  output = slim.fully_connected(hidden, output_size,activation_fn=None, scope='virtualgradient/fc_2')
  return output



n= 10    # the input shape
m = 100  # the output shape
N = 100 # the number of data we have
DATA_LIMIT = 1000
random_no = np.abs(np.random.randint(DATA_LIMIT))
META_LEARN_FLAG = True
learning_rate=0.002
meta_learning_rate = 0.002
init_variance = 0.01
perturbation = 0.001
batch_size = 1
META_STEPS = 15
STEPS_NUMBER = 30
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
X = np.random.random(size=(N,n)).astype(np.float32)
Y = np.random.random(size=(N,m)).astype(np.float32)
data_path = "D:\\mywork\\sublime\\GAN2\\data\\celebB"
base_path = "D:\\mywork\\sublime\\vgd"

# images = read_images_to_np(path,n,n,extension="all",allowmax=True,maxnbr=1000,d_type=np.float32,mode="RGB")
# gray_images  = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
if META_LEARN_FLAG:
  train_log_dir = os.path.join(base_path,'logs','VGD')
else :
  train_log_dir = os.path.join(base_path,'logs','SGD')
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default() as g:
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, init_variance),
                      weights_regularizer=slim.l2_regularizer(0.00001)):
    x = tf.placeholder(tf.float32, (None,n))
    labels = tf.placeholder(tf.float32, (None, m))

    f_x = objective_function(x,n)
    g_bb = black_box(f_x,m)
    # loss = slim.losses.sum_of_squares(g_bb, labels)
    loss = tf.reduce_mean(tf.nn.l2_loss(g_bb - labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    loss_summary= tf.summary.scalar('losses/main_loss', loss)
    # total_loss = slim.losses.get_total_loss()
    optimization_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='objective')
    # optimization_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='blackbox')
    for i in optimization_variables:
      print("\n\n the optimization variables ... ",i)
    # train_tensor = slim.learning.create_train_op(loss, optimizer,variables_to_train=optimization_variables)
    train_tensor = slim.learning.create_train_op(loss, optimizer)



with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  writer = tf.summary.FileWriter(train_log_dir, sess.graph)
  tf.global_variables_initializer().run()
  for step in range(STEPS_NUMBER):
    X = np.random.random(size=(N,n)).astype(np.float32)
    if META_LEARN_FLAG:
      tried_directions = []
      tried_fittness = []
      current_objectives,current_fitness = sess.run([optimization_variables,loss],feed_dict={x:X,labels:Y})
      # print("\n\n the current values .. ",current_objectives[0].shape)
      for meta in range(META_STEPS) :
        
        # perturbing the variables with random directions
        for idx , value in enumerate(current_objectives):
          some_random = np.random.uniform(-perturbation, perturbation, value.shape)
          new_objective = value + some_random
          optimization_variables[idx].assign(new_objective).op.run()
          tried_directions.append(- some_random)

        # X = np.random.random(size=(N,n)).astype(np.float32)
        new_fittness = sess.run(loss,feed_dict={x:X,labels:Y})
        print("     the loss at trial %3d is:  %3.4f" %(meta,new_fittness))
        # new_objective = 
      tried_fittness.append(new_fittness)
      _,new_summary = sess.run([train_tensor,loss_summary],feed_dict={x:X,labels:Y})

    else:
      _,new_summary,current_fitness = sess.run([train_tensor,loss_summary,loss],feed_dict={x:X,labels:Y})
    chosen_direction =  tried_directions[tried_fittness.index(min(tried_fittness))]
    writer.add_summary(new_summary,step)
    print("step number : %3d Loss: %3.4f \n" %(step,current_fitness))
  # slim.learning.train(train_tensor, train_log_dir,number_of_steps=STEPS_NUMBER,
  #   save_summaries_secs=60,save_interval_secs=600)
print("finished training ....")

