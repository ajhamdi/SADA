import numpy as np
import os
import math
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import scipy.misc
import time
import random
import _pickle as cPickle
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
import cv2
import rbfopt  # black_box_optimization library
from utils import *
from models import *
from scipy.linalg import circulant, norm
from scipy.linalg import dft
import scipy.io as sio
import tensorflow as tf
from skimage.util.shape import view_as_blocks
slim = tf.contrib.slim







def objective_function(x,output_size):
  hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
  output = slim.fully_connected(hidden, 2*output_size, scope='objective/fc_2')
  output = slim.fully_connected(output, 3*output_size, scope='objective/fc_3')
  output = slim.fully_connected(output, 3*output_size, scope='objective/fc_4')
  output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_5')

  return output

def dicrminator_cnn(x,conv_input_size, output_size, repeat_num ,reuse=False):
  with tf.variable_scope("oracle") as scope:
    if reuse:
      scope.reuse_variables()
    # Encoder
    x = slim.conv2d(x, conv_input_size, 3, 1, activation_fn=tf.nn.elu)

    prev_channel_num = conv_input_size
    for idx in range(repeat_num):
      channel_num = conv_input_size * (idx + 1)
      x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
      x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
      if idx < repeat_num - 1:
        x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu)
        #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

    x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
    z = x = slim.fully_connected(x, output_size, activation_fn=None)
    z_prob = tf.nn.sigmoid(z)
    return z , z_prob


def dicrminator_ann(x,output_size,reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()
    hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
    output = slim.fully_connected(hidden, 6*output_size, scope='objective/fc_2')
    output = slim.fully_connected(output, 3*output_size, scope='objective/fc_3')
    output = slim.fully_connected(output, 3*output_size, scope='objective/fc_4')
    output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_5')
  return output
def generator_ann(x,output_size,min_bound=-1,max_bound=1):
  range_required = np.absolute(max_bound - min_bound).astype(np.float64)
  with tf.variable_scope("generator") as scope:
    hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
    output = slim.fully_connected(hidden, 2*output_size, scope='objective/fc_2')
    output = slim.fully_connected(output, 3*output_size, scope='objective/fc_3')
    output = slim.fully_connected(output, 3*output_size, scope='objective/fc_4')
    output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_5')
    contrained_output =   range_required * tf.nn.sigmoid(output) + np.array(min_bound* tf.ones_like(output)).astype(mp.float64)
  return contrained_output



def black_box(input_vector,output_size=256,global_step=0,frames_path=None,params=None):
  b = Blender('init.py','trils.blend')
  b.basic_experiment(obj_name="Cube", vec=input_vector.tolist())
  b.save_image(output_size,output_size,path=frames_path,name=str(global_step))
  # b.save_file()
  b.execute()
  image = cv2.imread(os.path.join(frames_path,str(global_step)+".jpg"))
  image = forward_transform(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32))
  # images = read_images_to_np(frames_path,output_size,output_size,extension="all",allowmax=True,maxnbr=1000,d_type=np.float32,mode="RGB")
  return image





def virtual_gradient(x,output_size):
  hidden = slim.fully_connected(x, 50, scope='virtualgradient/fc_1')
  output = slim.fully_connected(hidden, output_size,activation_fn=None, scope='virtualgradient/fc_2')
  return output


 
class BlackBoxOptimizer():

  def __init__(self,exp_type="Random",exp_no=0,base_path=None):
    self.exp_type = exp_type
    self.exp_no = exp_no
    self.frames_path = os.path.join(base_path,"frames")
    self.generated_path = os.path.join(base_path,"generated")
    self.frames_log_dir = os.path.join(self.frames_path,self.exp_type+"_%d"%(self.exp_no))
    self.train_log_dir = os.path.join(base_path,'logs',self.exp_type+"_%d"%(self.exp_no))
    if not tf.gfile.Exists(self.train_log_dir):
      tf.gfile.MakeDirs(self.train_log_dir)
    if not tf.gfile.Exists(self.frames_log_dir):
      tf.gfile.MakeDirs(self.frames_log_dir)

    self.n= 6    # the input to blck_box shape
    self.m = 20  # the generator input shape 
    self.N = 100 # the number of data we have
    # self.DATA_LIMIT = 1000
    # np.random.seed(0)
    # self.random_no = np.abs(np.random.randint(self.DATA_LIMIT))
    self.META_LEARN_FLAG = False
    self.ALLOW_LOGGING = True
    self.bb_log_frq = 4
    self.gen_log_frq = 2
    self.learning_rate=0.0006 # originally 0.001
    self.beta1 = 0.5
    self.reg_hyp_param = 10.0
    self.solution_learning_rate = 0.000000005  # multiplied by the approx gradient
    self.init_variance = 0.05  
    self.stochastic_perturbation = 0.7   # perturbation to find local min using random search 
    self.gradient_perturbation = 0.003  # perturbation of gradient approxmation
    self.OUT_SIZE = 128
    self.loss_mormalization = 0.5 *(2 * self.OUT_SIZE **2)**2 # to normalize the L2 pixel loss 
    self.batch_size = 16
    self.generate_distribution_size = 1500
    self.generation_bound = 0.01
    # print("THe VALUE....  " , self.solution_learning_rate  / self.loss_mormalization )
    self.META_STEPS = 30
    self.epochs = 11
    self.STEPS_NUMBER = 1000
    self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # self.X = np.random.random(size=(self.N,self.n)).astype(np.float32)
    # self.Y = np.random.random(size=(self.N,self.m)).astype(np.float32)
    self.fake_target = cv2.imread(os.path.join(self.frames_path,"target"+".jpg"))
    self.fake_target = forward_transform(cv2.cvtColor(self.fake_target,cv2.COLOR_BGR2RGB).astype(np.float32))



  def train(self):
    if self.exp_type == "Random":
      self.train_random()
    elif self.exp_type == "Gradprox":
      self.train_gradprox()
    elif self.exp_type == "Lsearch":
      self.train_lsearch()
    elif self.exp_type == "RBFopt":
      self.train_rbfopt()
    print("finished training ....")
    self.visualize_optimum_solution()
    print("the total time spent is %3.2f minutes " %((time.time()-self.start_time)/60.0))
  # print("the range of values ", np.max(target),"---",np.min(target))
  # raise Exception("STOOOOP")


  # for ii in range(5):
    # my_random_vector = [np.random.random(),np.random.random(),np.random.random(),
    #  np.random.randint(7),np.random.randint(7),np.random.randint(-180,180)]
    # X = np.random.uniform(-1,1,6).tolist()
    # y = black_box(x,output_size=OUT_SIZE,global_step=ii,frames_path=frames_path)
    # plt.figure()
    # plt.imshow(y)
    # plt.title("the box %d" %(ii))
    # plt.show()


  # images = read_images_to_np(path,n,n,extension="all",allowmax=True,maxnbr=1000,d_type=np.float32,mode="RGB")
  # gray_images  = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

  def train_random(self,random_type="uniform"):
    with tf.Graph().as_default() as self.g:
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, self.init_variance),
                          weights_regularizer=slim.l2_regularizer(0.00001)):
        self.x = tf.placeholder(tf.float32, (self.n,))
        self.labels = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.y = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.loss = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(self.y - self.labels)))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.loss_summary= tf.summary.scalar('losses/main_loss', self.loss)


    self.real_targets = read_images_to_np(path=os.path.join(self.frames_path,"real"),h=self.OUT_SIZE,w=self.OUT_SIZE,d_type=np.float32,mode="RGB")
    self.real_targets = [forward_transform(x) for x in self.real_targets ]
    self.target_no = -1

    with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
      self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
      tf.global_variables_initializer().run()
      self.all_inputs = []
      self.all_losses = []
      self.start_time = time.time()
      scipy.misc.imsave(os.path.join(self.frames_path,"real_target.jpg"), inverse_transform(self.real_targets[self.target_no]))
      if random_type is "lhs":
        self.all_randoms = random_lhs(-1,1,self.n,self.STEPS_NUMBER)
      for step in range(self.STEPS_NUMBER):
        if random_type is "uniform":
          self.X = np.random.uniform(-1,1,self.n)
        elif random_type is "lhs":
          self.X = np.copy(self.all_randoms[step])
        self.Y = black_box(self.X,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path)
        # self.current_fitness,self.new_summary = self.sess.run([self.loss,self.loss_summary],feed_dict={self.x:self.X,self.labels:self.fake_target,self.y:self.Y})
        self.current_fitness,self.new_summary = self.sess.run([self.loss,self.loss_summary],feed_dict={self.x:self.X,self.labels:self.real_targets[self.target_no],self.y:self.Y})
        self.writer.add_summary(self.new_summary,step)
        print("step number : %2d Loss: %4.2f \n" %(step,self.current_fitness))
        self.all_inputs.append(self.X)
        self.all_losses.append(self.current_fitness)
        if self.ALLOW_LOGGING and (step % self.bb_log_frq == 0):
          scipy.misc.imsave(os.path.join(self.frames_log_dir,str(step)+".jpg"), inverse_transform(self.Y))
      


  def train_gradprox(self):
    with tf.Graph().as_default() as self.g:
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, self.init_variance),
                          weights_regularizer=slim.l2_regularizer(0.00001)):
        self.x = tf.placeholder(tf.float64, (self.n,))
        self.labels = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.y = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))


        # f_x = objective_function(x,n)
        # g_bb = black_box(f_x,m)
        # loss = slim.losses.sum_of_squares(g_bb, labels)
        self.gradient_norm = tf.Variable(0,dtype=tf.float32, name='gradient_norm',trainable=False)
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.y - self.labels))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.loss_summary= tf.summary.scalar('losses/main_loss', self.loss)
        self.grad_summary= tf.summary.scalar('gradients/main_gradient', self.gradient_norm)
        # total_loss = slim.losses.get_total_loss()
        # optimization_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='objective')
        # optimization_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='blackbox')
        # for i in optimization_variables:
        #   print("\n\n the optimization variables ... ",i)
        # train_tensor = slim.learning.create_train_op(loss, optimizer,variables_to_train=optimization_variables)
        # train_tensor = slim.learning.create_train_op(loss, optimizer)

    with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
      self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
      tf.global_variables_initializer().run()
      self.all_inputs = []
      self.all_losses = []
      self.start_time = time.time()
      self.X = np.random.uniform(-1,1,6).astype(np.float64)
      for step in range(self.STEPS_NUMBER):
        self.Y = black_box(self.X,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path)
        self.current_fitness,self.new_loss_summary = self.sess.run([self.loss,self.loss_summary],feed_dict={self.x:self.X,self.labels:self.fake_target,self.y:self.Y})
        self.approx_gradient = self.approximate_bb_gradient()
        # self.approx_gradient = self.approximate_bb_forgradient()
        self.gradient_norm.assign(norm(self.approx_gradient)).op.run()
        self.new_gradient_summary = self.sess.run(self.grad_summary)
        self.writer.add_summary(self.new_loss_summary,step)
        self.writer.add_summary(self.new_gradient_summary,step)
        print("\n\n\n\n\n\n\n\nstep number : %2d Loss: %4.2f  gradient_norm: %4.2f  \n" %(step,self.current_fitness,norm(self.approx_gradient)))
        self.all_inputs.append(self.X)
        self.all_losses.append(self.current_fitness)
        if self.ALLOW_LOGGING and (step % self.bb_log_frq == 0):
          scipy.misc.imsave(os.path.join(self.frames_log_dir,str(step)+".jpg"), inverse_transform(self.Y))
        # print("\n\n\n\n\n The size of X",self.approx_gradient.shape)
        self.X = self.X - self.solution_learning_rate * (self.approx_gradient) # / self.loss_normalization
        self.X = np.clip(self.X,-1,1)
        # print("the new  size of X",self.X.shape)
        # raise Exception("...")
  def train_rbfopt(self):
    with tf.Graph().as_default() as self.g:
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, self.init_variance),
                          weights_regularizer=slim.l2_regularizer(0.00001)):
        self.x = tf.placeholder(tf.float32, (self.n,))
        self.labels = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.y = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.loss = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(self.y - self.labels)))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.loss_summary= tf.summary.scalar('losses/main_loss', self.loss)

    def obj_funct(x):
      self.X = x
      self.Y = black_box(self.X,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path)
      self.current_fitness,self.new_summary = self.sess.run([self.loss,self.loss_summary],feed_dict={self.x:self.X,self.labels:self.fake_target,self.y:self.Y})
      self.writer.add_summary(self.new_summary,self.global_step)
      print("step number : %2d Loss: %4.2f \n" %(self.global_step,self.current_fitness))
      self.all_inputs.append(self.X)
      self.all_losses.append(self.current_fitness)
      if self.ALLOW_LOGGING and (self.global_step % self.bb_log_frq == 0):
        scipy.misc.imsave(os.path.join(self.frames_log_dir,str(self.global_step)+".jpg"), inverse_transform(self.Y))
      self.global_step += 1
      return np.copy(self.current_fitness)



    with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
      self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
      tf.global_variables_initializer().run()
      self.all_inputs = []
      self.all_losses = []
      self.global_step=0
      self.start_time = time.time()
      bb = rbfopt.RbfoptUserBlackBox(self.n, -np.ones(self.n), np.ones(self.n),
                                     np.array(['R']*self.n), obj_funct)
      settings = rbfopt.RbfoptSettings(max_evaluations=self.STEPS_NUMBER,global_search_method="sampling",algorithm="Gutmann")
      alg = rbfopt.RbfoptAlgorithm(settings, bb)
      val, x, itercount, evalcount, fast_evalcount = alg.optimize()

  def train_lsearch(self):
    with tf.Graph().as_default() as self.g:
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, self.init_variance),
                          weights_regularizer=slim.l2_regularizer(0.00001)):
        self.x = tf.placeholder(tf.float64, (self.n,))
        self.labels = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.y = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))

        self.descent_amount = tf.Variable(0,dtype=tf.float32, name='descent_amount',trainable=False)
        self.loss = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(self.y - self.labels)))
        self.extra_loss = tf.reduce_mean(tf.nn.relu(-tf.ones_like(self.x)-self.x)+ tf.nn.relu(self.x-tf.ones_like(self.x))) ## regulaizer to restrict the input in [-1,1]
        self.total_loss = tf.to_float(self.loss) + self.reg_hyp_param * tf.to_float(self.extra_loss)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        loss_summary= tf.summary.scalar('losses/main_loss', self.loss)
        extra_loss_summary= tf.summary.scalar('losses/extra_loss', self.extra_loss)
        total_loss_summary= tf.summary.scalar('losses/total_loss', self.total_loss)
        descent_summary= tf.summary.scalar('descent/main_descent', self.descent_amount)
        self.total_summary = tf.merge_summary([loss_summary,extra_loss_summary,total_loss_summary,descent_summary])
    with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
      self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
      tf.global_variables_initializer().run()
      self.all_inputs = []
      self.all_losses = []
      self.all_directions = []
      self.start_time = time.time()
      self.X = np.random.uniform(-1,1,6).astype(np.float64)
      for step in range(self.STEPS_NUMBER):
        self.Y = black_box(self.X,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path)
        self.current_fitness,summary_string = self.sess.run([self.total_loss,self.total_summary],
          feed_dict={self.x:self.X,self.labels:self.fake_target,self.y:self.Y})
        chosen_point, descent_amount, chosen_direction  = self.local_random_search()
        self.descent_amount.assign(descent_amount).op.run()
        new_descent_summary = self.sess.run(self.descent_summary)
        if descent_amount <= 0 :
          descent_amount = 0 ; chosen_point = np.copy(self.X)
        self.writer.add_summary(summary_string,step)
        # self.writer.add_summary(new_extra_loss_summary,step)
        # self.writer.add_summary(new_total_loss_summary,step)
        # self.writer.add_summary(new_descent_summary,step)
        print("\n\n\n\n\n\n\n\nstep number : %2d Loss: %4.2f  descent_amount: %4.2f  \n" %(step,self.current_fitness,descent_amount))
        self.all_inputs.append(self.X)
        self.all_losses.append(self.current_fitness)
        self.all_directions.append(chosen_direction)
        if self.ALLOW_LOGGING and (step % self.bb_log_frq == 0):
          scipy.misc.imsave(os.path.join(self.frames_log_dir,str(step)+".jpg"), inverse_transform(self.Y))
        # print("\n\n\n\n\n The size of X",self.approx_gradient.shape)
        self.X = np.copy(chosen_point)
        self.X = np.clip(self.X,-1,1)


  def approximate_bb_forgradient(self):
    approx_gradient = np.ones_like(self.X)
    temp1 = np.copy(self.X)
    for ii in range(len(self.X)):
      temp1[ii] = (self.X[ii]+self.gradient_perturbation)
      y1 = black_box(temp1,output_size=self.OUT_SIZE,global_step=1,frames_path=self.frames_path)
      f1 = self.sess.run(self.loss,feed_dict={self.x:temp1,self.labels:self.fake_target,self.y:y1})
      approx_gradient[ii] = (f1-self.current_fitness)/(self.gradient_perturbation)
      temp1 = np.copy(self.X)
    return approx_gradient 

  def approximate_bb_gradient(self):
    approx_gradient = np.ones_like(self.X)
    temp1,temp2 = np.copy(self.X) ,np.copy(self.X) 
    for ii in range(len(self.X)):
      # print("\n\n the first ... ",self.X[ii], "   AND   ", self.gradient_perturbation, "   AND   ",(temp1[ii]),"   AND   ",(temp2[ii]) )
      temp1[ii]  += self.gradient_perturbation
      temp2[ii]  -= self.gradient_perturbation
      # print("\n\n the difference between two ... " , norm(temp1-temp2) )
      y1 = black_box(temp1,output_size=self.OUT_SIZE,global_step=1,frames_path=self.frames_path)
      y2 = black_box(temp2,output_size=self.OUT_SIZE,global_step=1,frames_path=self.frames_path)
      f1 = self.sess.run(self.loss,feed_dict={self.x:temp1,self.labels:self.fake_target,self.y:y1})
      f2 = self.sess.run(self.loss,feed_dict={self.x:temp2,self.labels:self.fake_target,self.y:y2})
      print("\n\nf1: %3.5f     f2:  %3.5f" %(f1,f2))
      approx_gradient[ii] = (f1-f2)/(2*self.gradient_perturbation)
      temp1,temp2 = np.copy(self.X) ,np.copy(self.X)
    return approx_gradient 

  def local_random_search(self,random_type="uniform"):
    tried_fittness = []
    tried_x = []
    tried_directions = []
    if random_type is "lhs":
      random_grid = random_lhs(-self.stochastic_perturbation, self.stochastic_perturbation, len(self.X),number_samples=self.META_STEPS)
    for meta in range(self.META_STEPS) :
      # perturbing the variables with random directions
      if random_type is "uniform":
        some_random = np.random.uniform(-self.stochastic_perturbation, self.stochastic_perturbation, self.X.shape)
      elif random_type is "normal":
        some_random = np.random.normal(np.copy(self.X), self.stochastic_perturbation, self.X.shape)
      elif random_type is "lhs":
        some_random = random_grid[meta]
      new_temp_x = np.copy(self.X) + some_random.astype(np.float64)
      y1 = black_box(new_temp_x,output_size=self.OUT_SIZE,global_step=1,frames_path=self.frames_path)
      new_temp_fittness = self.sess.run(self.total_loss,feed_dict={self.x:new_temp_x,self.labels:self.fake_target,self.y:y1})
      tried_fittness.append(new_temp_fittness)
      tried_x.append(new_temp_x)
      tried_directions.append(- some_random)
      # print("     the loss at trial %3d is:  %3.4f" %(meta,new_fittness))
    choesen_x = tried_x[tried_fittness.index(min(tried_fittness))]
    chosen_direction = tried_directions[tried_fittness.index(min(tried_fittness))]
    fitness_decrease = np.copy(self.current_fitness) - min(tried_fittness)
    return choesen_x , fitness_decrease ,chosen_direction


  def visualize_optimum_solution(self):
    best_loss = min(self.all_losses)
    best_input = self.all_inputs[self.all_losses.index(best_loss)]
    print("the best loss is %4.2f at step: %d" %(best_loss,self.all_losses.index(best_loss)))
    _ = black_box(best_input,output_size=self.OUT_SIZE,global_step="best",frames_path=self.frames_log_dir)
    # slim.learning.train(train_tensor, train_log_dir,number_of_steps=STEPS_NUMBER,
    #   save_summaries_secs=60,save_interval_secs=600)




if __name__ == '__main__':
  AVAILABLE_Exps = ["Gradprox","Random","VGD","Lsearch","RBFopt"]
  exp_type = AVAILABLE_Exps[4]
  exp_no = 11
  data_path = "D:\\mywork\\sublime\\GAN2\\data\\celebB"
  base_path = "D:\\mywork\\sublime\\vgd"
  # logits = -1.1*np.ones(10)
  # labels = tf.ones_like(logits)
  # with tf.Session().as_default() as sess:
  #   k = tf.nn.relu(-labels-logits)+ tf.nn.relu(logits-labels)
  #   print("the loss.. : ", tf.reduce_mean(k).eval())
  invbbexp = BlackBoxOptimizer(exp_type=exp_type,exp_no=exp_no,base_path=base_path)
  invbbexp.train()



