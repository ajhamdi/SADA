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
from numbers import Number
from subprocess import Popen, PIPE


class Blender(object):
  def __init__(self, py_file, blend_file=None, *args):
    self._args = list(args)
    self._blend_file = blend_file
    self._commands = []

    # read code and register all top-level non-hidden functions
    with open(py_file, 'r') as file:
      self._code = file.read() + '\n' * 2
      for line in self._code.splitlines():
        if line.startswith('def ') and line[4] != '_':
          self._register(line[4:line.find('(')])

  def __call__(self, blender_func_name, *args, **kwargs):
    args = str(args)[1:-1] + ',' if len(args) > 0 else ''
    kwargs = ''.join([k + '=' + repr(v) + ',' for k, v in kwargs.items()])
    cmd = blender_func_name + '(' + (args + kwargs)[:-1] + ')'
    self._commands.append(cmd)

  def execute(self, timeout=None, encoding='utf-8'):
    # command to run blender in the background as a python console
    cmd = ['blender', '--background', '--python-console'] + self._args
    # setup a *.blend file to load when running blender
    if self._blend_file is not None:
      cmd.insert(1, self._blend_file)
    # compile the source code from the py_file and the stacked commands
    code = self._code + ''.join(l + '\n' for l in self._commands)
    byte_code = bytearray(code, encoding)
    # run blender and the compiled source code
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate(byte_code, timeout=timeout)
    # get the output and print it
    out = out.decode(encoding)
    err = err.decode(encoding)
    skip = len(self._code.splitlines())
    Blender._print(out, err, skip, self._commands)
    # empty the commands list
    self._commands = []

  def _register(self, func):
    call = lambda *a, **k: self(func, *a, **k)
    setattr(self, func, call)

  def _print(out, err, skip, commands):
    def replace(out, lst):
      lst = [''] * lst if isinstance(lst, Number) else lst
      for string in lst:
        i, j = out.find('>>> '), out.find('... ')
        ind = max(i, j) if min(i, j) == -1 else min(i, j)
        out = out[:ind] + string + out[ind + 4:]
      return out
    out = replace(out, skip)
    out = replace(out, ['${ ' + c + ' }$\n' for c in commands])
    print('${Running on Blender}$')
    out = out[:-19]
    print(out)
    err = err[err.find('(InteractiveConsole)') + 22:]
    if err:
      print(err)
    print('${Blender Done}$')
    return out, err


def random_lhs(min_bound,max_bound,dimensions,number_samples=1):
  import lhsmdu
  r_range = abs(max_bound-min_bound)
  lolo = r_range*np.array(lhsmdu.sample(dimensions,number_samples)) + min_bound
  required_list = []
  for ii in range(number_samples):
    required_list.append(lolo[:,ii].reshape(-1))
  return required_list

def shuffle_list(*ls):
  l =list(zip(*ls))

  random.shuffle(l)
  return zip(*l)
def splitting_train_test(all_Xs,all_Ys,percentage,shuffle=True):
  splitting_index = int(percentage*len(all_Xs)/100.0)
  train_x, test_x = all_Xs[0:splitting_index], all_Xs[splitting_index+1::]
  train_y, test_y = all_Ys[0:splitting_index], all_Ys[splitting_index+1::]
  if shuffle:
    train_x , train_y = shuffle_list(train_x,train_y)
    test_x , test_y = shuffle_list(test_x,test_y)
  return list(train_x) ,list(train_y) , list(test_x) , list(test_y)

def objective_function(x,output_size):
  hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
  output = slim.fully_connected(hidden, 2*output_size, scope='objective/fc_2')
  output = slim.fully_connected(output, 3*output_size, scope='objective/fc_3')
  output = slim.fully_connected(output, 3*output_size, scope='objective/fc_4')
  output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_5')

  return output

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
def generator_ann(x,output_size):
  with tf.variable_scope("generator") as scope:
    hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
    output = slim.fully_connected(hidden, 2*output_size, scope='objective/fc_2')
    output = slim.fully_connected(output, 3*output_size, scope='objective/fc_3')
    output = slim.fully_connected(output, 3*output_size, scope='objective/fc_4')
    output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_5')
  return output



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
    self.generation_no = 0
    self.frames_path = os.path.join(base_path,"frames")
    self.generated_path = os.path.join(base_path,"generated")
    self.frames_log_dir = os.path.join(self.frames_path,self.exp_type+"_%d"%(self.exp_no))
    self.generated_frames_train_dir = os.path.join(self.generated_path,"train_%d"%(self.generation_no))
    self.generated_frames_test_dir = os.path.join(self.generated_path,"G_%d_test_"%(self.exp_no))
    self.train_log_dir = os.path.join(base_path,'logs',self.exp_type+"_%d"%(self.exp_no))
    if not tf.gfile.Exists(self.train_log_dir):
      tf.gfile.MakeDirs(self.train_log_dir)
    if not tf.gfile.Exists(self.frames_log_dir):
      tf.gfile.MakeDirs(self.frames_log_dir)
    if self.exp_type is "Generator" and not tf.gfile.Exists(self.generated_frames_train_dir):
      tf.gfile.MakeDirs(self.generated_frames_train_dir)
    if self.exp_type is "Generator" and not tf.gfile.Exists(self.generated_frames_test_dir):
      tf.gfile.MakeDirs(self.generated_frames_test_dir)
    self.n= 6    # the input to blck_box shape
    self.m = 1  # the generator input shape 
    self.N = 100 # the number of data we have
    # self.DATA_LIMIT = 1000
    # np.random.seed(0)
    # self.random_no = np.abs(np.random.randint(self.DATA_LIMIT))
    self.META_LEARN_FLAG = False
    self.ALLOW_LOGGING = True
    self.log_frq = 4
    self.learning_rate=0.002
    self.reg_hyp_param = 10.0
    self.solution_learning_rate = 0.000000005  # multiplied by the approx gradient
    self.init_variance = 0.01
    self.stochastic_perturbation = 0.7   # perturbation to find local min using random search 
    self.gradient_perturbation = 0.003  # perturbation of gradient approxmation
    self.OUT_SIZE = 128
    self.loss_mormalization = 0.5 *(2 * self.OUT_SIZE **2)**2 # to normalize the L2 pixel loss 
    self.batch_size = 16
    self.generate_distribution_size = 500
    self.generation_bound = 0.01
    # print("THe VALUE....  " , self.solution_learning_rate  / self.loss_mormalization )
    self.META_STEPS = 30
    self.epochs = 50
    self.STEPS_NUMBER = 500
    self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # self.X = np.random.random(size=(self.N,self.n)).astype(np.float32)
    # self.Y = np.random.random(size=(self.N,self.m)).astype(np.float32)
    self.target = cv2.imread(os.path.join(self.frames_path,"target"+".jpg"))
    self.target = forward_transform(cv2.cvtColor(self.target,cv2.COLOR_BGR2RGB).astype(np.float32))


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
    self.visualize_optimum()
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

  def train_random(self):
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



    with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
      self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
      tf.global_variables_initializer().run()
      self.all_inputs = []
      self.all_losses = []
      self.start_time = time.time()
      for step in range(self.STEPS_NUMBER):
        self.X = np.random.uniform(-1,1,6)
        self.Y = black_box(self.X,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path)
        self.current_fitness,self.new_summary = self.sess.run([self.loss,self.loss_summary],feed_dict={self.x:self.X,self.labels:self.target,self.y:self.Y})
        self.writer.add_summary(self.new_summary,step)
        print("step number : %2d Loss: %4.2f \n" %(step,self.current_fitness))
        self.all_inputs.append(self.X)
        self.all_losses.append(self.current_fitness)
        if self.ALLOW_LOGGING and (step % self.log_frq == 0):
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
        self.current_fitness,self.new_loss_summary = self.sess.run([self.loss,self.loss_summary],feed_dict={self.x:self.X,self.labels:self.target,self.y:self.Y})
        self.approx_gradient = self.approximate_bb_gradient()
        # self.approx_gradient = self.approximate_bb_forgradient()
        self.gradient_norm.assign(norm(self.approx_gradient)).op.run()
        self.new_gradient_summary = self.sess.run(self.grad_summary)
        self.writer.add_summary(self.new_loss_summary,step)
        self.writer.add_summary(self.new_gradient_summary,step)
        print("\n\n\n\n\n\n\n\nstep number : %2d Loss: %4.2f  gradient_norm: %4.2f  \n" %(step,self.current_fitness,norm(self.approx_gradient)))
        self.all_inputs.append(self.X)
        self.all_losses.append(self.current_fitness)
        if self.ALLOW_LOGGING and (step % self.log_frq == 0):
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
      self.current_fitness,self.new_summary = self.sess.run([self.loss,self.loss_summary],feed_dict={self.x:self.X,self.labels:self.target,self.y:self.Y})
      self.writer.add_summary(self.new_summary,self.global_step)
      print("step number : %2d Loss: %4.2f \n" %(self.global_step,self.current_fitness))
      self.all_inputs.append(self.X)
      self.all_losses.append(self.current_fitness)
      if self.ALLOW_LOGGING and (self.global_step % self.log_frq == 0):
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
        self.loss_summary= tf.summary.scalar('losses/main_loss', self.loss)
        self.extra_loss_summary= tf.summary.scalar('losses/extra_loss', self.extra_loss)
        self.total_loss_summary= tf.summary.scalar('losses/total_loss', self.total_loss)
        self.descent_summary= tf.summary.scalar('descent/main_descent', self.descent_amount)

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
        self.current_fitness,new_loss_summary,new_extra_loss_summary,new_total_loss_summary = self.sess.run([self.total_loss,self.loss_summary,self.extra_loss_summary,
          self.total_loss_summary],feed_dict={self.x:self.X,self.labels:self.target,self.y:self.Y})
        chosen_point, descent_amount, chosen_direction  = self.local_random_search()
        self.descent_amount.assign(descent_amount).op.run()
        new_descent_summary = self.sess.run(self.descent_summary)
        if descent_amount <= 0 :
          descent_amount = 0 ; chosen_point = np.copy(self.X)
        self.writer.add_summary(new_loss_summary,step)
        self.writer.add_summary(new_extra_loss_summary,step)
        self.writer.add_summary(new_total_loss_summary,step)
        self.writer.add_summary(new_descent_summary,step)
        print("\n\n\n\n\n\n\n\nstep number : %2d Loss: %4.2f  descent_amount: %4.2f  \n" %(step,self.current_fitness,descent_amount))
        self.all_inputs.append(self.X)
        self.all_losses.append(self.current_fitness)
        self.all_directions.append(chosen_direction)
        if self.ALLOW_LOGGING and (step % self.log_frq == 0):
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
      f1 = self.sess.run(self.loss,feed_dict={self.x:temp1,self.labels:self.target,self.y:y1})
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
      f1 = self.sess.run(self.loss,feed_dict={self.x:temp1,self.labels:self.target,self.y:y1})
      f2 = self.sess.run(self.loss,feed_dict={self.x:temp2,self.labels:self.target,self.y:y2})
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
      new_temp_fittness = self.sess.run(self.total_loss,feed_dict={self.x:new_temp_x,self.labels:self.target,self.y:y1})
      tried_fittness.append(new_temp_fittness)
      tried_x.append(new_temp_x)
      tried_directions.append(- some_random)
      # print("     the loss at trial %3d is:  %3.4f" %(meta,new_fittness))
    choesen_x = tried_x[tried_fittness.index(min(tried_fittness))]
    chosen_direction = tried_directions[tried_fittness.index(min(tried_fittness))]
    fitness_decrease = np.copy(self.current_fitness) - min(tried_fittness)
    return choesen_x , fitness_decrease ,chosen_direction


  def visualize_optimum(self):
    best_loss = min(self.all_losses)
    best_input = self.all_inputs[self.all_losses.index(best_loss)]
    print("the best loss is %4.2f at step: %d" %(best_loss,self.all_losses.index(best_loss)))
    _ = black_box(best_input,output_size=self.OUT_SIZE,global_step="best",frames_path=self.frames_log_dir)
    # slim.learning.train(train_tensor, train_log_dir,number_of_steps=STEPS_NUMBER,
    #   save_summaries_secs=60,save_interval_secs=600)


  def generate_distribution(self):
    all_Xs = []
    all_Ys = []
    vec= np.array([-0.95,-0.95,0.8,0,0,0])
    focus = np.array([self.generation_bound,self.generation_bound,self.generation_bound,0.9,0.9,0.9])
    for gen in range(self.generate_distribution_size):
      x = np.random.uniform(vec-focus, vec+focus, self.n)
      y = black_box(x,output_size=self.OUT_SIZE,global_step=gen,frames_path=self.generated_frames_train_dir)
      all_Xs.append(x)
      all_Ys.append(y)
    saved_dict = {"x":all_Xs,"y":all_Ys}
    with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'wb') as fp:
      cPickle.dump(saved_dict,fp)

  def learn_distribution_random(self):
    with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'rb') as fp:
      saved_dict = cPickle.load(fp)
    all_Xs , all_Ys = saved_dict["x"] , saved_dict["y"] 
    train_x ,train_y , test_x , test_y = splitting_train_test(all_Xs,all_Ys,percentage=80,shuffle=True)
    x_batches = [train_x[ii:ii+self.batch_size] for ii in range(0, len(train_x), self.batch_size)]
    if len(train_x) % self.batch_size != 0 :
      x_batches.pop()
    # print(len(data_chunks[24]))
    # raise Exception("WAIIIT ....")
    with tf.Graph().as_default() as self.g:
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, self.init_variance),
                          weights_regularizer=slim.l2_regularizer(0.001)):
        self.z = tf.placeholder(tf.float32, shape=[None,self.m])
        self.x = objective_function(self.z,self.n)
        self.G_labels = tf.placeholder(tf.float32, shape=[None,self.n])
        self.y = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.G_loss = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(self.x - self.G_labels)))
        self.loss_summary= tf.summary.scalar('losses/G_loss', self.G_loss)
        slim.losses.add_loss(self.G_loss)
        self.total_loss = slim.losses.get_total_loss()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)



    with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
      self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
      tf.global_variables_initializer().run()

      self.global_step = 0
      self.start_time = time.time()
      for epoch in range(self.epochs):
        for step in range(len(x_batches)):
          # self.global_step = tf.train.get_global_step().eval(self.sess)
          self.Z = np.random.uniform(-1,1,[self.batch_size,self.m])
          # self.Y = black_box(self.X,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path)
          _,self.current_fitness,self.new_summary = self.sess.run([self.optimizer,self.total_loss,self.loss_summary],feed_dict={self.z:self.Z,self.G_labels:x_batches[step]})
          self.writer.add_summary(self.new_summary,self.global_step)
          print("epoch: %d step number: %2d Loss: %4.2f \n" %(epoch,self.global_step,self.current_fitness))
          self.global_step += 1
      self.visualize_optimum_distribution(20)


def learn_distribution_gan(self):
    with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'rb') as fp:
      saved_dict = cPickle.load(fp)
    all_Xs , all_Ys = saved_dict["x"] , saved_dict["y"] 
    train_x ,train_y , test_x , test_y = splitting_train_test(all_Xs,all_Ys,percentage=80,shuffle=True)
    x_batches = [train_x[ii:ii+self.batch_size] for ii in range(0, len(train_x), self.batch_size)]
    if len(train_x) % self.batch_size != 0 :
      x_batches.pop()
    # print(len(data_chunks[24]))
    # raise Exception("WAIIIT ....")
    with tf.Graph().as_default() as self.g:
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, self.init_variance),
                          weights_regularizer=slim.l2_regularizer(0.001)):
        self.z = tf.placeholder(tf.float32, shape=[None,self.m])
        self.x = generator_ann(self.z,self.n)
        self.G_labels = tf.placeholder(tf.float32, shape=[None,self.n])
        self.d = dicrminator_ann(self.G_labels,1)
        self.y = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
        self.G_loss = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(self.x - self.G_labels)))
        self.loss_summary= tf.summary.scalar('losses/G_loss', self.G_loss)
        slim.losses.add_loss(self.G_loss)
        self.total_loss = slim.losses.get_total_loss()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)



    with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
      self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
      tf.global_variables_initializer().run()
      self.global_step = 0
      self.start_time = time.time()
      for epoch in range(self.epochs):
        for step in range(len(x_batches)):
          # self.global_step = tf.train.get_global_step().eval(self.sess)
          self.Z = np.random.uniform(-1,1,[self.batch_size,self.m])
          # self.Y = black_box(self.X,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path)
          _,self.current_fitness,self.new_summary = self.sess.run([self.optimizer,self.total_loss,self.loss_summary],feed_dict={self.z:self.Z,self.G_labels:x_batches[step]})
          self.writer.add_summary(self.new_summary,self.global_step)
          print("epoch: %d step number: %2d Loss: %4.2f \n" %(epoch,self.global_step,self.current_fitness))
          self.global_step += 1
      self.visualize_optimum_distribution(20)
      ### EVALUATION 
      # test_Z = np.random.uniform(-1,1,[20,self.m])
      # test_X = self.sess.run(self.x,feed_dict={self.z:test_Z})
      # print(test_X)
        # self.all_inputs.append(self.X)
        # self.all_losses.append(self.current_fitness)
        # if self.ALLOW_LOGGING and (step % self.log_frq == 0):
        #   scipy.misc.imsave(os.path.join(self.frames_log_dir,str(step)+".jpg"), inverse_transform(self.Y))

  def visualize_optimum_distribution(self,size_dist=20):
    test_Z = np.random.uniform(-1,1,[size_dist,self.m])
    test_X = self.sess.run(self.x,feed_dict={self.z:test_Z})
    print(test_X)
    # print("the best loss is %4.2f at step: %d" %(best_loss,self.all_losses.index(best_loss)))
    for ii in range(size_dist):
      _ = black_box(test_X[ii],output_size=self.OUT_SIZE,global_step=ii,frames_path=self.generated_frames_test_dir)









        # if META_LEARN_FLAG:
        #   tried_directions = []
        #   tried_fittness = []
        #   current_objectives,current_fitness = sess.run([optimization_variables,loss],feed_dict={x:X,labels:Y})
        #   # print("\n\n the current values .. ",current_objectives[0].shape)
        #   for meta in range(META_STEPS) :
        #     for idx , value in enumerate(current_objectives):
        #       some_random = np.random.uniform(-stochastic_perturbation, stochastic_perturbation, value.shape)
        #       new_objective = value + some_random
        #       optimization_variables[idx].assign(new_objective).op.run()
        #       tried_directions.append(- some_random)
        #     new_fittness = sess.run(loss,feed_dict={x:X,labels:Y})
        #     print("     the loss at trial %3d is:  %3.4f" %(meta,new_fittness))
        #     # new_objective = 
        #   tried_fittness.append()
if __name__ == '__main__':
  AVAILABLE_Exps = ["Gradprox","Random","VGD","Lsearch","RBFopt","Generator"]
  exp_type = AVAILABLE_Exps[5]
  exp_no = 1
  data_path = "D:\\mywork\\sublime\\GAN2\\data\\celebB"
  base_path = "D:\\mywork\\sublime\\vgd"
  # logits = -1.1*np.ones(10)
  # labels = tf.ones_like(logits)
  # with tf.Session().as_default() as sess:
  #   k = tf.nn.relu(-labels-logits)+ tf.nn.relu(logits-labels)
  #   print("the loss.. : ", tf.reduce_mean(k).eval())
  bbexp = BlackBoxOptimizer(exp_type=exp_type,exp_no=exp_no,base_path=base_path)
  # bbexp.train()
  # bbexp.visualize_optimum()
  # bbexp.generate_distribution()
  # bbexp.learn_distribution_random()
  bbexp.learn_distribution_gan()


