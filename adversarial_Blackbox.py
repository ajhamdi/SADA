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
import pickle as cPickle
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm
from itertools import chain
from collections import deque
from distutils.dir_util import copy_tree
import shutil
from scipy.sparse import lil_matrix
import scipy 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from scipy import signal
import cv2
# import rbfopt  # black_box_optimization library
from utils import *
from models import *
from scipy.linalg import circulant, norm
from scipy.linalg import dft
import scipy.io as sio
import tensorflow as tf
slim = tf.contrib.slim
import imageio


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

        # x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        # x = tf.reshape(x, [x.shape[0], -1])
        x = slim.flatten(x)
        z = slim.fully_connected(x, output_size, activation_fn=None)
        z_prob = tf.nn.sigmoid(z)
        return z , z_prob


def discrminator_ann(x,output_size,reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
        output = slim.fully_connected(hidden, 10, scope='objective/fc_2')
        output = slim.fully_connected(output, 10, scope='objective/fc_3')
        output = slim.fully_connected(output, 10, scope='objective/fc_4')
        output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_5')
    return output
def generator_ann(x,output_size,min_bound=-1,max_bound=1):
    range_required = np.absolute(max_bound - min_bound).astype(np.float64)
    with tf.variable_scope("generator") as scope:
        hidden = slim.fully_connected(x, 10, scope='objective/fc_1')
        output = slim.fully_connected(hidden, 10, scope='objective/fc_2')
        output = slim.fully_connected(output, 10, scope='objective/fc_3')
        output = slim.fully_connected(output, 10, scope='objective/fc_4')
        output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_5')
        # contrained_output =   range_required * tf.nn.sigmoid(output) + min_bound* tf.ones_like(output)
        contrained_output = tf.nn.tanh(output)
    return contrained_output

def load_dataset_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def find_key(dict,value):
    for key, values in dict.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value == values:
            return key

def match_two_dictionaries(dict1,dict2):
    result_dict = {}
    for key2, vlue2 in dict2.items():
        if vlue2 in list(dict1.values()):
            result_dict[find_key(dict1,vlue2)] = key2 
    return result_dict

def function_batches(function,input_list=range(1000),slice_size=100):
    full_output = []
    x_batches = [input_list[ii:ii+slice_size] for ii in range(0, len(input_list), slice_size)]
    # if len(input_list) % self.slice_size != 0 :
    #     x_batches.pop()
    for ii in range(len(x_batches)):
        full_output.append(function(x_batches[ii]))
    return full_output
# a function that applies the function in the argument ( which accepts athe list of inputs ) as batches and returen a list of batched output 7


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        # color = tuple(np.random.randint(0, 256, 3))
        color = tuple(255,0,0)
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)

def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def black_box(input_vector,output_size=256,global_step=0,frames_path=None,cluster=False,parent_name='car'):
    b = Blender(cluster,'init.py','3d/training_pascal/training.blend')
    b.city_experiment(obj_name="myorigin", vec=input_vector.tolist(),parent_name=parent_name)
    b.save_image(output_size,output_size,path=frames_path,name=str(global_step))
    # b.save_file()
    b.execute()
    image = cv2.imread(os.path.join(frames_path,str(global_step)+".jpg"))
    image = forward_transform(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32))
    return image

def black_box_batch(input_vectors,output_size=256,global_step=0,frames_path=None,cluster=False,parent_name='car'):
    images =[]
    for input_vector in input_vectors:
        try:
            images.append(black_box(np.array(input_vector),output_size,global_step,frames_path,cluster,parent_name))
        except:
            continue
    return images

def read_csv_list(file_name=None,col_indices=[0]):
    import csv
    full_list = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for rows in csv_reader:
            if line_count != 0:
                full_list.append(np.array([float(col) for col in [rows[ii] for ii in range(len(rows)) if ii in col_indices] ]))
                line_count += 1
            else:
                line_count += 1
    return full_list               

def write_csv_list(mylist,file_name=None,col_names=['first']):
    import csv
    with open(file_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(col_names)
        for rows in mylist:
            csv_writer.writerow(list(rows))  
    pass

def normalize_vectors_list(vector_list,old_max,old_min,new_max,new_min):
    old_range = old_max - old_min
    new_range = new_max - new_min
    range_ratio = new_range/ old_range
    matrix = np.array(vector_list)
    matrix = np.broadcast_to(new_min,matrix.shape) + (matrix - np.broadcast_to(old_min,matrix.shape))* np.broadcast_to(range_ratio,matrix.shape)
    return list(matrix)


 
class BlackBoxOptimizer(object):
    CAR_CLASS = 2

    def __init__(self,FLAGS=None,base_path=None):
        self.dataset_nb = FLAGS.dataset_nb
        self.exp_type = FLAGS.exp_type
        self.exp_no = FLAGS.exp_no
        self.class_nb = FLAGS.class_nb
        self.evolution_nb = FLAGS.evolution_nb
        self.is_train = FLAGS.is_train
        self.is_gendist = FLAGS.is_gendist
        self.cont_train = FLAGS.cont_train
        self.optimize_oracle = FLAGS.optimize_oracle
        self.restore_all = FLAGS.restore_all
        self.is_focal = FLAGS.is_focal
        self.is_evolve = FLAGS.is_evolve
        self.keep_bank = FLAGS.keep_bank
        self.full_set = FLAGS.full_set
        self.is_cluster = FLAGS.is_cluster

        self.frames_path = os.path.join(base_path,"frames")
        self.generated_path = os.path.join(base_path,"generated")
        self.checkpoint_path = os.path.join(base_path,"checkpoint")
        self.detector_path = os.path.join(base_path,"detectors")
        if not FLAGS.is_selfdrive:
            from detectors.yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression
            # print("THe VALUE....  " , self.solution_learning_rate  / self.loss_mormalization )
            self.coco_classes = load_dataset_names(os.path.join(self.detector_path,"coco.names"))
            self.pascal_classes = load_dataset_names(os.path.join(self.detector_path,"pascal.names"))
            self.PASCAL_TO_COCO = match_two_dictionaries(self.pascal_classes,self.coco_classes)
            self.pascal_list = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car','chair','diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

        self.frames_log_dir = os.path.join(self.frames_path,self.exp_type,self.exp_type+"_%d"%(self.exp_no))
        self.generated_frames_train_dir = os.path.join(self.generated_path,"train_%d"%(self.dataset_nb))
        self.generated_frames_valid_dir = os.path.join(self.generated_path,"valid_%d"%(self.dataset_nb))
        self.generated_frames_test_dir = os.path.join(self.generated_path,self.exp_type,"test%d_%d"%(self.dataset_nb,self.exp_no))
        self.train_log_dir = os.path.join(base_path,'logs',self.exp_type,"data%d_exp%d"%(self.dataset_nb,self.exp_no))
        if not tf.gfile.Exists(self.train_log_dir):
            tf.gfile.MakeDirs(self.train_log_dir)
        else :
            shutil.rmtree(self.train_log_dir, ignore_errors=True)
            tf.gfile.MakeDirs(self.train_log_dir)
        if not tf.gfile.Exists(self.frames_log_dir):
            tf.gfile.MakeDirs(self.frames_log_dir)
        if self.exp_type is "Adversarial" and not tf.gfile.Exists(self.generated_frames_train_dir):
            tf.gfile.MakeDirs(self.generated_frames_train_dir)
        if self.exp_type is "Adversarial" and not tf.gfile.Exists(self.generated_frames_test_dir):
            tf.gfile.MakeDirs(self.generated_frames_test_dir)


        self.n= FLAGS.nb_paramters   # the input to blck_box shape
        self.m = 5  # the generator z input shape 

        self.ALLOW_LOGGING = True
        self.make_gif = True

        self.valid_size = FLAGS.valid_size
        self.log_frq = FLAGS.log_frq
        self.batch_size = FLAGS.batch_size
        self.K = FLAGS.K
        self.induced_size = FLAGS.induced_size
        self.ind_frq = FLAGS.ind_frq
        self.nb_steps = FLAGS.nb_steps
        self.gendist_size= FLAGS.gendist_size

        self.learning_rate_t = FLAGS.learning_rate_t
        self.learning_rate_g=  FLAGS.learning_rate_g
        self.gan_init_variance = FLAGS.gan_init_variance
        # self.gen_log_frq = 2
        # self.learning_rate = 0.0006 # originally 0.001

        self.retained_size = FLAGS.retained_size # int(0.5*self.K*self.induced_size)
        self.gan_regulaizer = 0.0005
        self.beta1 = 0.5
        self.reg_hyp_param = 10.0
        self.gamma = 1
        self.OUT_SIZE = 340
        self.generation_bound = 0.01
<<<<<<< HEAD
        # print("THe VALUE....  " , self.solution_learning_rate  / self.loss_mormalization )
        self.coco_classes = load_dataset_names(os.path.join(self.detector_path,"coco.names"))
        self.pascal_classes = load_dataset_names(os.path.join(self.detector_path,"pascal.names"))
        self.PASCAL_TO_COCO = match_two_dictionaries(self.pascal_classes,self.coco_classes)
        self.pascal_list = ['aeroplane','bench', 'bicycle', 'boat', 'bottle', 'bus', 'car','chair','diningtable', 'motorbike', 'train', 'truck']
=======
>>>>>>> 28fee06809c59bfaffac144ede54af90f913da2e
        self.conf_threshold=0.05
        self.iou_threshold=0.4
        self.weights_file= os.path.join(self.detector_path, FLAGS.weights_file)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,allow_growth = True)
        self.X_bank = [] ; self.Y_bank = []
        # self.X = np.random.random(size=(self.N,self.n)).astype(np.float32)
        # self.Y = np.random.random(size=(self.N,self.m)).astype(np.float32)
        if self.is_train:
            self.logger = open(os.path.join(self.train_log_dir,"message.txt"),"w") 
        elif self.is_gendist:
            self.logger = open(os.path.join(self.generated_frames_train_dir,"message.txt"),"w") 


    def generate_distribution(self,distribution_type="general"):
        all_Xs = []
        # all_Ys = []
        vec= np.array([-0.95,-0.95,0.8,0,0,0])
        focus = np.array([self.generation_bound,self.generation_bound,self.generation_bound,0.9,0.9,0.9])
        for gen in range(self.gendist_size):
            if distribution_type is "specific":
                x = np.random.uniform(vec-focus, vec+focus, self.n)
            elif distribution_type is "general":
                x = np.random.uniform(-1, 1, self.n)
            elif distribution_type is "zeros":
                x = np.zeros(self.n)
            try :
                y = black_box(x,output_size=self.OUT_SIZE,global_step=gen,frames_path=self.generated_frames_train_dir,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb])
            except:
                continue
            all_Xs.append(x)
            # all_Ys.append(y)
        saved_dict = {"x":all_Xs}
        with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'wb') as fp:
            cPickle.dump(saved_dict,fp)
        self.logger.write("The number of generated images : %d" %(len(all_Xs)) )


    def learn_selfdrive(self):
        all_Xs = read_csv_list(file_name=os.path.join('./','selfdriving/input.csv'),col_indices=[0,1,2])
        max_rows = np.max(np.array(all_Xs),axis=0)
        min_rows = np.min(np.array(all_Xs),axis=0)
        all_Xs = normalize_vectors_list(all_Xs,old_max=max_rows,old_min=min_rows,new_max=np.ones_like(max_rows),new_min=-np.ones_like(min_rows))
        all_scores =  read_csv_list(file_name=os.path.join('./','selfdriving/output.csv'),col_indices=[3])
        sorted_indices = np.argsort(np.array(all_scores).flatten(),axis=0).tolist()
        chosen_x =  self.K*[all_Xs[ii] for ii in sorted_indices[:self.induced_size]]
        chosen_scores =  self.K*[all_scores[ii] for ii in sorted_indices[:self.induced_size]]
        with tf.Graph().as_default() as self.g:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.truncated_normal_initializer(0.0, self.gan_init_variance),
                                                    weights_regularizer=slim.l2_regularizer(0.0001)):
                self.z = tf.placeholder(tf.float32, shape=[None,self.m])
                self.x = generator_ann(self.z,self.n)
                self.G_labels = tf.placeholder(tf.float32, shape=[None,self.n])
                self.d_real = discrminator_ann(self.G_labels,1)
                self.d_fake = discrminator_ann(self.x,1,reuse=True)

                self.y = tf.placeholder(tf.float32, (self.OUT_SIZE,self.OUT_SIZE, 3))
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real,labels= tf.ones_like(self.d_real)))
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
                self.d_loss = self.d_loss_fake + self.d_loss_real
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,labels= tf.ones_like(self.d_fake))) 
                self.g_loss_summary= tf.summary.scalar('losses/G_loss', self.g_loss)
                self.d_loss_summary= tf.summary.scalar('losses/D_loss', self.d_loss)

                g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
                d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
                all_vars = g_vars  + d_vars
                self.saver =  tf.train.Saver(var_list=all_vars)
                self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_g, beta1=self.beta1).minimize(self.g_loss,var_list=g_vars)
                self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t, beta1=self.beta1).minimize(self.d_loss,var_list=d_vars)




        with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
            self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
            tf.global_variables_initializer().run()
            self.global_step = 0
            self.start_time = time.time()
            for step in range(self.nb_steps):
                x_batches, _ = sample_batch(chosen_x,self.batch_size)
                self.Z = np.random.normal(np.zeros(self.m),1,[self.batch_size,self.m])
                _,self.current_d_loss,new_summary = self.sess.run([self.d_optimizer,self.d_loss,self.d_loss_summary],feed_dict={self.z:self.Z,self.G_labels:np.array(x_batches)})
                self.writer.add_summary(new_summary,self.global_step)
                _,self.current_g_loss,new_summary = self.sess.run([self.g_optimizer,self.g_loss,self.g_loss_summary],feed_dict={self.z:self.Z})
                _,self.current_g_loss,new_summary = self.sess.run([self.g_optimizer,self.g_loss,self.g_loss_summary],feed_dict={self.z:self.Z})
                self.writer.add_summary(new_summary,self.global_step)
                print("step number: %2d g_Loss: %2.5f ,d_Loss: %2.5f \n" %(self.global_step,self.current_g_loss,self.current_d_loss))
                if ((step+1) % self.log_frq == 0):
                    self.saver.save(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model"), global_step=0, write_meta_graph=False)
                self.global_step += 1


            generated = self.generate_selfdrive(nb_paramters=self.valid_size)
            generated = normalize_vectors_list(generated,new_max=max_rows,new_min=min_rows,old_max=np.ones_like(max_rows),old_min=-np.ones_like(min_rows))
            write_csv_list(generated,file_name=os.path.join('./','selfdriving/generated.csv'),col_names=['z_pos','pitch','weather'])

    def generate_selfdrive(self,nb_paramters=20):
        test_Z = np.random.normal(np.zeros(self.m),1,[nb_paramters,self.m])
        test_X = self.sess.run(self.x,feed_dict={self.z:test_Z})
        return test_X.tolist()

    def learn_bbgan(self,random_type="uniform"):
        improvements = []
        # self.real_targets = read_images_to_np(path=os.path.join(self.frames_path,"real_%d"%(real_no)),h=self.OUT_SIZE,w=self.OUT_SIZE,d_type=np.float32,mode="RGB")
        # self.real_targets = [forward_transform(x) for x in self.real_targets ]
        # self.target_no = -1
        # if augment:
        #     self.real_targets = self.real_targets +  add_salt_pepper_noise(self.real_targets) + add_gaussian_noise(self.real_targets)  + flip_images(self.real_targets) 
        # scipy.misc.imsave(os.path.join(self.frames_path,"real_target.jpg"),inverse_transform(self.real_targets[self.target_no]))
        with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'rb') as fp:
            saved_dict = cPickle.load(fp)
        self.all_Xs = saved_dict["x"] 
        self.all_Ys, missing_indices  = my_read_images(self.generated_frames_train_dir,self.OUT_SIZE,self.OUT_SIZE,extension="jpg",d_type=np.float32,normalize=True)
        if len(self.all_Ys) != len(self.all_Xs):
            raise Exception("some images were not read properly ... the corrsponding Xs are not correct")

        self.retained_Ys = self.all_Ys.copy()      
        for self.evolve_step in range(self.evolution_nb):
            improvements.append(self.train_bbgan())
        for ii in range(self.evolution_nb):
            print("the score of step %d is : "%(ii),improvements)

    def train_bbgan(self):
        with tf.Graph().as_default() as self.g:
            self.z = tf.placeholder(tf.float32, shape=[None,self.m])
            self.x_ind = tf.placeholder(tf.float32, shape=[None,self.n])
            self.oracle_labels = tf.placeholder(tf.float32, shape=[None,self.OUT_SIZE,self.OUT_SIZE, 3])
            self.oracle_scores = tf.placeholder(tf.float32, shape=[None,1])
            self.focal_weights =self.oracle_scores ** self.gamma
            self.focal_weights_avg = tf.reduce_mean(self.focal_weights)
            self.y = tf.placeholder(tf.float32, [None,self.OUT_SIZE,self.OUT_SIZE, 3])

            with tf.device('/GPU:0'):
                with tf.variable_scope('detector'):
                    detections = yolo_v3(self.y, len(self.coco_classes), data_format='NHWC')
                    load_ops = load_weights(tf.global_variables(scope='detector'), self.weights_file)

                self.boxes = detections_boxes(detections)


            with tf.device('/GPU:1'):
                with slim.arg_scope([slim.fully_connected],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.truncated_normal_initializer(0.0, self.gan_init_variance),
                                                    weights_regularizer=slim.l2_regularizer(self.gan_regulaizer)):
                    self.x = generator_ann(self.z,self.n)
                    self.transmitter_good = discrminator_ann(self.x_ind,1)
                    self.transmitter_bad = discrminator_ann(self.x,1,reuse=True)
     
            if self.is_focal:
                self.t_loss_good  = tf.reduce_mean(tf.losses.compute_weighted_loss(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.transmitter_good,
                    labels= tf.ones_like(self.transmitter_good)),weights=(self.focal_weights)))   / self.focal_weights_avg  
            else :
                self.t_loss_good = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.transmitter_good,labels= tf.ones_like(self.transmitter_good)))
            # self.t_loss_bad  = tf.reduce_mean(tf.losses.compute_weighted_loss(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.transmitter_bad,
            #   labels= tf.zeros_like(self.transmitter_bad)),weights=(self.focal_weights_avg * tf.ones_like(self.focal_weights))))
            # self.g_loss = tf.reduce_mean(tf.losses.compute_weighted_loss(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.transmitter_bad,
            #   labels= tf.ones_like(self.transmitter_bad)),weights=(self.focal_weights_avg * tf.ones_like(self.focal_weights))))
            self.t_loss_bad  =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.transmitter_bad,labels= tf.zeros_like(self.transmitter_bad)))
            self.g_loss =   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.transmitter_bad,labels= tf.ones_like(self.transmitter_bad)))
            t_regularization_loss = slim.losses.get_regularization_losses(scope="discriminator")[0]
            self.t_loss =    (self.t_loss_bad + self.t_loss_good)  # + t_regularization_loss

            g_loss_summary= tf.summary.scalar('losses/G_loss', self.g_loss)
            t_loss_real_summary= tf.summary.scalar('losses/t_loss_good', self.t_loss_good)
            t_loss_bad_summary= tf.summary.scalar('losses/t_loss_bad', self.t_loss_bad)
            t_loss_summary= tf.summary.scalar('losses/t_loss', self.t_loss)
            self.total_summary = tf.summary.merge([g_loss_summary,t_loss_summary,t_loss_real_summary,t_loss_bad_summary]) # 
            g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
            # d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='detector')
            all_vars = g_vars  + t_vars
            self.saver =  tf.train.Saver(var_list=all_vars)
            # restorer = tf.train.Saver(var_list=d_vars)
            # slim.losses.add_loss(self.g_loss)
            # self.total_loss = slim.losses.get_total_loss()
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_g, beta1=self.beta1).minimize(self.g_loss,var_list=g_vars)
            self.t_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t, beta1=self.beta1).minimize(self.t_loss,var_list=t_vars)



        with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
            self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
            tf.global_variables_initializer().run()
            self.sess.run(load_ops)
            if self.cont_train:
                # if self.restore_all:
                self.saver.restore(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model-0"))
                # else:
                #     restorer.restore(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model-0"))
            if self.is_train:
                self.start_time = time.time()
                self.global_step = 0
                for step in range(self.nb_steps):
                    print("step:",step)
                    self.Z = np.random.normal(np.zeros(self.m),1,[self.batch_size,self.m])
  

                    if (step % self.ind_frq == 0):
                        self.inducer_bbgan(induced_size=self.induced_size)
 
                    ## evolutionary step
                    if self.keep_bank or self.full_set:
                        self.X_IND, indx = sample_batch(self.X_bank,self.batch_size)
                        self.Y = [self.Y_bank[ii] for ii in indx]

                    ### training step
                    if not self.optimize_oracle:
                        current_oracle_scores = self.detector_agent(self.Y)

                    _,current_t_loss = self.sess.run([self.t_optimizer,self.t_loss],
                         feed_dict={self.z:self.Z,self.y:self.Y,self.x_ind:self.X_IND,self.oracle_scores:current_oracle_scores})

                    _,current_g_loss = self.sess.run([self.g_optimizer,self.g_loss],
                         feed_dict={self.z:self.Z,self.y:self.Y,self.oracle_scores:np.array(current_oracle_scores)})
                    _,current_g_loss,new_summary = self.sess.run([self.g_optimizer,self.g_loss,self.total_summary],
                         feed_dict={self.z:self.Z,self.y:self.Y,self.x_ind:self.X_IND,self.oracle_scores:current_oracle_scores})
                    self.writer.add_summary(new_summary,self.global_step)

                    self.logger.write("step number: %2d , train t_Loss: %2.5f train g_Loss: %2.5f \n" %(self.global_step,current_t_loss,current_g_loss))

                    self.global_step += 1
                    if ((step+1) % self.log_frq == 0):
                        avg_loss, self.all_prob , stdloss,self.all_stdprob = self.validating_bbgan(valid_size=self.valid_size)
                        self.visualize_bbgan(step=step)
                        # self.saver.save(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model"), global_step=self.global_step, write_meta_graph=False)
                        self.saver.save(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model"), global_step=0, write_meta_graph=False)

                        self.logger.write("\n\nepoch: %d validation: %2.5f   standard: %2.5f   improvmetn: %2.5f \n\n" %(step,avg_loss,stdloss,avg_loss-stdloss))
            else : 
                self.saver.restore(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model-0"))
                avg_loss, self.all_prob , stdloss,self.all_stdprob = self.validating_bbgan(valid_size=self.valid_size)
                self.visualize_bbgan()
            # print("\n\n\n validation loss : ", avg_loss)
            if self.is_evolve:
                retained_X , inxs = sample_batch(self.all_Xs,self.retained_size) ; retained_Y =  [self.all_Ys[ii] for ii in inxs ]
                self.all_Xs = self.X_bank + self.K * self.test_X.tolist() + retained_X
                self.all_Ys = self.Y_bank + self.K * self.test_targets + retained_Y

            return avg_loss-stdloss

    def inducer_bbgan(self,induced_size=32):
        if self.keep_bank:
            X_IND, indx = sample_batch(self.all_Xs, self.K* induced_size)
            Y = [self.all_Ys[ii] for ii in indx ]
            current_oracle_scores = function_batches(self.detector_agent,Y,50)
            current_oracle_scores = np.vstack(np.array(current_oracle_scores)).reshape(-1)
            # sorted_indices = flip(np.argsort(current_oracle_scores.flatten()),axis=0).tolist()
            sorted_indices = np.argsort(current_oracle_scores,axis=0).tolist()
            self.X_bank = self.X_bank + [X_IND[ii] for ii in sorted_indices[:induced_size]]
            self.Y_bank = self.Y_bank + [Y[ii] for ii in sorted_indices[:induced_size]]

        elif self.full_set:
            self.ind_frq = 100000
            X_IND = self.all_Xs
            Y = self.all_Ys
            current_oracle_scores = function_batches(self.detector_agent,Y,50)
            current_oracle_scores = np.vstack(np.array(current_oracle_scores)).reshape(-1)
            # sorted_indices = flip(np.argsort(current_oracle_scores),axis=0).tolist()
            sorted_indices = np.argsort(current_oracle_scores,axis=0).tolist()
            self.X_bank = self.X_bank + self.K*[X_IND[ii] for ii in sorted_indices[:induced_size]]
            self.Y_bank = self.Y_bank + self.K*[Y[ii] for ii in sorted_indices[:induced_size]]

        else:
            self.X_IND, indx = sample_batch(self.all_Xs,self.batch_size)
            self.Y = [self.all_Ys[ii] for ii in indx ]
    

    def detector_agent(self, imgs, class_detected=None):
        class_detected = self.PASCAL_TO_COCO[self.class_nb]
        imgs = [inverse_transform(img) for img in imgs]
        detected_boxes = self.sess.run(self.boxes, feed_dict={self.y: np.array(imgs, dtype=np.float32)})
        filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=self.conf_threshold,
                     iou_threshold=self.iou_threshold)
        result = []
        for image in filtered_boxes:
            score = 0 if class_detected not in image else np.max([score for box, score in image[class_detected]])
            result.append(score)
        # print(self.coco_classes[class_detected],result)
        return np.expand_dims(np.array(result, dtype=np.float32),axis=-1)
    
    # def detection_batches(self, imgs, class_detected=CAR_CLASS):


    def validating_bbgan(self,valid_size=32):
        test_Z = np.random.normal(np.zeros(self.m),1,[valid_size,self.m])
        self.test_X = self.x.eval(feed_dict={self.z:test_Z},session=self.sess)
        self.test_targets = black_box_batch(self.test_X.tolist(),output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb])
        test_prob = self.detector_agent(np.array(self.test_targets))

        if not self.is_evolve:
            self.test_std,_ = sample_batch(self.all_Ys,valid_size)
        else :
            self.test_std,_ = sample_batch(self.retained_Ys,valid_size)


        test_stdprob = self.detector_agent(np.array(self.test_std))
        # all_losses.append(loss)
        all_prob =  test_prob.flatten().tolist()
        all_stdprob = test_stdprob.flatten().tolist()
        # print("the average of %d validation score :" %(len(test_prob)), np.mean(np.array(all_prob),axis=1))
        avg_loss = np.mean(test_prob.flatten())
        avg_stdloss = np.mean(test_stdprob.flatten())

        return avg_loss ,all_prob ,avg_stdloss, all_stdprob

    def visualize_bbgan(self,step=100):
        # for ii , img in enumerate(sorted_validation):
        #   scipy.misc.imsave(os.path.join(self.generated_frames_test_dir,"s_%d.jpg"%(ii)),inverse_transform(img))
        if self.make_gif:
            imageio.mimsave(os.path.join(self.generated_frames_test_dir,"s_%d_%d.gif"%(step,self.evolve_step)),[inverse_transform(img) for img in self.test_targets])
            # imageio.mimsave(os.path.join(self.generated_frames_test_dir,"r_%d.gif"%(epoch)),[inverse_transform(img) for img in self.test_std])
        # if self.is_cluster:

        plt.figure()
        plt.hist([self.all_prob,self.all_stdprob],color=["b","r"], label=["BBGAN","Random"], bins=100, range=(0.0,1.0))
        plt.legend()
        plt.savefig(os.path.join(self.generated_frames_test_dir,"histogram_%d_%d.jpg"%(step,self.evolve_step)))
        return


if __name__ == '__main__':
    AVAILABLE_Exps = ["Adversarial"]
    exp_type = AVAILABLE_Exps[0]
    epochs_list = [2,4,6,8]
    real_list = [0,1,2]
    base_path = os.getcwd()
    exp_no = 3

    # for epoch in epochs_list:
    # for real_no in real_list:
    for i in range(3):
        bbexp = BlackBoxOptimizer(exp_type=exp_type,exp_no=exp_no,base_path=base_path)
    #   bbexp.learn_bbgan(search=False,augment=True,train=True,grid_space= 50,cont_train=False ,optimize_oracle=True, hp_iterations=10,epochs=7,
    #     restore_all=True,log_frq=4,real_no=real_no,valid_size=2,evolve=False,keep_bank=False)
    #   del bbexp
    #   for ii in range(2):  
    # # for ii in range(10):
    # #   exp_no = 20 + ii
        # bbexp = BlackBoxOptimizer(exp_type=exp_type,exp_no=exp_no,base_path=base_path)
        bbexp.learn_bbgan(search=False,augment=True,train=True,grid_space= 50,cont_train=False ,optimize_oracle=False, hp_iterations=10,epochs=20,
          restore_all=False,log_frq=2,valid_size=20,focal=False,evolve=False,keep_bank=True,full_set=False)
        exp_no = exp_no + 1 
        del bbexp


    # bbexp.generate_distribution()
    # bbexp.learn_distribution_random()
    # bbexp.learn_distribution_gan()

