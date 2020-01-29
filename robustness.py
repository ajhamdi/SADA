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
from collections import OrderedDict
from glob import glob
from scipy import spatial
from tqdm import tqdm
from itertools import chain
from collections import deque
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
from scipy.sparse import lil_matrix
import scipy 
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
# import seaborn as sns
from scipy import signal
import cv2
# import rbfopt  # black_box_optimization library
from detectors.yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression
from utils import *
from models import *
from scipy.linalg import circulant, norm
from scipy import linalg
from sklearn import mixture
import scipy.io as sio
import tensorflow as tf
import imageio

def sample_from_learned_gaussian(points_to_learn,n_components=1,n_samples=10,is_truncate=True ,is_reject=False ,min_value=-1,max_value=1):
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full',max_iter=50000).fit(points_to_learn)
    if is_truncate:
        return np.clip(gmm.sample(n_samples=n_samples)[0],min_value,max_value)
    elif is_reject:
        sample_list = []
        MAX_ITER = 100000000
        iteration = 0
        a = list(gmm.sample(n_samples=100*n_samples)[0])    
        while len(sample_list)< n_samples and iteration<MAX_ITER:
            if (a[iteration] >= min_value).all() and (a[iteration] <= max_value).all():
                sample_list.append(a[iteration])
            iteration += 1
        return np.array(sample_list)
    else :
        return gmm.sample(n_samples=n_samples)[0]  #, gmm.means_, gmm.covariances


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

def string_to_float_list(A):
    return [float(x) for x in A[1:-1].split(',')]


def check_folder(data_dir):
    """
    checks if folder exists and create if doesnt exist
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

def discrminator_ann(x,output_size,reuse=False,network_size=3):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        output = slim.fully_connected(x, 10, scope='objective/fc_1')
        for ii in range(2,2+network_size):
            output = slim.fully_connected(output, 10, scope='objective/fc_%d'%(ii))
        output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_%d'%(2+network_size))
    return output
def generator_ann(x,output_size,min_bound=-1,max_bound=1,network_size=3):
    range_required = np.absolute(max_bound - min_bound).astype(np.float64)
    with tf.variable_scope("generator") as scope:
        output = slim.fully_connected(x, 10, scope='objective/fc_1')
        for ii in range(2,2+network_size):
            output = slim.fully_connected(output, 10, scope='objective/fc_%d'%(ii))
        output = slim.fully_connected(output, output_size,activation_fn=None, scope='objective/fc_%d'%(2+network_size))
        # contrained_output =   range_required * tf.nn.sigmoid(output) + min_bound* tf.ones_like(output)
        contrained_output = tf.nn.tanh(output)
    return contrained_output




def prepare_config_dict(mydict,ommit_list=[]):
    for k in ommit_list:
        mydict.pop(k, None)
    for k ,v in mydict.items():
        if isinstance(v, bool):
            mydict[k] = int(v)
    return mydict

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


def black_box(input_vector,output_size=256,global_step=0,frames_path=None,cluster=False,parent_name='car',scenario_nb=0):
    b = Blender(cluster,'init.py','3d/training_pascal/training.blend')
    b.city_experiment(obj_name="myorigin", vec=np.array(input_vector).tolist(),parent_name=parent_name,scenario_nb=scenario_nb)
    b.save_image(output_size,output_size,path=frames_path,name=str(global_step))
    # b.save_file()
    b.execute()
    image = cv2.imread(os.path.join(frames_path,str(global_step)+".jpg"))
    image = forward_transform(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32))
    return image

def black_box_batch(input_vectors,output_size=256,global_step=0,frames_path=None,cluster=False,parent_name='car',scenario_nb=0):
    images =[]
    for input_vector in input_vectors:
        try:
            images.append(black_box(np.array(input_vector),output_size,global_step,frames_path,cluster,parent_name,scenario_nb))
            # images.append(black_box(np.array(input_vector),output_size,len(images),frames_path,cluster,parent_name,scenario_nb))

        except:
            continue
    # print("&&&&&&&&&&&&&&&&&&&&&\n\n",np.linalg.norm(np.mean(np.array(images) - np.broadcast_to(np.mean(np.array(images),axis=0),np.array(images).shape),axis=2),ord="fro"))
    return images


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
        FLAGS = self.fix_paramters_to_scenario(FLAGS)
        for k ,v in FLAGS.flag_values_dict().items():
            setattr(self, k,v )
        ommit_list = ['exp_type','weights_file','h','help','helpfull','helpshort']    
        self.config_dict = prepare_config_dict(FLAGS.flag_values_dict(),ommit_list=ommit_list)


        self.frames_path = os.path.join(base_path,"frames")
        self.generated_path = os.path.join(base_path,"generated")
        self.checkpoint_path = os.path.join(base_path,"checkpoint")
        self.detector_path = os.path.join(base_path,"detectors")
        # if not FLAGS.is_selfdrive:
        #     from detectors.yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression
            # print("THe VALUE....  " , self.solution_learning_rate  / self.loss_mormalization )
        if FLAGS.is_genknn:
            self.frames_path = os.path.join(self.frames_path,"KNN_{}".format(str(self.exp_no)))
            check_folder(self.frames_path)
            self.frames_path_gen = os.path.join(self.frames_path,"gen")
            self.frames_path_knn = os.path.join(self.frames_path, "knn")
            check_folder(self.frames_path_gen) ; check_folder(self.frames_path_knn)

        self.coco_classes = load_dataset_names(os.path.join(self.detector_path,"coco.names"))
        self.pascal_classes = load_dataset_names(os.path.join(self.detector_path,"pascal.names"))
        self.PASCAL_TO_COCO = match_two_dictionaries(self.pascal_classes,self.coco_classes)
        self.pascal_list = ['aeroplane','bench', 'bicycle', 'boat', 'bottle', 'bus', 'car','chair','diningtable', 'motorbike', 'train', 'truck']
        if self.scenario_nb == 0:
            # self.paramters_list = ["camera distance to object"  ,"Camera azimuth(-180,180)" ,"camera pitch (0,50)" ,"light azimth wrt camera(-180,180)" , "light pitch (0,90)",
            #  "texture R-channel","texture G-channel","texture B-channel"]
            self.paramters_list =['cameraDistanceToObject',    'CameraAzimuth__180_180_',    'cameraPitch_0_50_',    'lightAzimthWrtCamera__180_180_',    'lightPitch_0_90_',    'textureR_channel',    'textureG_channel',    'textureB_channel']
        elif self.scenario_nb in list(range(1,6)):
            self.paramters_list = ["Camera azimuth(-180,180)" ,"camera pitch (0,50)" ,"light azimth wrt camera(-180,180)" , "light pitch (0,90)"]
        elif self.scenario_nb in list(range(6,11)):
            self.paramters_list = ["Camera azimuth(-180,180)" ,"camera pitch (0,50)" ,"occluder horizontal shift"]

        self.generated_frames_train_dir = os.path.join(self.generated_path,"train_%d"%(self.dataset_nb),str(self.scenario_nb),self.pascal_list[self.class_nb])
        self.generated_frames_valid_dir = os.path.join(self.generated_path,"valid_%d"%(self.dataset_nb),str(self.scenario_nb),self.pascal_list[self.class_nb])
        self.generated_frames_test_dir = os.path.join(self.generated_path,self.exp_type,"test_%d"%(self.dataset_nb),str(self.scenario_nb),self.pascal_list[self.class_nb],str(self.exp_no))
        self.train_log_dir = os.path.join(base_path,'logs',self.exp_type,"data_%d"%(self.dataset_nb),str(self.scenario_nb),self.pascal_list[self.class_nb],str(self.exp_no))
        if not tf.gfile.Exists(self.train_log_dir) and not self.is_gendist:
            tf.gfile.MakeDirs(self.train_log_dir)
        else :
            shutil.rmtree(self.train_log_dir, ignore_errors=True)
            tf.gfile.MakeDirs(self.train_log_dir)
        if not tf.gfile.Exists(self.generated_frames_train_dir):
            tf.gfile.MakeDirs(self.generated_frames_train_dir)
        if not tf.gfile.Exists(self.generated_frames_test_dir) and not self.is_gendist:
            tf.gfile.MakeDirs(self.generated_frames_test_dir)


        self.ALLOW_LOGGING = True

        self.retained_size = FLAGS.retained_size # int(0.5*self.K*self.induced_size)
        self.gan_regulaizer = 0.0005
        self.beta1 = 0.5
        self.reg_hyp_param = 10.0
        self.gamma = 1
        self.OUT_SIZE = 340
        self.generation_bound = 0.01

        self.conf_threshold=0.05
        self.SUCCESS_THRESHOLD = 0.3
        self.iou_threshold=0.4
        self.weights_file= os.path.join(self.detector_path, FLAGS.weights_file)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth = True)
        self.X_bank = [] ; self.Y_bank = []
        # self.X = np.random.random(size=(self.N,self.nb_parameters)).astype(np.float32)
        # self.Y = np.random.random(size=(self.N,self.z_dim)).astype(np.float32)
        if self.is_train:
            self.logger = open(os.path.join(self.train_log_dir,"message.txt"),"w") 
        elif self.is_gendist:
            self.logger = open(os.path.join(self.generated_frames_train_dir,"message.txt"),"w") 


    def fix_paramters_to_scenario(self,FLAGS):
        if FLAGS.is_varsteps:
            steps_list = [400,420,400,200,450,250,550,400,420,480]
        else:
            steps_list = 10*[FLAGS.nb_steps]
        FLAGS.exp_no = np.random.randint(100000,1000000)
        if FLAGS.scenario_nb in list(range(1,6)):
            FLAGS.nb_parameters = 4
            FLAGS.nb_steps = steps_list[FLAGS.scenario_nb-1]
        elif FLAGS.scenario_nb in list(range(6,11)):
            FLAGS.nb_parameters = 3
            FLAGS.nb_steps = steps_list[FLAGS.scenario_nb-1]
        if FLAGS.scenario_nb in [1,3,6,8]:
            FLAGS.class_nb = 6
        elif FLAGS.scenario_nb in [2,4,5,7,9,10]:
            FLAGS.class_nb = 9
        return FLAGS
    def generate_distribution(self,distribution_type="general"):
        """
        gerberate distribution of the parameters we have of size  self.gendist_size

        """
        all_Xs = []
        saved_dict = {}
        # all_Ys = []
        vec= np.array([-0.95,-0.95,0.8,0,0,0])
        focus = np.array([self.generation_bound,self.generation_bound,self.generation_bound,0.9,0.9,0.9])
        for gen in range(self.gendist_size):
            if distribution_type is "specific":
                x = np.random.uniform(vec-focus, vec+focus, self.nb_parameters)
            elif distribution_type is "general":
                x = np.random.uniform(-1, 1, self.nb_parameters)
            elif distribution_type is "zeros":
                x = np.zeros(self.nb_parameters)
            try :
                y = black_box(x,output_size=self.OUT_SIZE,global_step=gen,frames_path=self.generated_frames_train_dir,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)
            except:
                continue
            all_Xs.append(x)
            # all_Ys.append(y)
            if (gen%10) == 0:
                with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'wb') as fp:
                    saved_dict = {'x':all_Xs}
                    cPickle.dump(saved_dict,fp)
        with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'wb') as fp:
            saved_dict = {'x':all_Xs}
            cPickle.dump(saved_dict,fp)
        self.logger.write("The number of generated images : %d" %(len(all_Xs)) )



    def generate_set(self):

        # sphere_params = pd.read_csv(os.path.join(self.generated_path,"requested_params","{}_gp_regression_best.csv".format(str(self.class_nb).rjust(2, '0'))))
        # sphere_params = pd.read_csv(os.path.join(self.generated_path,"requested_params","{}_svm_multi_best.csv".format(str(self.class_nb).rjust(2, '0'))))
        sphere_params = pd.read_csv(os.path.join(self.generated_path,"requested_params","{}_gmm.csv".format(str(self.class_nb).rjust(2, '0'))))
        

        mid_list = np.array([list(sphere_params[self.paramters_list[param]]) for param in range(self.nb_parameters) ]).T
        self.X_bank = [mid_list[ii,:] for ii in range(mid_list.shape[0])]
        print("start learning GP regression")
        for ii , param in enumerate(self.X_bank):
            _ = black_box(np.array(param),output_size=self.OUT_SIZE,global_step=ii,frames_path=self.frames_path,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)

    def generated_nearest_neighbor(self):
        self.nb_parameters = 8
        self.best_exp_nb_dict = {'aeroplane':47696, 'bench':33638, 'bicycle':38004, 'boat':76661, 'bottle':67537, 'bus':48619, 'car':80804, 'chair':71567, 'diningtable':21909, 'motorbike':39234, 'train':63706, 'truck':69093}
        self.best_exp_no = self.best_exp_nb_dict[self.pascal_list[self.class_nb]]
        self.generate_dir = os.path.join("")
        self.paramters_list_gen =['cameraDistanceToObject',    'CameraAzimuth__180_180_',    'cameraPitch_0_50_',    'lightAzimthWrtCamera__180_180_',    'lightPitch_0_90_',    'textureR_channel',    'textureG_channel',    'textureB_channel']
        self.paramters_list_all = ["camera distance to object", "Camera azimuth(-180,180)", "camera pitch (0,50)", "light azimth wrt camera(-180,180)", "light pitch (0,90)","texture R-channel","texture G-channel","texture B-channel"]
        all_params = pd.read_csv(os.path.join(self.generated_path,"all_params","class_{}.csv".format(str(self.class_nb))))

        all_params_list = np.array([list(all_params[self.paramters_list_all[param]]) for param in range(self.nb_parameters)]).T
        generated_parms = pd.read_csv(open(os.path.join(self.generated_path, "requested_params", "{}".format(
            str(self.best_exp_no)), "test_params.csv"), 'rU'), encoding='utf-8', engine='c')
        generated_parms_list = np.array(
            [list(generated_parms[self.paramters_list_gen[param]]) for param in range(self.nb_parameters)]).T
        self.X_all = [all_params_list[ii, :]for ii in range(all_params_list.shape[0])]
        tree = spatial.KDTree(self.X_all) 
        self.X_generated = [generated_parms_list[ii, :]
                            for ii in range(generated_parms_list.shape[0])]
        result_dict ={} ; all_gen_indx = [] ; all_knn_indx = [] ; all_knn_param_distance = [] ;  all_knn_image_distance = []
        print("start generating NN")
        for ii, param in enumerate(self.X_generated):
            gen_image = black_box(np.array(param),output_size=self.OUT_SIZE,global_step=ii,frames_path=self.frames_path_gen,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)
            # print("@@@@ ", np.max(np.max(np.max(gen_image))),np.min(np.min(np.min(gen_image))) )
            # raise Exception("HHHH")
            param_distance, target_indx =  tree.query(param)
            nearest_param = self.X_all[target_indx]
            nearest_image = black_box(np.array(nearest_param),output_size=self.OUT_SIZE,global_step=ii,frames_path=self.frames_path_knn,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)
            image_distance = np.linalg.norm(nearest_image-gen_image)

            all_gen_indx.append(ii)
            all_knn_indx.append(target_indx)
            all_knn_param_distance.append(param_distance)
            all_knn_image_distance.append(image_distance)

        result_dict = {"gen_image_nb":all_gen_indx , "neigbor_nb":all_knn_indx , "pram_distance":all_knn_param_distance ,  "img_distance":all_knn_image_distance }
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(os.path.join(self.generated_path, "requested_params", "{}".format(
            str(self.best_exp_no)), "test_params_KNN_{}.csv".format(str(self.exp_no))), sep=',', index=False)

    def learn_bbgan(self,random_type="uniform"):
        with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'rb') as fp:
            saved_dict = cPickle.load(fp)
        self.all_Xs = saved_dict["x"] 
        self.all_Ys, missing_indices  = my_read_images(self.generated_frames_train_dir,self.OUT_SIZE,self.OUT_SIZE, expected_number=len(self.all_Xs),extension='jpg',d_type=np.float32,normalize=True)
        if len(self.all_Ys) != len(self.all_Xs):
            print("@@@@@@@@@@@",len(self.all_Xs))
            for ii in missing_indices:
                del self.all_Xs[ii]
        if len(self.all_Ys) != len(self.all_Xs):
            self.all_Ys = self.all_Ys[0:len(self.all_Xs)]
            # raise ValueError("some images were not read properly ... the corrsponding Xs are not correct")

        self.retained_Ys = self.all_Ys.copy()      
        for self.evolve_step in range(self.evolution_nb):
            self.train_bbgan()


    def train_bbgan(self):
        with tf.Graph().as_default() as self.g:
            self.z = tf.placeholder(tf.float32, shape=[None,self.z_dim])
            self.x_ind = tf.placeholder(tf.float32, shape=[None,self.nb_parameters])
            self.oracle_labels = tf.placeholder(tf.float32, shape=[None,self.OUT_SIZE,self.OUT_SIZE, 3])
            self.oracle_scores = tf.placeholder(tf.float32, shape=[None,])
            self.success_rate = tf.to_float(tf.count_nonzero(tf.less(self.oracle_scores,self.SUCCESS_THRESHOLD)))/tf.constant(float(self.valid_size))
            self.score_mean = tf.reshape(tf.nn.moments(self.oracle_scores,axes=0)[0],[])
            self.input_variance = tf.reshape(tf.nn.moments(tf.nn.moments(self.x_ind,axes=0)[1],axes=0)[0],[])
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
                    self.x = generator_ann(self.z,self.nb_parameters)
                    self.transmitter_good = discrminator_ann(self.x_ind,1,network_size=self.network_size)
                    self.transmitter_bad = discrminator_ann(self.x,1,reuse=True,network_size=self.network_size)
     
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

            self.define_metrics()

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
                print("@@@@@@@ START RESTORING")
                # if self.restore_all:
                self.saver.restore(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model-%d" %(self.task_nb)))
                # X_IND =  np.random.uniform(-1, 1, [self.K*self.induced_size,self.nb_parameters])
                X_IND, indx = sample_batch(self.all_Xs, self.K* self.induced_size)
                Y = [self.all_Ys[ii] for ii in indx ]
                X_batches = [X_IND[ii:ii+self.batch_size] for ii in range(0, len(X_IND), self.batch_size)]
                discriminator_Scores = np.concatenate([self.sess.run(self.transmitter_good,feed_dict={self.x_ind:np.array(X_batches[ii])}).reshape(-1) for ii in range(len(X_batches))],axis=0)
                sorted_indices = flip(np.argsort(discriminator_Scores),axis=0).tolist()
                # sorted_indices = np.argsort(discriminator_Scores,axis=0).tolist()
                self.X_bank = self.X_bank + [list(X_IND)[ii] for ii in sorted_indices[:self.induced_size]]
                self.test_targets =  [list(Y)[ii] for ii in sorted_indices[:self.induced_size]]
                self.X_bad =  [list(X_IND)[ii] for ii in sorted_indices[-self.induced_size:]]
                self.test_bad =  [list(Y)[ii] for ii in sorted_indices[-self.induced_size:]]



                # else:
                #     restorer.restore(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model-0"))
            elif self.is_train:
                self.start_time = time.time()
                self.global_step = 0
                for step in range(self.nb_steps):
                    print("step:",step)
                    self.Z = np.random.normal(np.zeros(self.z_dim),1,[self.batch_size,self.z_dim])
  

                    if (step % self.ind_frq == 0):
                        self.inducer_bbgan(induced_size=self.induced_size)
 
                    ## evolutionary step
                    if self.keep_bank or self.full_set:
                        self.X_IND, indx = sample_batch(self.X_bank,self.batch_size)
                        self.Y = [self.Y_bank[ii] for ii in indx]

                    ### training step
                    if not self.optimize_oracle:
                        current_oracle_scores = self.detector_agent(self.Y).reshape(-1)

                    _,current_t_loss = self.sess.run([self.t_optimizer,self.t_loss],
                         feed_dict={self.z:self.Z,self.y:self.Y,self.x_ind:self.X_IND,self.oracle_scores:current_oracle_scores})

                    _,current_g_loss = self.sess.run([self.g_optimizer,self.g_loss],
                         feed_dict={self.z:self.Z,self.y:self.Y,self.oracle_scores:np.array(current_oracle_scores)})
                    _,current_g_loss,new_summary = self.sess.run([self.g_optimizer,self.g_loss,self.total_loss_summary],
                         feed_dict={self.z:self.Z,self.y:self.Y,self.x_ind:self.X_IND,self.oracle_scores:current_oracle_scores})
                    self.writer.add_summary(new_summary,self.global_step)

                    self.logger.write("step number: %2d , train t_Loss: %2.5f train g_Loss: %2.5f \n" %(self.global_step,current_t_loss,current_g_loss))

                    self.global_step += 1
                    if ((step+1) % self.log_frq == 0):
                        self.validating_bbgan(valid_size=self.valid_size)
                        if self.is_visualize:
                            self.visualize_bbgan(step=step)
                        # self.saver.save(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model"), global_step=self.global_step, write_meta_graph=False)
                        self.saver.save(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model"), global_step=self.task_nb, write_meta_graph=False)

                        self.logger.write("\n\nepoch: %d validation: %2.5f   standard: %2.5f   improvmetn: %2.5f \n\n" %(step,self.avg_loss,self.avg_stdloss,self.avg_loss-self.avg_stdloss))
            else : 
                self.saver.restore(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model-%d" %(self.task_nb)))
                self.validating_bbgan(valid_size=self.valid_size)
                if self.is_visualize:
                    self.visualize_bbgan()
            # print("\n\n\n validation loss : ", avg_loss)
            self.validating_bbgan(valid_size=self.valid_size)
            if self.is_visualize:
                self.visualize_bbgan(step=0)
            self.saver.save(self.sess,save_path=os.path.join(self.checkpoint_path,"oracle-model"), global_step=self.task_nb, write_meta_graph=False)            
            self.register_metrics()

            if self.is_evolve:
                retained_X , inxs = sample_batch(self.all_Xs,self.retained_size) ; retained_Y =  [self.all_Ys[ii] for ii in inxs ]
                self.all_Xs = self.X_bank + self.K * self.test_X.tolist() + retained_X
                self.all_Ys = self.Y_bank + self.K * self.test_targets + retained_Y
            return 

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
            if  self.exp_type == "Gaussian" or self.exp_type == "GP":
                self.X_bank = self.X_bank + [X_IND[ii] for ii in sorted_indices[:induced_size]]
                self.Y_bank = self.Y_bank + [Y[ii] for ii in sorted_indices[:induced_size]]
            else :
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
        if self.exp_type == "Adversarial":
            if self.is_train:
                test_Z = np.random.normal(np.zeros(self.z_dim),1,[valid_size,self.z_dim])
                self.test_X = self.x.eval(feed_dict={self.z:test_Z},session=self.sess)
            if self.cont_train:
                # self.test_bad = black_box_batch(self.X_bad,output_size=self.OUT_SIZE,global_step=0,frames_path=self.frames_path,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)
                self.all_bad_prob = function_batches(self.detector_agent,self.test_bad,50)
                self.all_bad_prob = np.vstack(np.array(self.all_bad_prob)).reshape(-1)
                # self.all_bad_prob =  self.detector_agent(np.array(self.test_bad)).flatten().tolist()
                self.test_X = np.array(self.X_bank) 
        elif self.exp_type == "Gaussian" or self.exp_type == "Baysian" :
            self.test_X = sample_from_learned_gaussian(self.X_bank, n_components=self.gaussian_nb , n_samples=valid_size)
            # self.test_X = 10*self.X_bank #####################################

        elif self.exp_type == "GP":
            self.test_X = self.X_bank.copy()
            # self.test_targets = self.Y_bank.copy() #######################


        if not self.cont_train: # and  self.exp_type != "GP" : ################
            self.test_targets = black_box_batch(self.test_X,output_size=self.OUT_SIZE,global_step=self.task_nb,frames_path=self.frames_path,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)
        # self.test_prob = self.detector_agent(np.array(self.test_targets))
        self.test_prob = function_batches(self.detector_agent,self.test_targets,50)
        self.test_prob = np.vstack(np.array(self.test_prob)).reshape(-1)


        if not self.is_evolve:
            self.test_std,indx = sample_batch(self.all_Ys,self.valid_size)
            self.test_stdX = [self.all_Xs[ii] for ii in indx ]
            # if self.exp_type == "GP":
            #     self.test_std = self.all_Ys.copy()
            #     self.test_stdX = self.all_Xs.copy()
        else :
            self.test_std,indx = sample_batch(self.retained_Ys,self.valid_size)
            self.test_stdX = [self.all_Xs[ii] for ii in indx ]


        # self.test_stdprob = self.detector_agent(np.array(self.test_std))
        self.test_stdprob = function_batches(self.detector_agent,self.test_std,50)
        self.test_stdprob = np.vstack(np.array(self.test_stdprob)).reshape(-1)
        # all_losses.append(loss)
        self.all_prob =  self.test_prob.flatten().tolist()
        self.all_stdprob = self.test_stdprob.flatten().tolist()
        self.avg_loss = np.mean(self.test_prob.flatten())
        self.avg_stdloss = np.mean(self.test_stdprob.flatten())


        return 
        


    def visualize_bbgan(self,step=100):
        # for ii , img in enumerate(sorted_validation):
        #   scipy.misc.imsave(os.path.join(self.generated_frames_test_dir,"s_%d.jpg"%(ii)),inverse_transform(img))
        imageio.mimsave(os.path.join(self.generated_frames_test_dir,"s_%d_%d.gif"%(step,self.evolve_step)),[inverse_transform(img) for img in self.test_targets])
        save_image(np.array([inverse_transform(img) for img in sample_batch(self.test_targets,16)[0]]), os.path.join(self.generated_frames_test_dir,"s_%d_%d.png"%(step,self.evolve_step)),
         nrow=4, padding=2,normalize=False, scale_each=False)
        imsave(np.array([inverse_transform(img) for img in sample_batch(self.test_targets,16)[0]]),size=self.OUT_SIZE,path=os.path.join(self.generated_frames_test_dir,"s_%d_%d.png"%(step,self.evolve_step)),is_all=True)
        imsave(np.array([inverse_transform(img) for img in sample_batch(self.test_std,16)[0]]),size=self.OUT_SIZE,path=os.path.join(self.generated_frames_test_dir,"r_%d_%d.png"%(step,self.evolve_step)),is_all=True)
            # imageio.mimsave(os.path.join(self.generated_frames_test_dir,"r_%d.gif"%(epoch)),[inverse_transform(img) for img in self.test_std])
        # if self.is_cluster:
        plt.figure(figsize = (8, 6))
        plt.hist([self.all_prob,self.all_stdprob],color=["b","r"], label=["BBGAN","Random"], bins=100, range=(0.0,1.0))
        plt.legend()
        plt.savefig(os.path.join(self.generated_frames_test_dir,"histogram_%d_%d.jpg"%(step,self.evolve_step)))
        plt.close()

        if self.cont_train:
            plt.figure(figsize = (8, 6))
            plt.hist([self.all_prob,self.all_stdprob,self.all_bad_prob],color=["b","r",'g'], label=["good_discrminator","Random","bad_discrminator"], bins=100, range=(0.0,1.0))
            plt.legend()
            plt.savefig(os.path.join(self.generated_frames_test_dir,"histogram_%d_%d.jpg"%(step,self.evolve_step)))
            plt.close()

        
        plt.figure(figsize = (8, 6))
        for ii in range(self.nb_parameters):
            sns.kdeplot(np.array(self.test_X)[:,ii].tolist(), linewidth = 2, shade = False, label=self.paramters_list[ii],clip=(-1,1))
        plt.legend()
        plt.xlim(-1,1)    
        plt.savefig(os.path.join(self.generated_frames_test_dir,"parmeters_%d_%d.jpg"%(step,self.evolve_step)))
        plt.close()

        sphere_params = OrderedDict()
        for param in range(self.nb_parameters):
            sphere_params[self.paramters_list[param]] = np.array(self.test_X)[:,param].tolist()
        sphere_df = pd.DataFrame(sphere_params)
        sphere_df.to_csv(os.path.join(self.generated_frames_test_dir,'test_params.csv'),sep=',',index=False)

        return

    def learn_gaussian(self):
        self.evolve_step=0
        with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'rb') as fp:
            saved_dict = cPickle.load(fp)
        self.all_Xs = saved_dict["x"] 
        self.all_Ys, missing_indices  = my_read_images(self.generated_frames_train_dir,self.OUT_SIZE,self.OUT_SIZE,expected_number=len(self.all_Xs),extension="jpg",d_type=np.float32,normalize=True)
        if len(self.all_Ys) != len(self.all_Xs):
            print("@@@@@@@@@@@",len(self.all_Xs))
            for ii in missing_indices:
                del self.all_Xs[ii]
        if len(self.all_Ys) != len(self.all_Xs):
            self.all_Ys = self.all_Ys[0:len(self.all_Xs)]
            # raise ValueError("some images were not read properly ... the corrsponding Xs are not correct")
        self.retained_Ys = self.all_Ys.copy() 

        with tf.Graph().as_default() as self.g:
            self.z = tf.placeholder(tf.float32, shape=[None,self.z_dim])
            self.x_ind = tf.placeholder(tf.float32, shape=[None,self.nb_parameters])
            self.oracle_labels = tf.placeholder(tf.float32, shape=[None,self.OUT_SIZE,self.OUT_SIZE, 3])
            self.oracle_scores = tf.placeholder(tf.float32, shape=[None,])
            self.success_rate = tf.to_float(tf.count_nonzero(tf.less(self.oracle_scores,self.SUCCESS_THRESHOLD)))/tf.constant(float(self.valid_size))
            self.score_mean = tf.reshape(tf.nn.moments(self.oracle_scores,axes=0)[0],[])
            self.input_variance = tf.reshape(tf.nn.moments(tf.nn.moments(self.x_ind,axes=0)[1],axes=0)[0],[])
            self.focal_weights =self.oracle_scores ** self.gamma
            self.focal_weights_avg = tf.reduce_mean(self.focal_weights)
            self.y = tf.placeholder(tf.float32, [None,self.OUT_SIZE,self.OUT_SIZE, 3])

            with tf.device('/GPU:0'):
                with tf.variable_scope('detector'):
                    detections = yolo_v3(self.y, len(self.coco_classes), data_format='NHWC')
                    load_ops = load_weights(tf.global_variables(scope='detector'), self.weights_file)

                self.boxes = detections_boxes(detections)

            self.define_metrics()
        with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
            self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
            tf.global_variables_initializer().run()
            self.sess.run(load_ops)
            if self.is_train:
                self.start_time = time.time()
                self.inducer_bbgan(induced_size=self.induced_size)
                print("start learning mixture of gaussians")
                self.validating_bbgan(valid_size=self.valid_size)
                self.register_metrics()
                if self.is_visualize:
                    self.visualize_bbgan(step=0)

    def learn_gp(self):
        self.evolve_step=0
        with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'rb') as fp:
            saved_dict = cPickle.load(fp)
        self.all_Xs = saved_dict["x"] 
        self.all_Ys, missing_indices  = my_read_images(self.generated_frames_train_dir,self.OUT_SIZE,self.OUT_SIZE,expected_number=len(self.all_Xs),extension="jpg",d_type=np.float32,normalize=True)
        if len(self.all_Ys) != len(self.all_Xs):
            print("@@@@@@@@@@@",len(self.all_Xs))
            for ii in missing_indices:
                del self.all_Xs[ii]
        if len(self.all_Ys) != len(self.all_Xs):
            self.all_Ys = self.all_Ys[0:len(self.all_Xs)]
            # raise ValueError("some images were not read properly ... the corrsponding Xs are not correct")
        self.retained_Ys = self.all_Ys.copy() 

        with tf.Graph().as_default() as self.g:
            self.z = tf.placeholder(tf.float32, shape=[None,self.z_dim])
            self.x_ind = tf.placeholder(tf.float32, shape=[None,self.nb_parameters])
            self.oracle_labels = tf.placeholder(tf.float32, shape=[None,self.OUT_SIZE,self.OUT_SIZE, 3])
            self.oracle_scores = tf.placeholder(tf.float32, shape=[None,])
            self.success_rate = tf.to_float(tf.count_nonzero(tf.less(self.oracle_scores,self.SUCCESS_THRESHOLD)))/tf.constant(float(self.valid_size))
            self.score_mean = tf.reshape(tf.nn.moments(self.oracle_scores,axes=0)[0],[])
            self.input_variance = tf.reshape(tf.nn.moments(tf.nn.moments(self.x_ind,axes=0)[1],axes=0)[0],[])
            self.focal_weights =self.oracle_scores ** self.gamma
            self.focal_weights_avg = tf.reduce_mean(self.focal_weights)
            self.y = tf.placeholder(tf.float32, [None,self.OUT_SIZE,self.OUT_SIZE, 3])

            with tf.device('/GPU:0'):
                with tf.variable_scope('detector'):
                    detections = yolo_v3(self.y, len(self.coco_classes), data_format='NHWC')
                    load_ops = load_weights(tf.global_variables(scope='detector'), self.weights_file)

                self.boxes = detections_boxes(detections)

            self.define_metrics()
        with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
            self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
            tf.global_variables_initializer().run()
            self.sess.run(load_ops)
            if self.is_train:
                self.start_time = time.time()
                self.inducer_bbgan(induced_size=self.induced_size)
                print("start learning GP regression")
                self.validating_bbgan(valid_size=self.valid_size)
                self.register_metrics()

                sphere_params = OrderedDict()
                for param in range(self.nb_parameters):
                    sphere_params[self.paramters_list[param]] = np.array(self.all_Xs)[:,param].tolist()
                sphere_params['score'] = self.test_stdprob.tolist()
                sphere_df = pd.DataFrame(sphere_params)
                sphere_df.to_csv(os.path.join(self.generated_path,"gp_params","class_{}.csv".format(str(self.class_nb))),sep=',',index=False)
            
            else:
                # sphere_params = pd.read_csv(os.path.join(self.generated_path,"gp_params","{}_gp_regression_best.csv".format(str(self.class_nb).rjust(2, '0'))))
                # sphere_params = pd.read_csv(os.path.join(self.generated_path,"gp_params","{}_svm_multi_best.csv".format(str(self.class_nb).rjust(2, '0'))))
                sphere_params = pd.read_csv(os.path.join(self.generated_path,"scenario_params","test_params_{}.csv".format(str(self.task_nb))))
                mid_list = np.array([list(sphere_params[self.paramters_list[param]]) for param in range(self.nb_parameters) ]).T
                self.X_bank = [mid_list[ii,:] for ii in range(mid_list.shape[0])]
                self.start_time = time.time()
                print("start learning GP regression")
                self.validating_bbgan(valid_size=self.valid_size)
                self.register_metrics()





                # if self.is_visualize:
                #     self.visualize_bbgan(step=0)


    def learn_baysian(self):
        from hyperopt import hp
        from hyperopt.pyll.stochastic import sample
        from hyperopt import rand, tpe
        from hyperopt import Trials
        from hyperopt import fmin
        from hyperopt import STATUS_OK
        tpe_trials = Trials()
        tpe_algo = tpe.suggest
        self.all_Ys = []
        if not self.is_train:
            self.all_Ys, missing_indices  = my_read_images(self.generated_frames_train_dir,self.OUT_SIZE,self.OUT_SIZE, extension='jpg',d_type=np.float32,normalize=True)
        self.all_Xs = []
        self.all_loss = []
        vars_list  = ["x"+str(ii) for ii in range(self.nb_parameters)]
        space = {}
        for keys in vars_list:
            space[keys] = hp.uniform(keys, -1, 1)

        with tf.Graph().as_default() as self.g:
            self.z = tf.placeholder(tf.float32, shape=[None,self.z_dim])
            self.x_ind = tf.placeholder(tf.float32, shape=[None,self.nb_parameters])
            self.oracle_labels = tf.placeholder(tf.float32, shape=[None,self.OUT_SIZE,self.OUT_SIZE, 3])
            self.oracle_scores = tf.placeholder(tf.float32, shape=[None,])
            self.success_rate = tf.to_float(tf.count_nonzero(tf.less(self.oracle_scores,self.SUCCESS_THRESHOLD)))/tf.constant(float(self.valid_size))
            self.score_mean = tf.reshape(tf.nn.moments(self.oracle_scores,axes=0)[0],[])
            self.input_variance = tf.reshape(tf.nn.moments(tf.nn.moments(self.x_ind,axes=0)[1],axes=0)[0],[])
            self.focal_weights =self.oracle_scores ** self.gamma
            self.focal_weights_avg = tf.reduce_mean(self.focal_weights)
            self.y = tf.placeholder(tf.float32, [None,self.OUT_SIZE,self.OUT_SIZE, 3])

            with tf.device('/GPU:0'):
                with tf.variable_scope('detector'):
                    detections = yolo_v3(self.y, len(self.coco_classes), data_format='NHWC')
                    load_ops = load_weights(tf.global_variables(scope='detector'), self.weights_file)

                self.boxes = detections_boxes(detections)

            self.define_metrics()
        with tf.Session(graph=self.g,config=tf.ConfigProto(gpu_options=self.gpu_options)) as self.sess:
            self.writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
            tf.global_variables_initializer().run()
            self.sess.run(load_ops)
            self.start_time = time.time()
            global ITERATION
            ITERATION = 0


            def objective(xs):
                ########### @@@@@@@@@@@@@@@@@ play with the uinput to make it vector !! 
                global ITERATION
                ITERATION += 1
                keylist = xs.keys()
                list(keylist).sort()
                x = [xs[ii] for ii in keylist]
                print("@@@@@@@@@@@@@","ITERATION : ", ITERATION)
                # raise Exception
                try:
                    y = black_box(x,output_size=self.OUT_SIZE,global_step=self.task_nb,frames_path=self.frames_path,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)
                except:
                    try:
                        y = black_box(x,output_size=self.OUT_SIZE,global_step=self.task_nb,frames_path=self.frames_path,cluster=self.is_cluster,parent_name=self.pascal_list[self.class_nb],scenario_nb=self.scenario_nb)
                    except:
                        y = self.all_Ys[-1]
                self.all_Ys.append(y)
                self.all_Xs.append(x)
                loss = np.squeeze(self.detector_agent(np.expand_dims(y, axis=0)))
                self.all_loss.append(loss)
                if ((ITERATION) % self.log_frq == 0):
                    tpe_results = pd.DataFrame({'loss': self.all_loss, 'iteration': range(ITERATION),'x':self.all_Xs })
                    tpe_results.to_csv(os.path.join(self.generated_frames_train_dir,'baysian.csv'),sep=',',index=False)
                return {'loss': loss, 'xs': xs, 'iteration': ITERATION,'status': STATUS_OK}


            if self.is_train:
                tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_trials,max_evals=self.gendist_size, rstate= np.random.RandomState(50))
                print('Minimum loss attained with TPE:    {:.4f}'.format(tpe_trials.best_trial['result']['loss']))
                self.all_Xs = [[vars_dics[keys] for keys in vars_list]for vars_dics in [x['xs'] for x in tpe_trials.results]]
                tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results], 'iteration': [x['iteration'] for x in tpe_trials.results],'x':self.all_Xs })
                tpe_results.to_csv(os.path.join(self.generated_frames_train_dir,'baysian.csv'),sep=',',index=False)
                with open(os.path.join(self.generated_frames_train_dir,"baysian.pkl"),'wb') as fp:
                    cPickle.dump(tpe_results.to_dict(),fp)


            self.evolve_step=0
            try:
                with open(os.path.join(self.generated_frames_train_dir,"baysian.pkl"),'rb') as fp:
                    tpe_results = pd.DataFrame(cPickle.load(fp))
                self.all_Xs = list(tpe_results["x"]) 
            except:
                tpe_results = pd.read_csv(os.path.join(self.generated_frames_train_dir,'baysian.csv'))
                self.all_Xs = [string_to_float_list(a) for a in list(tpe_results["x"]) ]

            self.X_bank = self.X_bank + self.all_Xs[-self.induced_size:]
            self.Y_bank = self.Y_bank + self.all_Ys[-self.induced_size:]

            if not self.is_train:
                with open(os.path.join(self.generated_frames_train_dir,"save.pkl"),'rb') as fp:
                    saved_dict = cPickle.load(fp)
                self.all_Xs = saved_dict["x"] 
            print("start learning mixture of gaussians for the baysian")
            self.validating_bbgan(valid_size=self.valid_size)
            self.register_metrics()
            if self.is_visualize:
                self.visualize_bbgan(step=0)




    def define_metrics(self):
        if self.exp_type == "Adversarial":
            g_loss_summary= tf.summary.scalar('losses/G_loss', self.g_loss)
            t_loss_real_summary= tf.summary.scalar('losses/t_loss_good', self.t_loss_good)
            t_loss_bad_summary= tf.summary.scalar('losses/t_loss_bad', self.t_loss_bad)
            t_loss_summary= tf.summary.scalar('losses/t_loss', self.t_loss)
            self.total_loss_summary = tf.summary.merge([g_loss_summary,t_loss_summary,t_loss_real_summary,t_loss_bad_summary]) # 

        score_summary = tf.summary.scalar('metric/score_mean', self.score_mean)
        std_score_summary = tf.summary.scalar('metric/std_score_mean', self.score_mean)
        success_summary = tf.summary.scalar('metric/success_rate', self.success_rate)
        std_success_summary = tf.summary.scalar('metric/std_success_rate', self.success_rate)
        var_summary = tf.summary.scalar('metric/input_variance', self.input_variance)
        std_var_summary = tf.summary.scalar('metric/std_input_variance', self.input_variance)
        self.total_config_summary = tf.summary.merge([tf.summary.scalar('config/{}'.format(k),v) for k ,v in self.config_dict.items() ]) # 
        self.total_metric_summary = tf.summary.merge([score_summary,var_summary,success_summary]) # 
        self.std_total_metric_summary = tf.summary.merge([std_score_summary,std_var_summary,std_success_summary]) # 


    def register_metrics(self):
        self.writer.add_summary(self.total_metric_summary.eval(feed_dict={self.oracle_scores:self.test_prob,self.x_ind:self.test_X},session=self.sess),0)    
        self.writer.flush()
        self.writer.add_summary(self.total_config_summary.eval(feed_dict=None,session=self.sess),0)
        self.writer.flush()
        self.writer.add_summary(self.std_total_metric_summary.eval(feed_dict={self.oracle_scores:self.test_stdprob,self.x_ind:self.test_stdX},session=self.sess),0)
        self.writer.flush()


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

