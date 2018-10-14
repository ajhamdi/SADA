# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from adversarial_Blackbox import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('exp_type', 'Adversarial', 'the experiment type')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('dataset_nb', 5, 'the dataset ID number used  ')
tf.app.flags.DEFINE_integer('exp_no', 0, 'the exp number used ')
tf.app.flags.DEFINE_integer('nb_paramters', 8, 'the number of paramters learnt by BBGNA ')
tf.app.flags.DEFINE_integer('class_nb', 0, 'the exp number used ')
tf.app.flags.DEFINE_integer('task_nb', 0, 'the exp number used ')
tf.app.flags.DEFINE_integer('evolution_nb', 1, 'the number of evolutionary steps ')
tf.app.flags.DEFINE_integer('valid_size', 20, 'the size of the validation set for evaluation and visualization')
tf.app.flags.DEFINE_integer('log_frq', 40, 'the grequency of logging ')
tf.app.flags.DEFINE_integer('batch_size', 64, 'the size of batch in training ')
tf.app.flags.DEFINE_integer('K', 10, 'the degree of pickiness ')
tf.app.flags.DEFINE_integer('induced_size', 50, 'the size of the indiced set to be added to training bank ')
tf.app.flags.DEFINE_integer('retained_size', 5000,'the size of the partial retained set to be used in the evolved GAN if evolve is choice is used ')
tf.app.flags.DEFINE_integer('ind_frq', 2, 'the frequency of the indcing ')
tf.app.flags.DEFINE_integer('nb_steps', 600, 'the number of training steps ')
tf.app.flags.DEFINE_integer('gendist_size', 10000, 'the size of the generated distribution ')

tf.app.flags.DEFINE_boolean('is_train',True," training mode")
tf.app.flags.DEFINE_boolean('is_selfdrive',False," trainign selfdriving agent")
tf.app.flags.DEFINE_boolean('is_gendist',False," generate distribution mode")
tf.app.flags.DEFINE_boolean('is_cluster',False," is it running in a cluster ?")
tf.app.flags.DEFINE_boolean('cont_train',False," continue training trained model")
tf.app.flags.DEFINE_boolean('optimize_oracle',False," optimimze the main AI agent")
tf.app.flags.DEFINE_boolean('restore_all',False," restore both the agent AND the gan")
tf.app.flags.DEFINE_boolean('is_focal',False," use focal loss in the BBGAN")
tf.app.flags.DEFINE_boolean('is_evolve',False," use the output of BBGAN to improve later stage")
tf.app.flags.DEFINE_boolean('keep_bank',False," use partial set iteratively to update the training bank")
tf.app.flags.DEFINE_boolean('full_set',True," use the full set once in the training")

tf.app.flags.DEFINE_float('learning_rate_t',0.0002, 'the adam learning rate of the D')
tf.app.flags.DEFINE_float('learning_rate_g',0.0002, 'the adam learning rate of the G ')
tf.app.flags.DEFINE_float('gan_init_variance',0.06 , 'gan inital variance for the D and G')
    

def main(argv=None):
    # AVAILABLE_Exps = ["Adversarial"]
    # exp_type = AVAILABLE_Exps[0]
    # epochs_list = [2,4,6,8]
    # real_list = [0,1,2]
    base_path = os.getcwd()
    exp_no = 3
    if FLAGS.is_cluster:
        if FLAGS.is_train:
            FLAGS.exp_no += FLAGS.task_nb
        elif FLAGS.is_gendist:
            FLAGS.class_nb = FLAGS.task_nb
            FLAGS.dataset_nb = FLAGS.dataset_nb + FLAGS.task_nb 
    # for epoch in epochs_list:
    # for real_no in real_list:
    # for i in range(3):
    print(FLAGS.flag_values_dict())
    raise Exception 
    bbexp = BlackBoxOptimizer(FLAGS = FLAGS,base_path=base_path)

    if FLAGS.is_train:
        if not FLAGS.is_selfdrive:
            bbexp.learn_bbgan()
        else :
            bbexp.learn_selfdrive()
     
        # exp_no = exp_no + 1 
        # del bbexp

    elif FLAGS.is_gendist:
        bbexp.generate_distribution()

    
    # bbexp.learn_distribution_random()
    # bbexp.learn_distribution_gan()

if __name__ == '__main__':
    tf.app.run()
