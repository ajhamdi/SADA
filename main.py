# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from adversarial_Blackbox import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('exp_type', 'Adversarial', 'the experiment type')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('scenario_nb', 0, 'the scenario of training , data generation to follow  ')
tf.app.flags.DEFINE_integer('dataset_nb', 2, 'the dataset ID number used  ')
tf.app.flags.DEFINE_integer('exp_no', 0, 'the exp number used ')
tf.app.flags.DEFINE_integer('nb_parameters', 8, 'the number of paramters learnt by BBGNA ')
tf.app.flags.DEFINE_integer('class_nb', 0, 'the exp number used ')
tf.app.flags.DEFINE_integer('task_nb', 0, 'the exp number used ')
tf.app.flags.DEFINE_integer('evolution_nb', 1, 'the number of evolutionary steps ')
tf.app.flags.DEFINE_integer('gaussian_nb', 1, 'the number of mixtures of gaussian learnt in Gaussian exp ')
tf.app.flags.DEFINE_integer('valid_size', 250, 'the size of the validation set for evaluation and visualization')
tf.app.flags.DEFINE_integer('log_frq', 40, 'the grequency of logging ')
tf.app.flags.DEFINE_integer('batch_size', 64, 'the size of batch in training ')
tf.app.flags.DEFINE_integer('K', 10, 'the degree of pickiness ')
tf.app.flags.DEFINE_integer('z_dim', 17, 'the size of the z latent vector ')
tf.app.flags.DEFINE_integer('network_size', 2, 'the size of hidden layers of the G and D networks')
tf.app.flags.DEFINE_integer('induced_size', 1000, 'the size of the indiced set to be added to training bank ')
tf.app.flags.DEFINE_integer('retained_size', 5000,'the size of the partial retained set to be used in the evolved GAN if evolve is choice is used ')
tf.app.flags.DEFINE_integer('ind_frq', 2, 'the frequency of the indcing ')
tf.app.flags.DEFINE_integer('nb_steps', 500, 'the number of training steps ')
tf.app.flags.DEFINE_integer('gendist_size', 10000, 'the size of the generated distribution ')

tf.app.flags.DEFINE_boolean('is_train',True," training mode")
tf.app.flags.DEFINE_boolean('is_selfdrive',False," trainign selfdriving agent")
tf.app.flags.DEFINE_boolean('is_gendist',False," generate distribution mode")
tf.app.flags.DEFINE_boolean('is_visualize',True," visualize scores and images of that training batch")
tf.app.flags.DEFINE_boolean('is_cluster',False," is it running in a cluster ?")
tf.app.flags.DEFINE_boolean('is_randomize',False," randomize the conifguration of the exps")
tf.app.flags.DEFINE_boolean('is_varsteps',False," use adaptive steps obtained previously to each scenario")
tf.app.flags.DEFINE_boolean('is_gaussian',False," is it running gaussian baseline as well ?")
tf.app.flags.DEFINE_boolean('is_baysian',False," is it running baysian baseline as well ?")
tf.app.flags.DEFINE_boolean('is_focal',False," is it focal loss ?")
tf.app.flags.DEFINE_boolean('cont_train',False," continue training trained model")
tf.app.flags.DEFINE_boolean('optimize_oracle',False," optimimze the main AI agent")
tf.app.flags.DEFINE_boolean('restore_all',False," restore both the agent AND the gan")
tf.app.flags.DEFINE_boolean('is_evolve',False," use the output of BBGAN to improve later stage")
tf.app.flags.DEFINE_boolean('keep_bank',False," use partial set iteratively to update the training bank")
tf.app.flags.DEFINE_boolean('full_set',True," use the full set once in the training")

tf.app.flags.DEFINE_float('learning_rate_t',0.0003, 'the adam learning rate of the D')
tf.app.flags.DEFINE_float('learning_rate_g',0.0003, 'the adam learning rate of the G ')
tf.app.flags.DEFINE_float('gan_init_variance',0.053 , 'gan inital variance for the D and G')
    
def randomize_setup(flags):
    exp_type_list = ["Adversarial","Gaussian","Baysian","Finite","Neural"]
    scenario_nb_list = list(range(1,10))
    class_nb_list = list(range(11))
    gaussian_nb_list = list(range(1,flags.valid_size))
    evolution_nb_list = [1,2,3]
    K_list = list(range(2,20))
    z_dim_list = list(range(1,20))
    network_size_list = list(range(0,5))
    induced_size_list = list(range(200,1000))
    nb_steps_list = list(range(400,620))
    variamce_list = list(np.linspace(0.05,0.07,20))
    learning_rate_list  = list(np.linspace(0.0002,0.0005,20))

    # flags.exp_type = np.random.choice(exp_type_list)
    # flags.class_nb = np.random.choice(class_nb_list)
    # flags.gaussian_nb = np.random.choice(gaussian_nb_list)
    # flags.evolution_nb  = np.random.choice(evolution_nb_list)
    # flags.K = np.random.choice(K_list)
    # flags.z_dim = np.random.choice(z_dim_list)
    # flags.network_size = np.random.choice(network_size_list)
    # flags.induced_size = np.random.choice(induced_size_list)
    flags.nb_steps = np.random.choice(nb_steps_list)
    # flags.gan_init_variance = np.random.choice(variamce_list)
    flags.learning_rate_g = np.random.choice(learning_rate_list)
    flags.learning_rate_t = flags.learning_rate_g

def main(argv=None):
    base_path = os.getcwd()
    if FLAGS.is_cluster and FLAGS.is_randomize :
        randomize_setup(flags=FLAGS)
        # raise Exception("###############")
    if FLAGS.is_cluster:
        pass

    bbexp = BlackBoxOptimizer(FLAGS = FLAGS,base_path=base_path)
    
    if FLAGS.is_gendist:
            bbexp.generate_distribution()
    elif FLAGS.exp_type == "Adversarial":
        if FLAGS.is_train or FLAGS.cont_train:
            if not FLAGS.is_selfdrive:
                bbexp.learn_bbgan()
            else :
                bbexp.learn_selfdrive()
    elif FLAGS.exp_type == "Gaussian":
        FLAGS.is_gaussian = True
        bbexp.learn_gaussian()

    elif FLAGS.exp_type == "Baysian":
        FLAGS.is_baysian = True
        bbexp.learn_baysian()
    
    # bbexp.learn_distribution_random()
    # bbexp.learn_distribution_gan()

if __name__ == '__main__':
    tf.app.run()
