import numpy as np
import os
from ops import *
import tensorflow as tf
slim = tf.contrib.slim


def discrminator_ann(x, output_size, reuse=False, network_size=3):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        output = slim.fully_connected(x, 10, scope='objective/fc_1')
        for ii in range(2, 2+network_size):
            output = slim.fully_connected(
                output, 10, scope='objective/fc_%d' % (ii))
        output = slim.fully_connected(
            output, output_size, activation_fn=None, scope='objective/fc_%d' % (2+network_size))
    return output


def generator_ann(x, output_size, min_bound=-1, max_bound=1, network_size=3):
    range_required = np.absolute(max_bound - min_bound).astype(np.float64)
    with tf.variable_scope("generator") as scope:
        output = slim.fully_connected(x, 10, scope='objective/fc_1')
        for ii in range(2, 2+network_size):
            output = slim.fully_connected(
                output, 10, scope='objective/fc_%d' % (ii))
        output = slim.fully_connected(
            output, output_size, activation_fn=None, scope='objective/fc_%d' % (2+network_size))
        # contrained_output =   range_required * tf.nn.sigmoid(output) + min_bound* tf.ones_like(output)
        contrained_output = tf.nn.tanh(output)
    return contrained_output



def int_shape(tensor):
	shape = tensor.get_shape().as_list()
	return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
	shape = int_shape(tensor)
	# always return [N, H, W, C]
	if data_format == 'NCHW':
		return [shape[0], shape[2], shape[3], shape[1]]
	elif data_format == 'NHWC':
		return shape

def nchw_to_nhwc(x):
	return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
	return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
	if data_format == 'NCHW':
		x = tf.reshape(x, [-1, c, h, w])
	else:
		x = tf.reshape(x, [-1, h, w, c])
	return x

def resize_nearest_neighbor(x, new_size, data_format):
	if data_format == 'NCHW':
		x = nchw_to_nhwc(x)
		x = tf.image.resize_nearest_neighbor(x, new_size)
		x = nhwc_to_nchw(x)
	else:
		x = tf.image.resize_nearest_neighbor(x, new_size)
	return x

def upscale(x, scale, data_format):
	_, h, w, _ = get_conv_shape(x, data_format)
	return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):

	'''From https://github.com/ethereon/caffe-tensorflow
	'''
	c_i = input.get_shape()[-1]
	assert c_i%group==0
	assert c_o%group==0
	convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	
	
	if group==1:
		conv = convolve(input, kernel)
	else:
		input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
		kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
		output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
	return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
