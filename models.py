import numpy as np
import os
from ops import *
import tensorflow as tf
slim = tf.contrib.slim

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
	with tf.variable_scope("G", reuse=reuse) as vs:
		num_output = int(np.prod([8, 8, hidden_num]))
		x = slim.fully_connected(z, num_output, activation_fn=None)
		x = reshape(x, 8, 8, hidden_num, data_format)
		
		for idx in range(repeat_num):
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			if idx < repeat_num - 1:
				x = upscale(x, 2, data_format)

		out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

	variables = tf.contrib.framework.get_variables(vs)
	return out, variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
	with tf.variable_scope("D") as vs:
		# Encoder
		x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

		prev_channel_num = hidden_num
		for idx in range(repeat_num):
			channel_num = hidden_num * (idx + 1)
			x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			if idx < repeat_num - 1:
				x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
				#x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

		x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
		z = x = slim.fully_connected(x, z_num, activation_fn=None)

		# Decoder
		num_output = int(np.prod([8, 8, hidden_num]))
		x = slim.fully_connected(x, num_output, activation_fn=None)
		x = reshape(x, 8, 8, hidden_num, data_format)
		
		for idx in range(repeat_num):
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			if idx < repeat_num - 1:
				x = upscale(x, 2, data_format)

		out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

	variables = tf.contrib.framework.get_variables(vs)
	return out, z, variables

def DiscriminatorCNN_encoder(x,conv_input_size, output_size, repeat_num , data_format):
	with tf.variable_scope("D") as vs:
		# Encoder
		x = slim.conv2d(x, conv_input_size, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

		prev_channel_num = conv_input_size
		for idx in range(repeat_num):
			channel_num = conv_input_size * (idx + 1)
			x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
			if idx < repeat_num - 1:
				x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
				#x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

		x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
		z = x = slim.fully_connected(x, output_size, activation_fn=None)
		return z

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
def comparitor(xxx,output_height):
	with tf.variable_scope("comparitor") as scope:
		if output_height != 227 :
			xx = tf.image.resize_images(xxx,[227,227])
		r, g, b = tf.split(xx, 3, axis=3)
		x = tf.concat([r, b, g], axis=3)
		k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
		# net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
		net_data = np.load(open(os.path.join(os.getcwd(),"bvlc_alexnet.npy"), "rb"), encoding="latin1").item()

		conv1W = tf.constant(net_data["conv1"][0])
		conv1b = tf.constant(net_data["conv1"][1])
		conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
		conv1 = tf.nn.relu(conv1_in)

		#lrn1
		#lrn(2, 2e-05, 0.75, name='norm1')
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn1 = tf.nn.local_response_normalization(conv1,
															depth_radius=radius,
															alpha=alpha,
															beta=beta,
															bias=bias)

		#maxpool1
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


		#conv2
		#conv(5, 5, 256, 1, 1, group=2, name='conv2')
		k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv2W = tf.constant(net_data["conv2"][0])
		conv2b = tf.constant(net_data["conv2"][1])
		conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv2 = tf.nn.relu(conv2_in)


		#lrn2
		#lrn(2, 2e-05, 0.75, name='norm2')
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn2 = tf.nn.local_response_normalization(conv2,
															depth_radius=radius,
															alpha=alpha,
															beta=beta,
															bias=bias)

		#maxpool2
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

		#conv3
		#conv(3, 3, 384, 1, 1, name='conv3')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
		conv3W = tf.constant(net_data["conv3"][0])
		conv3b = tf.constant(net_data["conv3"][1])
		conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv3 = tf.nn.relu(conv3_in)

		#conv4
		#conv(3, 3, 384, 1, 1, group=2, name='conv4')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
		conv4W = tf.constant(net_data["conv4"][0])
		conv4b = tf.constant(net_data["conv4"][1])
		conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv4 = tf.nn.relu(conv4_in)


		#conv5
		#conv(3, 3, 256, 1, 1, group=2, name='conv5')
		k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv5W = tf.constant(net_data["conv5"][0])
		conv5b = tf.constant(net_data["conv5"][1])
		conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv5 = tf.nn.relu(conv5_in)

		#maxpool5
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

		#fc6
		#fc(4096, name='fc6')
		fc6W = tf.constant(net_data["fc6"][0])
		fc6b = tf.constant(net_data["fc6"][1])
		fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

		#fc7
		#fc(4096, name='fc7')
		fc7W = tf.constant(net_data["fc7"][0])
		fc7b = tf.constant(net_data["fc7"][1])
		fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

		#fc8
		#fc(1000, relu=False, name='fc8')
		fc8W = tf.constant(net_data["fc8"][0])
		fc8b = tf.constant(net_data["fc8"][1])
		fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
		# chosen = tf.zeros_like(fc7)

		return fc7,fc6

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
