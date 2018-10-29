from __future__ import print_function

import os
import math
import json
import random
import logging
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from datetime import datetime
import scipy
from numbers import Number
from subprocess import Popen, PIPE


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

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    # for path in [config.log_dir, config.data_dir, config.model_dir]:
    #     if not os.path.exists(path):
    #         os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def forward_transform(image):
    return np.array(image)/127.5 - 1.

def inverse_transform(images):
    return np.floor(127.5*(images+1.)).astype(np.uint8)
def save_config(config,trainer):
    param_path = os.path.join(trainer.model_dir, "params.json")

    print("[*] MODEL dir: %s" % trainer.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
                normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
                 normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx % size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path,is_all=False):
    if is_all:
        for ii in range(images.shape[0]):
            filename, file_extension = os.path.splitext(path)
            subimage_name = filename + '_P{}'.format(ii) + file_extension
            scipy.misc.imsave(subimage_name, images[ii,:,:,:])
    else:
        return scipy.misc.imsave(path, merge(images, size))

def resize_image(image, height, width,channels=None,resize_mode=None):
    """
    Resizes an image and returns it as a np.array
    Arguments:
    image -- a PIL.Image or numpy.ndarray
    height -- height of new image
    width -- width of new image
    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    """
    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)
 
    if channels not in [None, 1, 3]:
        raise ValueError('unsupported number of channels: %s' % channels)
 
        image = np.array(image)
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.reshape(image.shape[:2])
        if channels is None:
            if image.ndim == 2:
                channels = 1
            elif image.ndim == 3 and image.shape[2] == 3:
                channels = 3
            else:
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 1:
            if image.ndim != 2:
                if image.ndim == 3 and image.shape[2] == 3:
                    # color to grayscale
                    image = np.dot(image, [0.299, 0.587, 0.114]).astype(np.uint8)
                else:
                    raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 3:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 3).reshape(image.shape + (3,))
            elif image.shape[2] != 3:
                raise ValueError('invalid image shape: %s' % (image.shape,))
    else:
        raise ValueError('resize_image() expected a PIL.Image.Image or a numpy.ndarray')
 
    # No need to resize
    if image.shape[0] == height and image.shape[1] == width:
        return image
 
    # Resize
    interp = 'bilinear'
 
    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if resize_mode == 'squash' or width_ratio == height_ratio:
        return cv2.resize(image, (height, width))
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:

            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = cv2.resize(image, ( resize_width,resize_height))
 
        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width - width) / 2.0))
            return image[:, start:start + width,:]
        else:
            start = int(round((resize_height - height) / 2.0))
            return image[start:start + height, :,:]
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = cv2.resize(image, ( resize_width,resize_height))
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = cv2.resize(image, ( resize_width,resize_height))
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width - width) / 2.0))
                image = image[:, start:start + width]
            else:
                start = int(round((resize_height - height) / 2.0))
                image = image[start:start + height, :]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)
 
        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = (height - resize_height) / 2
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = (width - resize_width) / 2
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)
 
        return image


def read_images_to_np(path,h,w,extension="all",allowmax=False,maxnbr=0,d_type=np.float32,mode="BGR",normalize=False):
    images = []
    for root, dirnames, filenames in os.walk(path):
        exit_subdir = False
        indices_missing = []
        indx = 0
        max_nbr_pictures = maxnbr
        for filename in filenames:
            if not exit_subdir and (((extension is not "all") and filename.lower().endswith("."+extension) ) or extension is "all") :
                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath)
            if image is None :
                indices_missing.append(indx)
                indx += 1
                continue
            if mode is "RGB":
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image_resized = resize_image(image, h, w,resize_mode="crop") 
            if normalize:
                images.append(forward_transform(image_resized))
            else:
                images.append(image_resized)
            if allowmax:
                max_nbr_pictures = max_nbr_pictures-1
                if max_nbr_pictures is 0:
                    exit_subdir = True
            indx += 1

    print("Finished reading the %d images ..."%(len(images)))
    return images , indices_missing


def my_read_images(path,h,w,expected_number=0,extension="jpg",d_type=np.float32,normalize=False):
    images = []
    indices_missing = []
    file_list = sorted(os.listdir(path))
    images_list = [item for item in file_list if item.endswith('.'+extension)]
    images_numbers = [int(os.path.splitext(img)[0]) for img in images_list]
    if not expected_number:
        expected_number = len(images_numbers)
    expected_numbers = list(range(expected_number))
    indx = 0
    for img in images_list:
        img_name = os.path.join(path, img)
        image = cv2.imread(img_name)
        if image is None :
            indices_missing.append(images_numbers[indx])
            indx += 1
            continue
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image_resized = resize_image(image, h, w,resize_mode="crop") 
        if normalize:
            images.append(forward_transform(image))
        else:
            images.append(image)
        indx += 1
    print("Finished reading the %d images ...missing: %d images "%(len(images),expected_number-len(images)))
    indices_missing += [x for x in expected_numbers if x not in images_numbers]
    return images , indices_missing




def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (X_imgs[0].shape[0], X_imgs[0].shape[0], 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    # X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, clr = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for X_img in X_imgs:
        gaussian = np.random.normal(0, sigma, (row,col,clr)).astype(np.float32)
        gaussian_img = cv2.addWeighted(X_img.astype(np.float32), 0.75, gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    # gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs
    
# https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

def sample_batch(all_list,batch_size):
    indices = np.random.choice(len(all_list), batch_size)
    return [all_list[i] for i in indices ] , indices

class Blender(object):
    def __init__(self,cluster, py_file, blend_file=None, *args):
        self._args = list(args)
        self.cluster = cluster
        self._blend_file = blend_file
        self._commands = []
        self.relative_path = "../bin/blender_folder/"

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
        if not self.cluster :
            cmd = ['blender', '--background', '--python-console'] + self._args
        else :
            cmd = ['./%sblender'%(self.relative_path), '--background', '--python-console'] + self._args
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

