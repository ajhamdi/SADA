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
from tqdm import trange
from itertools import chain
from collections import deque
from distutils.dir_util import copy_tree
import shutil
from scipy.sparse import lil_matrix
import scipy 
import matplotlib.pyplot as plt
from scipy import signal
import hdf5storage
import cv2
from utils import *
from models import *
from scipy.linalg import circulant
from scipy.linalg import dft
import scipy.io as sio
import tensorflow as tf
from skimage.util.shape import view_as_blocks
# def unroll_kernel(kernel, n, sparse=True):

#     m = kernel.shape[0]
#     if sparse:
#          unrolled_K = lil_matrix(((n - m)**2, n**2))
#     else:
#          unrolled_K = np.zeros(((n - m)**2, n**2))

#     skipped = 0
#     for i in range(n ** 2):
#          if (i % n) < (n - m) and((i / n) % n) < (n - m):
#              for j in range(m):
#                  for l in range(m):
#                     unrolled_K[i - skipped, i + j * n + l] = kernel[j, l]
#          else:
#              skipped += 1
#     return unrolled_K
# def unroll_matrix(X, m):
#   flat_X = X.flatten()
#   n = X.shape[0]
#   unrolled_X = np.zeros(((n - m) ** 2, m**2))
#   skipped = 0
#   for i in range(n ** 2):
#       if (i % n) < n - m and ((i / n) % n) < n - m:
#           for j in range(m):
#               for l in range(m):
#                   unrolled_X[i - skipped, j * m + l] = flat_X[i + j * n + l]
#       else:
#           skipped += 1
#   return unrolled_X

# def crop_filtered_patch(patch,skip=0,c_dim=1):
#   size = patch.shape[0]
#   cropped_patch = np.zeros((size-(2*skip),size-(2*skip)))
#   for ii in range(skip,size-skip):
#     cropped_patch[ii-skip,ii-skip] = patch[ii,ii]
#   return cropped_patch


n= 129    # the image shape
m = 5  # the kernel shape
RANDOM_NO = np.abs(np.random.random_integers(1000))
LOAD_MAT = True
skip = int(np.abs(n-m)/2)
path = "D:\\mywork\\sublime\\GAN2\\data\\celebB"
images = read_images_to_np(path,n,n,extension="all",allowmax=True,maxnbr=1000,d_type=np.float32,mode="RGB")
gray_images  = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
plt.figure()
plt.imshow(gray_images[RANDOM_NO],cmap='gray')
plt.title("Filtering by matrix multiplicatin")
plt.show(block=False)
print("finished loading the images\n")
if LOAD_MAT:
  mat = hdf5storage.loadmat('filtering.mat')
  k_padded = mat["k_padded"]
  A= mat["A_unfolded"]
  K = mat["kernel"]
  print("the Kernel\n",K,"\n\n")
  # print("the Kernel padded \n",k_padded,"\n\n")
  # k_padded = sio.loadmat("filtering.mat")["kernel"]
  # A = sio.loadmat("filtering.mat")["A_unfolded"]
  X = gray_images[RANDOM_NO]
  x = X.reshape(-1,1)
   

# K = np.array(range(1,m**2+1)).reshape(m,m)
else:
  K = np.ones((m,m),np.float32)
  k_padded = np.pad(K, ((skip,skip),(skip,skip)), 'constant')
  print("the Kernel\n",K,"\n\n")
  print("the Kernel padded \n",k_padded,"\n\n")
  A = circulant(k_padded.reshape(1,-1))
  print("the shape of amtrix A : ", A.shape)
  print("The Matrix A ... \n",A,"\n\n")
  X = np.array(range(1,n**2+1)).reshape(n,n)
  x = X.reshape(-1,1)


DFT = dft(n**2)

print("the image\n",X,"\n\n")
print("the image vectorized\n",x,"\n\n")


# conv_result_cropped = crop_filtered_patch(conv_result,skip)
# img = cv2.imread('opencv_logo.png')
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,kernel)
# conv_result = cv2.filter2D(img,-1,K)


# A = unroll_kernel(K,n,False)


conv_result = signal.convolve2d(X, K, boundary='wrap', mode='same')
plt.figure()
plt.imshow(conv_result,cmap='gray')
plt.title("Filtering by matrix Convlution")
plt.show(block=False)
conv_as_matmult_result = np.matmul(A, x)
reconstructed_image = conv_as_matmult_result.reshape(n,n) # the image is blockwise-ill 
blocked = view_as_blocks(reconstructed_image[:-1,:-1],(np.floor(n/2).astype(int),np.floor(n/2).astype(int)))
print(blocked.shape)
left = np.concatenate((blocked[1,1,:,:], blocked[0,1,:,:]), axis=0)
right = np.concatenate((blocked[1,0,:,:], blocked[0,0,:,:]), axis=0)
reconstructed_image_adjusted = np.concatenate((left, right), axis=1)
# blocked[0,0,:,:] , blocked[1,1,:,:] = blocked[1,1,:,:] , blocked[0,0,:,:]
# blocked[1,0,:,:] , blocked[0,1,:,:] = blocked[0,1,:,:] , blocked[1,0,:,:]
# blocks_lsit = [blocked[i,j,:,:] for i in  range(2) for  j in range(2)  ]
 
# for i in range(4):
  # plt.figure()
  # plt.imshow(blocks_lsit[i],cmap='gray')
  # plt.title("Filtering by matrix multiplicatin")
  # plt.show(block=False)

# reconstructed_image_adjusted = blocked.reshape(n-1,n-1) #  (np.floor(n/2)**2).astype(int)
# print("the result of convolution\n",conv_result,"\n")
plt.figure()
plt.imshow(reconstructed_image_adjusted,cmap='gray')
plt.title("Filtering by matrix multiplicatin")
plt.show()
# print("the result of matrix-product convolution\n",reconstructed_image,"\n")
error =  scipy.linalg.norm(reconstructed_image_adjusted - resize_image(conv_result, n-1, n-1,resize_mode="crop"))
print("\n\n the difference is ",error)