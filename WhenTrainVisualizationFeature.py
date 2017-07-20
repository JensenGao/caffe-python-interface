# -*- coding: utf-8 -*-

Spyder Editor

This is a temporary script file.
"""

import caffe
import matplotlib.pyplot as plt     
import numpy as np

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    if data.shape[2] == 1:
        data = data[:,:,0]
    plt.imshow(data); plt.axis('off')

if __name__ == '__main__':
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('E:/caffe-master/examples/mnist/netpara/lenet_solver.prototxt')
    solver.step(1)
    input_data = solver.net.blobs['data'].data  
    plt.figure(0)
    vis_square(input_data.transpose(0, 2, 3, 1))  
    filters = solver.net.params['conv1'][0].data
    plt.figure(1)
    vis_square(filters.transpose(0, 2, 3, 1))
