# -*- coding: utf-8 -*-

import time
start=time.clock()

import matplotlib.pyplot as plt
import caffe
caffe.set_device(0)#
caffe.set_mode_gpu()
solver=caffe.SGDSolver('G:/caffe-master/examples/mnist/net/lenet_solver.prototxt')
solver.net.blobs.items() #各个输入输出的blob
solver.net.params.items()#只有有参数的层（卷积层和全连接层）

solver.net.forward()
solver.test_nets[0].forward()
#test_nets[0]:当有多个测试网络时，用第一个

plt.figure(),plt.title('output')
plt.imshow(solver.net.blobs['data'].data[:8,0].transpose(1,0,2).reshape(28,8*28),cmap='gray')
plt.axis('off')#不显示坐标轴
#transpose:转换图像通道
#reshape：显示的图像的大小
#cmap='gray':以灰度图显示

solver.step(1) #表示SGD反向传播一个batch
plt.figure(),plt.title('params')
plt.imshow(solver.net.params['conv1'][0].diff[:,0].reshape(4,5,5,5).transpose(0,2,1,3).reshape(4*5,5*5),cmap='gray')
#显示conv1的参数（参数包括权重和偏置，这里可视化的是权重）的增量；
#可视化形式为：高为4*5，宽为5*5的图像，以灰度图显示

cost_time=time.clock()-start
print cost_time
